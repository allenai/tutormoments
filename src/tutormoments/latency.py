"""Latency probe: the reportable, cross-model-comparable TTFT/TTLT number.

Two metrics matter for tutoring, and neither is throughput:

- **TTFT** (time to first visible token) drives how responsive the tutor
  feels. Ed-tech partners consistently name this as the thing students react
  to.
- **TTLT** (time to last token) is when the student can actually reply.

Tutor turns are short, so tokens/sec is recorded as a diagnostic and never
surfaced as a headline.

Why a separate command rather than reading the numbers off a benchmark run:
a normal run replays moments through a thread pool (`--concurrency`, default
4), and concurrency distorts latency by a model-dependent amount. Requests
from one account share a decode batch, so per-stream throughput falls as
concurrency rises; and near rate limits providers queue at the edge before
prefill, inflating TTFT. Because rate-limit tiers differ per model, the
distortion differs per model too -- which is precisely what breaks
cross-model comparison. This probe runs strictly serially so the numbers mean
the same thing for every model.

Numbers are only comparable within one measurement environment: we cannot pin
an egress region the way Artificial Analysis pins GCP us-central1-a, so the
environment is recorded alongside every result instead.
"""

import datetime
import logging
import os
from pathlib import Path

from tutormoments import conversation, results
from tutormoments.moments import (
    PROBE_IDS_FILENAME,
    packaged_probe_ids,
    read_probe_ids,
    subsample_id,
)

logger = logging.getLogger(__name__)

LATENCY_FILENAME = "latency.json"

# Only used when the release carries no frozen id list; a frozen list always
# wins, since keeping the sample fixed is what makes latency a time series.
# Matches tutormoments_build.latency_subsample.DEFAULT_SUBSAMPLE_SIZE.
DEFAULT_SUBSAMPLE_SIZE = 112

# Fewer cache hits than this and there is no warm figure to report -- show it
# as unavailable rather than publishing a percentile over a handful of
# samples. Deliberately a *count*, not a share of all samples: the share is
# fixed by max_turns (exactly 0.5 at max_turns=3, 0.67 at 5), so a rate
# threshold would make publishability depend on a run knob rather than on
# whether the provider actually caches. Providers with no real prompt cache
# report `unknown` or zero hits and are excluded either way.
# See docs/latency.md ("Caching fidelity").
MIN_CACHE_HIT_SAMPLES = 10

# A "warm" turn is meant to be one where *this conversation's* transcript was
# served from cache. But `cached_tokens > 0` alone does not establish that:
# providers with automatic prefix caching report incidental hits on whatever
# prefix happens to be shared, which here is the system prompt every moment
# starts with. Measured: Anthropic's explicit breakpoint reads back a median
# of 7.4k-10k tokens (the actual conversation head), while Together reports a
# median of exactly 256 -- one quantised block of shared system prompt, with
# hits appearing on first turns and misses on later ones, i.e. tracking run
# warmup rather than session position.
#
# So qualify a warm figure on how *much* was read back. The threshold sits
# above any incidental block and below the smallest real head observed
# (1,180 tokens): every moment's head is system prompt + pre-cut transcript,
# which always exceeds this.
MIN_SESSION_CACHE_READ_TOKENS = 1024


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def latency_stats(samples: list[float]) -> dict | None:
    """Mean / p5 / p50 / p95 over per-call latency samples. None on empty.

    Sorted-index percentiles. The p50/p95 rule is carried over verbatim from
    the pre-streaming implementation so previously published figures stay
    reproducible; p5 follows the same rule. P5/P50/P95 matches the percentile
    set Artificial Analysis reports, so our figures line up with the shape
    readers already know.
    """
    if not samples:
        return None
    s = sorted(samples)
    n = len(s)
    p50 = s[n // 2]
    p5_idx = max(0, min(n - 1, int(round(0.05 * n)) - 1))
    p95_idx = max(0, min(n - 1, int(round(0.95 * n)) - 1))
    total = sum(s)
    return {
        "n": n,
        "total_seconds": round(total, 3),
        "mean_seconds": round(total / n, 3),
        "p5_seconds": round(s[p5_idx], 3),
        "p50_seconds": round(p50, 3),
        "p95_seconds": round(s[p95_idx], 3),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_timings(timings: list[dict]) -> dict:
    """Aggregate per-call timings into the latency block.

    TTFT and TTLT are split by *observed* cache state rather than pooled.
    Pooling would make the figure drift with how many turns each conversation
    happened to run (``[END]`` truncates some early), and the two states are
    genuinely different student experiences: a miss is the first message of a
    session, a hit is every later message.

    `cache_hit_rate` is reported beside the numbers so a reader can tell a
    real warm figure from one computed over a sample that mostly missed.
    """
    known = [t for t in timings if t.get("cache_state") in ("hit", "miss")]
    hits = [t for t in timings if t.get("cache_state") == "hit"]
    hit_rate = (len(hits) / len(known)) if known else None

    def _split(field: str) -> dict:
        def _vals(rows):
            return [r[field] for r in rows if r.get(field) is not None]

        return {
            "hit": latency_stats(_vals(hits)),
            "miss": latency_stats(
                _vals([t for t in timings if t.get("cache_state") == "miss"])
            ),
            "all": latency_stats(_vals(timings)),
        }

    # Calls that emitted no visible token at all -- the model spent its whole
    # max_tokens budget thinking. They carry no TTFT by definition, so the
    # percentiles below are conditional on a turn having produced output.
    # That conditioning has to be visible: a model can look fast simply by
    # not answering, and in the benchmark these turns are silently replaced
    # with "..." and scored as if the tutor said that.
    no_output = [t for t in timings if t.get("ttft_seconds") is None]
    tps = [t["output_tps"] for t in timings if t.get("output_tps") is not None]
    # How much was actually served from cache on a hit -- the difference
    # between a genuine session cache and an incidental shared-prefix hit.
    reads = sorted(
        t["cache_read_input_tokens"]
        for t in hits
        if t.get("cache_read_input_tokens") is not None
    )
    return {
        "n_samples": len(timings),
        "cache_read_p50_on_hits": reads[len(reads) // 2] if reads else None,
        "n_no_visible_output": len(no_output),
        "no_visible_output_rate": (
            round(len(no_output) / len(timings), 3) if timings else None
        ),
        "cache_hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
        "cache_state_known": len(known),
        "ttft": _split("ttft_seconds"),
        "ttlt": _split("ttlt_seconds"),
        # Diagnostic: ttft - ttfc is roughly how long the model spent
        # thinking before the student saw anything. Aggregated in the same
        # cache-state split as the headline metrics so the "ttfc well below
        # ttft on thinking models" check in docs/latency.md can be read off
        # this block rather than recomputed from `samples`.
        "ttfc": _split("ttfc_seconds"),
        # Diagnostic only -- tutor turns are short, so throughput is not a
        # headline metric for this benchmark.
        "output_tps_mean": round(sum(tps) / len(tps), 2) if tps else None,
    }


def warm_figure_is_publishable(block: dict) -> bool:
    """Whether a warm (cache-hit) figure should be shown rather than dashed.

    Only the Anthropic path sends a real cache breakpoint. Gemini, Together
    and OpenAI concatenate the cacheable prefix into the prompt and rely on
    whatever automatic prefix caching the provider does, which is not this
    conversation's transcript in the general case. Publishing a "warm" number
    for those models would make them look slower -- or faster -- for a harness
    reason rather than a model reason.

    Three conditions, each ruling out a different way the number could lie:

    - the provider reports cache tokens at all;
    - enough hit samples to support a percentile (a count, not a *rate* --
      the rate is fixed by max_turns, so gating on it would let a run knob
      decide whether a model gets a published figure);
    - the hits actually served this conversation's transcript rather than an
      incidental shared prefix (see MIN_SESSION_CACHE_READ_TOKENS).
    """
    if block.get("cache_hit_rate") is None:
        return False  # provider reports no cache tokens at all
    hits = ((block.get("ttft") or {}).get("hit") or {}).get("n") or 0
    if hits < MIN_CACHE_HIT_SAMPLES:
        return False
    read = block.get("cache_read_p50_on_hits")
    return read is not None and read >= MIN_SESSION_CACHE_READ_TOKENS


# ---------------------------------------------------------------------------
# Reading probe results back (leaderboard + website join)
# ---------------------------------------------------------------------------


def probe_figures(block: dict) -> dict:
    """The publishable TTFT/TTLT p50s from one probe's ``latency.json``.

    Returns ``ttft_p50`` / ``ttlt_p50`` (pooled over all samples) plus the
    cold and warm halves, which are ``None`` unless the cache split can be
    trusted.

    Pooled is always populated: it is measured identically on every provider,
    whatever that provider reports about caching, so it is the only figure
    that ranks the whole roster. The split refines it where cache state is
    knowable.

    **The split is published as a unit or not at all.** The obvious reading is
    that `warm_figure_is_publishable` guards only the warm number, but its
    three conditions establish that this provider's hit/miss labels mean
    session warmth at all -- and a label that cannot be trusted for the hits
    cannot be trusted for their complement either. Concretely: Together's
    "misses" are the calls its automatic prefix cache happened not to serve,
    which cluster at run start rather than at session start, and on the pilot
    that put its cold p50 at 15.03s against a pooled 7.94s. Publishing that as
    "first message of a session" would be wrong in the same way publishing its
    warm figure would be.

    The probe's own terminal summary is deliberately looser -- it prints the
    cold figure with a NOTE explaining why warm was withheld, because there it
    is a diagnostic read by whoever just ran the probe. These figures go into
    the leaderboard and onto the website, where they are read without that
    context.
    """
    ttft = block.get("ttft") or {}
    ttlt = block.get("ttlt") or {}

    def _p50(metric: dict, state: str):
        return (metric.get(state) or {}).get("p50_seconds")

    split_ok = warm_figure_is_publishable(block)
    return {
        "ttft_p50": _p50(ttft, "all"),
        "ttlt_p50": _p50(ttlt, "all"),
        "ttft_cold_p50": _p50(ttft, "miss") if split_ok else None,
        "ttft_warm_p50": _p50(ttft, "hit") if split_ok else None,
    }


def probe_runs(results_root: str = "results") -> dict:
    """Latest comparable probe result per ``(tutor_model, mode)``.

    Scans *results_root* for run directories carrying a ``latency.json`` and
    returns ``{(tutor_model, mode): block}``. This is the join the leaderboard
    and the website both need: a probe writes its own run directory, which
    nothing else reads.

    Only probes that measured a **frozen** subsample **in full** are eligible.
    A derived sample spans no particular prompt-length distribution and is
    comparable to nothing; an incomplete one dropped ids, so it is not the
    same sample as the run before it. Neither belongs in a table that invites
    cross-model comparison.

    Among eligible probes for one cell the newest by ``measured_at`` wins, so
    re-measuring a model supersedes its earlier figure without anyone having
    to delete the old run. Callers that publish several cells should check the
    selected blocks agree on ``subsample_id``: eligibility is per-probe, and
    two frozen-but-different samples would each pass it (see
    `probe_subsample_ids`).
    """
    out: dict[tuple[str, str], dict] = {}
    best_key: dict[tuple[str, str], tuple] = {}

    for run_id in results.list_runs(results_root):
        block = results.read_latency(run_id, results_root=results_root)
        if not block:
            continue
        sub = block.get("subsample") or {}
        if not str(sub.get("subsample_source", "")).startswith("frozen"):
            continue
        if not sub.get("subsample_complete", False):
            continue
        cell = (block.get("tutor_model", ""), block.get("mode", ""))
        env = block.get("measurement_environment") or {}
        # measured_at then run_id: a run directory written without a timestamp
        # still orders deterministically instead of depending on scan order.
        rank = (env.get("measured_at") or "", run_id)
        if cell not in best_key or rank > best_key[cell]:
            best_key[cell] = rank
            out[cell] = block
    return out


def probe_subsample_ids(probes: dict) -> set:
    """The distinct ``subsample_id`` values across selected probe blocks.

    More than one means the figures were measured over different prompt sets
    and must not be printed in one table -- the point of the id is that this
    is visible rather than quietly averaged.
    """
    return {
        (block.get("subsample") or {}).get("subsample_id") for block in probes.values()
    }


# ---------------------------------------------------------------------------
# Subsample resolution
# ---------------------------------------------------------------------------


def resolve_subsample(
    moments: list, n: int, data_path: str | None
) -> tuple[list, dict]:
    """Pick the moments to measure, preferring a frozen list.

    The frozen list is what makes latency a time series: a later release that
    is a superset of this one still resolves every id, so measurements taken
    months apart are over the same prompts. Deriving a sample at run time
    instead would silently re-pick it whenever the dataset changed.

    Resolution order:

    1. ``latency_probe_ids.json`` in the release directory -- a dataset's own
       statement about itself outranks anything shipped with the code.
    2. The list packaged with the runtime. This is the path most runs take:
       the default config loads moments from the published Hugging Face
       dataset, where there is no local release directory to read.
    3. Derivation, which is not comparable to anything.

    Returns (moments, provenance). Provenance always records which path was
    taken so a derived sample can never be mistaken for a frozen one.
    """
    by_id = {m.id: m for m in moments}

    frozen = read_probe_ids(data_path) if data_path else None
    source = "frozen_release"
    if not frozen:
        frozen = packaged_probe_ids()
        source = "frozen_packaged"

    if frozen:
        present = [by_id[i] for i in frozen if i in by_id]
        missing = [i for i in frozen if i not in by_id]
        if missing:
            logger.warning(
                "%d frozen probe id(s) are absent from this dataset; the "
                "sample is not comparable to earlier runs (first: %s)",
                len(missing),
                missing[0],
            )
        return present, {
            "subsample_source": source,
            "subsample_id": subsample_id(frozen),
            "subsample_complete": not missing,
            "missing_ids": missing,
            "n_requested": len(frozen),
        }

    # Neither source available -- fall back to the first n in released order
    # and say so loudly. This sample spans no particular prompt-length
    # distribution and must not be compared against frozen runs.
    derived = [m.id for m in moments[:n]]
    logger.warning(
        "No %s in the release or the package; deriving a subsample of %d. "
        "These numbers are NOT comparable to runs over the frozen subsample.",
        PROBE_IDS_FILENAME,
        len(derived),
    )
    return moments[:n], {
        "subsample_source": "derived",
        "subsample_id": subsample_id(derived),
        "subsample_complete": True,
        "missing_ids": [],
        "n_requested": n,
    }


# ---------------------------------------------------------------------------
# Measurement environment
# ---------------------------------------------------------------------------


def measurement_environment(*, package_version: str) -> dict:
    """Record where and how the measurement was taken.

    Artificial Analysis pins itself to GCP us-central1-a; we run wherever the
    user is, so instead of pretending to a fixed environment we record the one
    we had. Set TUTORMOMENTS_LATENCY_LOCATION to label it (e.g. "gcp-us-central1").
    """
    return {
        "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "concurrency": 1,
        "location": os.environ.get("TUTORMOMENTS_LATENCY_LOCATION") or None,
        "tutormoments_version": package_version,
    }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def run_probe(
    tutor: str,
    mode: str,
    *,
    cfg,
    n: int,
    results_root: str = "results",
    date: str | None = None,
    package_version: str = "",
    _run_conversation=None,
) -> tuple[str, dict]:
    """Measure TTFT/TTLT for one tutor, strictly serially.

    Returns (run_id, latency_block) and writes results/<run_id>/latency.json.

    No thread pool, deliberately: see the module docstring. Reuses
    run_conversation unchanged so the probe measures exactly the prompts the
    benchmark uses -- a latency number over a synthetic prompt would not
    describe this benchmark.
    """
    from tutormoments.moments import load_moments

    run_conv = _run_conversation or conversation.run_conversation
    date = date or datetime.date.today().strftime("%Y%m%d")

    all_moments, _ = load_moments(
        dataset=cfg.dataset,
        data_path=cfg.data_path,
        revision=cfg.dataset_revision,
        config=cfg.dataset_config,
    )
    subsample, provenance = resolve_subsample(all_moments, n, cfg.data_path)
    if not subsample:
        raise RuntimeError(
            "Latency probe resolved zero moments; check --data_path and the "
            f"release's {PROBE_IDS_FILENAME}."
        )

    run_id = results.make_run_id(tutor, mode, "latency", date)
    logger.info(
        "Latency probe: tutor=%s mode=%s moments=%d (serial)",
        tutor,
        mode,
        len(subsample),
    )

    tutor_timings: list[dict] = []
    student_timings: list[dict] = []
    failed: list[str] = []

    for i, moment in enumerate(subsample, 1):
        logger.info("[%d/%d] %s", i, len(subsample), moment.id)
        try:
            transcript = run_conv(
                moment,
                tutor_id=tutor,
                tutor_mode=mode or None,
                student_id=(cfg.student or {}).get("model"),
                student_mode=(cfg.student or {}).get("mode", "oracle"),
                max_turns=cfg.max_turns,
            )
        except Exception as e:  # noqa: BLE001 -- one bad moment must not
            # discard an otherwise complete measurement run.
            logger.warning("latency probe failed on %s: %s", moment.id, e)
            failed.append(moment.id)
            continue
        for t in getattr(transcript, "tutor_timings", []) or []:
            tutor_timings.append({**t, "moment_id": moment.id})
        for t in getattr(transcript, "student_timings", []) or []:
            student_timings.append({**t, "moment_id": moment.id})

    block = {
        "source": "probe",
        "tutor_model": tutor,
        "mode": mode,
        "tutor": aggregate_timings(tutor_timings),
        "student": aggregate_timings(student_timings),
        "subsample": provenance,
        "failed_moments": failed,
        "measurement_environment": measurement_environment(
            package_version=package_version
        ),
        "samples": tutor_timings,
    }

    results.write_latency(run_id, block, results_root=results_root)
    logger.info("Wrote %s", Path(results_root) / run_id / LATENCY_FILENAME)
    return run_id, block


def withheld_reason(block: dict) -> str:
    """Why the warm figure is dashed, tested in the same order as the gate.

    The order matters. A provider that reports cache tokens but recorded no
    hits has `cache_read_p50_on_hits` of None, so testing the read size first
    reports "hits read only None tokens" -- a cache *fidelity* problem, when
    the actual problem is that there were no hits to judge the fidelity of.
    Mirror `warm_figure_is_publishable` exactly: provider, then count, then
    what the hits actually read back.
    """
    if block.get("cache_hit_rate") is None:
        return "provider reports no cache tokens"
    hits = ((block.get("ttft") or {}).get("hit") or {}).get("n") or 0
    if hits < MIN_CACHE_HIT_SAMPLES:
        return f"only {hits} cache hit(s), need {MIN_CACHE_HIT_SAMPLES}"
    return (
        f"hits read only {block.get('cache_read_p50_on_hits')} tokens "
        "(incidental shared prefix, not this conversation)"
    )


def format_probe_summary(block: dict) -> str:
    """Render a compact terminal summary of a probe result."""
    t = block.get("tutor") or {}
    sub = block.get("subsample") or {}
    env = block.get("measurement_environment") or {}

    def _p50(metric, state):
        stats = ((t.get(metric) or {}).get(state)) or {}
        v = stats.get("p50_seconds")
        return f"{v:.3f}" if isinstance(v, (int, float)) else "-"

    warm_ok = warm_figure_is_publishable(t)
    lines = [
        f"Latency probe: {block.get('tutor_model', '-')} / {block.get('mode') or '-'}",
        "  " + "-" * 44,
        # Always shown: measured identically on every provider, whatever its
        # cache reporting. The cold/warm split below refines it where cache
        # state is knowable, but must never be the only thing on offer -- a
        # provider that reports no cache tokens still has a real distribution.
        f"  {'TTFT p50 all (s)':<26} {_p50('ttft', 'all')}",
        f"  {'TTLT p50 all (s)':<26} {_p50('ttlt', 'all')}",
        f"  {'TTFT p50 cold/warm (s)':<26} "
        f"{_p50('ttft', 'miss')} / {_p50('ttft', 'hit') if warm_ok else '-'}",
        f"  {'TTLT p50 cold/warm (s)':<26} "
        f"{_p50('ttlt', 'miss')} / {_p50('ttlt', 'hit') if warm_ok else '-'}",
        f"  {'Samples':<26} {t.get('n_samples', 0)}",
        f"  {'No visible output':<26} "
        f"{t.get('n_no_visible_output', 0)}/{t.get('n_samples', 0)}"
        + (
            "  <-- hit max_tokens while thinking; these turns are"
            " recorded as '...' in a benchmark run"
            if t.get("n_no_visible_output")
            else ""
        ),
        f"  {'Cache hit rate':<26} "
        + (
            f"{t['cache_hit_rate']:.2f}"
            if t.get("cache_hit_rate") is not None
            else "unknown (provider reports no cache tokens)"
        ),
        f"  {'Subsample':<26} "
        f"{sub.get('subsample_source', '-')} ({sub.get('subsample_id', '-')})",
    ]
    if not sub.get("subsample_complete", True):
        lines.append(
            f"  {'WARNING':<26} {len(sub.get('missing_ids') or [])} frozen id(s) "
            "missing -- not comparable to earlier runs"
        )
    if not warm_ok:
        lines.append(f"  {'NOTE':<26} warm figure withheld: {withheld_reason(t)}")
    if env.get("location"):
        lines.append(f"  {'Location':<26} {env['location']}")
    return "\n".join(lines)
