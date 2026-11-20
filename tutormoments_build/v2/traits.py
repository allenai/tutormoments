"""Generate the frozen per-moment student personas v2 replays read.

Reuses v1's trait machinery (``tutormoments_build.traits.TraitGenerator``,
``prompts/trait_generator/``, mode ``joined-3``: one 3-sentence description per
dimension, joined) on each moment's pre-cut dialogue. Differences from v1:
calls pass ``max_tokens=0`` (v1's adapter capped at 1024; the benchmark imposes
no cap), every call's token usage is recorded, the description is taken after
the *last* ``DESCRIPTION:`` marker rather than the first
(``LastMarkerTraitGenerator``), and each dimension's raw output is kept.

Personas are generated once and frozen: a replay reads them and raises on a
missing one rather than generating its own. The file is resumable -- moments
already in it are skipped -- and only ``--overwrite`` regenerates.

Usage:
    python -m tutormoments_build.v2.traits --dry-run
    python -m tutormoments_build.v2.traits --limit 2
    python -m tutormoments_build.v2.traits --print 02e89625
"""

import argparse
import datetime
import logging
import random
import sys

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments.usage import add_usage
from tutormoments_build.traits import (
    ALL_DIMENSIONS,
    DEFAULT_TRAIT_MODE,
    TraitGenerator,
)
from tutormoments_build.v2 import excerpts, jsonl_runs, replay_config, replay_data

logger = logging.getLogger("tutormoments_build.v2.traits")

DEFAULT_OUT = "data/ground_truth/v2_student_traits.jsonl"


class UsageRecordingAdapter:
    """``TraitGenerator``'s model interface, uncapped and usage-recording.

    ``TraitGenerator`` calls ``call(non_system_messages, system_prompt, ...)``
    and wants text back; this forwards each call to ``client.generate`` with the
    system prompt as the cacheable prefix, as v1's adapter does, and sums the
    usage of every call it makes.
    """

    def __init__(self, client, thinking: dict) -> None:
        self._client = client
        self._thinking = thinking
        self.usage: dict = {}
        self.calls = 0

    def call(self, non_system_messages, system_prompt, **_ignored) -> str:
        # TraitGenerator's max_tokens/temperature are ignored: no output cap,
        # and v1's adapter never forwarded temperature either.
        user_text = "\n\n".join(
            m.get("content", "")
            for m in (non_system_messages or [])
            if m.get("role") in (None, "user")
        )
        response = self._client.generate(
            user_text,
            json_mode=False,
            max_tokens=0,
            thinking=self._thinking,
            cacheable_prefix=system_prompt or None,
        )
        add_usage(self.usage, response.usage)
        self.calls += 1
        return response.text or ""


MARKER = "DESCRIPTION:"


class LastMarkerTraitGenerator(TraitGenerator):
    """v1's generator, parsing on the last ``DESCRIPTION:`` and keeping raw text.

    The prompt asks for visible step-by-step thinking before ``DESCRIPTION:``.
    v1 keeps everything after the *first* marker, so a model that quotes the
    required format while planning (``DESCRIPTION:\\n[description]``) has its
    whole plan kept as the persona. The final description always follows the
    last marker. Each dimension's raw output is kept so parsing can be redone
    from the file without another call.
    """

    def __init__(self, model) -> None:
        super().__init__(model)
        self.raw_outputs: dict[str, str] = {}
        self._last_raw = ""

    def parse_trait_output(self, raw_output: str) -> tuple:
        self._last_raw = raw_output
        if MARKER in raw_output:
            return raw_output.rsplit(MARKER, 1)[1].strip(), raw_output
        return raw_output.strip(), raw_output

    def generate_trait(self, conversation_text, trait_mode, **kwargs) -> str:
        description = super().generate_trait(
            conversation_text=conversation_text, trait_mode=trait_mode, **kwargs
        )
        base, _ = self.parse_trait_type(trait_mode)
        if base in ALL_DIMENSIONS:
            if not description:
                raise ValueError(f"empty {base} description")
            self.raw_outputs[base] = self._last_raw
        return description


def generate_persona(
    prefix: str, client, thinking: dict, trait_mode: str = DEFAULT_TRAIT_MODE
) -> tuple[str, dict, int, dict]:
    """``(persona, summed usage, call count, raw output by dimension)``."""
    adapter = UsageRecordingAdapter(client, thinking)
    generator = LastMarkerTraitGenerator(adapter)
    persona = generator.generate_trait(conversation_text=prefix, trait_mode=trait_mode)
    return persona.strip(), adapter.usage, adapter.calls, generator.raw_outputs


def build_record(moment, persona, usage, calls, raw, spec, trait_mode) -> dict:
    return {
        "moment_id": moment.moment_id,
        "transcript_id": moment.moment["transcript_id"],
        "cut_turn": moment.moment["cut_turn"],
        "persona": persona,
        "trait_mode": trait_mode,
        "parse": f"text after the last {MARKER!r}",
        "raw_outputs": raw,
        "generator_model": spec.model,
        "thinking": spec.thinking,
        "calls": calls,
        "usage": usage,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    }


def load_traits(path: str = DEFAULT_OUT) -> dict[str, dict]:
    """``{moment_id: trait record}``; empty when the file does not exist."""
    return jsonl_runs.load_by_id(path)


def persona_for(traits: dict[str, dict], moment_id: str) -> str:
    """The frozen persona for a moment; raises rather than generating one."""
    record = traits.get(moment_id)
    if not record or not record.get("persona"):
        raise KeyError(
            f"no frozen persona for moment {moment_id}; run "
            "`python -m tutormoments_build.v2.traits` first"
        )
    return record["persona"]


def generate(
    moments,
    out_path: str,
    *,
    client,
    spec,
    overwrite: bool = False,
    concurrency: int = 4,
    trait_mode: str = DEFAULT_TRAIT_MODE,
) -> list[dict]:
    """Generate personas for ``moments`` missing from ``out_path``.

    Written as they are made and resumable; ``overwrite`` regenerates the given
    moments and replaces the file only once all succeed (see ``jsonl_runs``).
    """

    def one(moment):
        prefix = replay_data.dialogue_prefix(moment)
        persona, usage, calls, raw = generate_persona(
            prefix, client, spec.thinking, trait_mode
        )
        return build_record(moment, persona, usage, calls, raw, spec, trait_mode)

    return jsonl_runs.run(
        moments, out_path, one, overwrite=overwrite, concurrency=concurrency
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.traits",
        description="Generate frozen student personas for v2 replays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument(
        "--ground-truth", default=excerpts.DEFAULT_GROUND_TRUTH_DIR, metavar="DIR"
    )
    parser.add_argument(
        "--transcripts",
        nargs="+",
        default=list(excerpts.DEFAULT_TRANSCRIPTS),
        metavar="FILE",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, metavar="FILE")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Generate for at most N moments still missing a persona",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Generate for a seeded random N of the moments still missing a "
        "persona (file order clusters by provider and transcript)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4, metavar="N")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate personas already in the file",
    )
    parser.add_argument(
        "--print",
        dest="print_moment",
        metavar="MOMENT_ID",
        help="Print one moment's persona source (id prefix accepted) and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without any API call"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    spec = replay_config.role_spec("traits")
    moments, dropped = replay_data.load_moments(args.ground_truth, args.transcripts)

    if args.print_moment:
        for moment in moments:
            if moment.moment_id.startswith(args.print_moment):
                print(replay_data.dialogue_prefix(moment))
                return 0
        print(f"no moment matching {args.print_moment!r}", file=sys.stderr)
        return 1

    existing = load_traits(args.out)
    todo = [m for m in moments if args.overwrite or m.moment_id not in existing]
    if args.sample is not None:
        todo = random.Random(args.seed).sample(todo, min(args.sample, len(todo)))
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f"traits: {spec.model} {spec.thinking}, mode {DEFAULT_TRAIT_MODE}\n"
        f"  moments: {len(moments)} located"
        + (f", dropped {dict(dropped)}" if dropped else "")
        + f"\n  already in {args.out}: "
        f"{sum(m.moment_id in existing for m in moments)}\n"
        f"  to generate: {len(todo)}"
    )
    if args.dry_run or not todo:
        return 0

    from tutormoments.client import get_client

    written = generate(
        todo,
        args.out,
        client=get_client(spec.model),
        spec=spec,
        overwrite=args.overwrite,
        concurrency=args.concurrency,
    )
    usage: dict = {}
    for record in written:
        add_usage(usage, record["usage"])
    print(
        f"  wrote {len(written)} persona(s) to {args.out}\n"
        f"  tokens: {usage.get('input_tokens', 0)} in / "
        f"{usage.get('output_tokens', 0)} out / {usage.get('total_tokens', 0)} total"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
