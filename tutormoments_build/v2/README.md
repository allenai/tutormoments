### Getting the data: the release build

The research scripts read a v2 release build: the benchmark labels (`benchmark/dev.jsonl`, `benchmark/test.jsonl`) and transcripts (`transcripts/<provider>.jsonl`). Which build is pinned in `release.json` beside this README. Fetch it from the `tutormoments` root:

```
export TUTORMOMENTS_RELEASE_ROOT=s3://<bucket>
tutormoments-build release pull
```

This puts the build in `temp/release/builds/<build_id>/`. Every file is checked against the build's manifest, and the manifest against the hash in `release.json`. Files a local build already has are hard-linked, not downloaded again. `--prune` deletes other local builds, and `--only benchmark/ transcripts/` fetches just what these scripts read.

- **New builds.** Moving to a new build is a PR that changes `release.json`. After pulling that change, the scripts stop with a message to run `release pull` until you've fetched the build.
- **Trying a build before its PR.** Point `TUTORMOMENTS_V2_RELEASE` at its folder. Outputs record that it wasn't the pinned build.
- **What every output records.** Excerpts and predictions carry a `release` block: the build ID, whether it is the pinned build, and the sha256 of each input read.

The splits are `dev` and `test`. `iteration` is retired: regenerate old excerpts, and rename old prediction files from `iteration.jsonl` to `dev.jsonl` (or regenerate them).

`build_ground_truth`, `splits`, `adjudicator_agreement` and `decompose_situation` still read the raw annotation exports for now (`data/v2_annotations/source/<provider>_annotations.jsonl` and `annotator_labels.json`). `classify_excerpts` reads the student-state facets from `data/ground_truth/student_state.jsonl`.

Excerpts exist because they format what's shown to LMs (the tutor and action classifiers): the entire transcript up to the moment to the moment's end. Once created, these are saved in `data/excerpts/`. 

```
python -m tutormoments_build.v2.excerpts
```

### Running and evaluating the scorer on human-human data

Example with one model: 

```
python -m tutormoments_build.v2.classify_excerpts --model claude-opus-5
python -m tutormoments_build.v2.evaluate_scorer --model claude-opus-5
```

Here, `evaluate_scorer` refers to evaluating the LM action classifier, not evaluating the LM tutor. 

### Replays? 

For replays, you'll want to format each replay in a similar manner as we format human-human data for evaluating our LM scorer, and classify the AI tutor's action similar to how we do it for humans in `classify_excerpts.py`. Sharing things between human-human data and replays could maybe be good for consistency. 

You can use the latest prompts in `tutormoments_build/prompts/v2/<version>/`, with `claude-opus-5` as the scorer when classifying tutor actions. When tracking experiments, keep track of both prompt and model versions! 