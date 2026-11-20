### Putting together teacher-annotated human-human data

Data is first pulled from AWS. From the `tutormoments` root:  

```
cd data/v2_annotations/source
aws s3 cp s3://<bucket>/v2_annotations/source/tutoring_provider_a_annotations.jsonl ./
aws s3 cp s3://<bucket>/v2_annotations/source/tutoring_provider_c_annotations.jsonl ./
aws s3 cp s3://<bucket>/source/tutoring_provider_a.jsonl ./
aws s3 cp s3://<bucket>/source/tutoring_provider_c.jsonl ./
```

Two files that are not in the repo also go in `data/v2_annotations/`, both keyed on `annotator_id`: `annotator_labels.json` (id → de-identified label) and `excluded_annotators.json` (a list of project staff ids to drop).

Ground truth files are also on s3: 

```
cd data/ground_truth
aws s3 cp s3://<bucket>/synthetic/ground_truth_v2/*.jsonl ./ 
```

Excerpts exist because they format what's shown to LMs (the tutor and action classifiers): the entire transcript up to the moment to the moment's end. Once created, these are saved in `data/excerpts/`. 

```
python -m tutormoments_build.v2.excerpts
```

### Running and evaluating the scorer on human-human data

Example with one model: 

```
python -m tutormoments_build.v2.classify_excerpts --model gemini-4-argon
python -m tutormoments_build.v2.evaluate_scorer --model gemini-4-argon
```

Here, `evaluate_scorer` refers to evaluating the LM action classifier, not evaluating the LM tutor. 

### Replays

A replay starts from a ground-truth moment's cut point. An LM tutor (a `benchmark_models` arm) continues the conversation, a simulated student replies between tutor turns, and the result is classified with the same v2 scorer prompts as the human-human excerpts. Both ground-truth splits are replayed. Models for the student, scorer and persona roles are in the `v2_replay` block of `src/tutormoments/default_config.yaml`.

**1. Student personas.** These are generated once per moment from the pre-cut dialogue and frozen in `data/ground_truth/v2_student_traits.jsonl`. They're only needed when the student speaks (`--turns` above 1). The run is resumable: moments already in the file are skipped.

```
python -m tutormoments_build.v2.traits --dry-run
python -m tutormoments_build.v2.traits --sample 20
python -m tutormoments_build.v2.traits
```

**2. Replays.** `--turns N` is the number of generated tutor turns; the replay always starts and ends on a tutor turn. The tutor sees the whole transcript up to and including the cut row, but not the cut marker. The student also sees the real post-cut turns through the moment's end, as a reference to imitate. Prompts are in `tutormoments_build/prompts/v2/{tutor,student}/<version>/`. The run is resumable, and `--overwrite` regenerates.

```
python -m tutormoments_build.v2.replays --tutor gemini-4-argon --dry-run
python -m tutormoments_build.v2.replays --tutor gemini-4-argon --show-prompt 02e8
python -m tutormoments_build.v2.replays --tutor gemini-4-argon --sample 5
python -m tutormoments_build.v2.replays --tutor gemini-4-argon --turns 3 --tutor-prompt scaffolding_rigor
python -m tutormoments_build.v2.replays --tutor gemini-4-argon --print 02e8
```

Output: `outputs/v2_replays/<tutor arm>/<tutor prompt>_v<n>/n<N>/student-<student model>_sp<n>.jsonl`.

**3. Classifying replays.** Each replay record holds excerpts shaped like the human ones: the lead-up, `>>> CUT POINT <<<`, then the generated turns. The lead-up at each width (`full`, `20`, `5`) is identical to the human excerpt's, and the default is 20 turns, as in `classify_excerpts`. Classification runs as two batches:

1. `action_direction` on every replay.
2. `over-scaffolding` only where stage 1 *predicted* scaffolding. `{action}` comes from that prediction; `{is_appropriate_for}` and `{descriptions}` are gold, since they describe the moment at the cut.

Replays that aren't sent to stage 2 record why ("not routed: predicted no scaffolding" or "not routed: action_direction unparseable"), so they're never counted as a "no".

```
python -m tutormoments_build.v2.classify_replays outputs/v2_replays/gemini-4-argon/plain_v1/n1/student-gemini-4-argon_sp1.jsonl --dry-run
python -m tutormoments_build.v2.classify_replays <replays.jsonl>
python -m tutormoments_build.v2.classify_replays <replays.jsonl> --context-turns full
python -m tutormoments_build.v2.classify_replays <replays.jsonl> --model claude-opus-5-5
```

Output sits beside the replay file: `…_sp<n>__scorer-<scorer model>_v<prompt version>_ctx<width>.jsonl`. If a run is interrupted, `--action-batch-id` / `--over-batch-id` resume its batches.

Every record stores the settings it was made with: tutor arm and prompt version, N, student model and prompt version, scorer model, prompt version and width. Every call records its token usage.
