### Putting together teacher-annotated human-human data

Data is first pulled from AWS. From the `tutormoments` root:  

```
cd data/v2_annotations/source
aws s3 cp s3://ai-tutor-bench-deidentified/v2_annotations/source/tutoring_provider_a_annotations.jsonl ./
aws s3 cp s3://ai-tutor-bench-deidentified/v2_annotations/source/tutoring_provider_a_v2_transcripts.jsonl ./
```

Then, we build our "ground truth" (moments adjudicated into single labels for situation and action) and student-tutor moment "excerpts" with cut points from transcripts: 

```
python -m tutormoments_build.v2.splits
python -m tutormoments_build.v2.build_ground_truth
python -m tutormoments_build.v2.excerpts
```

Excerpts exist because they format what's shown to action classifiers: the entire transcript up to the moment to the moment's end. 

To inspect annotator agreement: 

```
python -m tutormoments_build.v2.adjudicator_agreement
```

### Running and evaluating the scorer

Example with one model: 

```
python -m tutormoments_build.v2.classify_excerpts --model claude-opus-5
python -m tutormoments_build.v2.evaluate_scorer --model claude-opus-5
```

`tutormoments_build.v2.evaluate_scorer` refers to evaluating the LM action classifier. 

### Replays? 

The ground truth is saved in `data/ground_truth` into `iteration` and `test` splits. We develop our scorer on iteration and then evaluate the final version on the held out test split. For running replays, you can use moments' situations from both. 

"Adjudicated" ground truth isn't commited to Github but uploaded instead to `s3://ai-tutor-bench-deidentified/synthetic/ground_truth_v2/`. You'll likely want to use the `situation_type` key within each ground truth moment's json. The value of this key provides a sense of situation types marked by level of agreement, e.g. ``scaffold_only``, ``scaffold_maj``, ``fifty_fifty``, ``rigor_maj``, ``rigor_only`.  

If you prefer to also try aggregating results based on binary scaffolding is/isn't appropriate and rigor is/isn't apropriate annotations, you could use the `labels` true/false values for `scaffolding_appropriate` and `rigor_appropriate`. 

Since we're updating how moments are saved at the beginning of the pipeline (with the enrichment, turn numbering problem) the format of `data/ground_truth/*.jsonl` will likely change.  

Likely, for replays, you'll want to format each replay in a similar manner as excerpts, and classify the AI tutor's action similar to how we do it for humans in `classify_excerpts`. Sharing things between human-human data and replays could maybe be good for consistency. 

You can use the latest prompts in `tutormoments_build/prompts/v2/<version>/`, with `claude-opus-5` as the scorer when classifying tutor actions. When tracking experiments, keep track of both prompt and model versions. 