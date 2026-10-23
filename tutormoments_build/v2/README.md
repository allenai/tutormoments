### Putting together teacher-annotated human-human data

Data is first pulled from AWS. From the `tutormoments` root:  

```
cd data/v2_annotations/source
aws s3 cp s3://ai-tutor-bench-deidentified/v2_annotations/source/tutoring_provider_a_annotations.jsonl ./
aws s3 cp s3://ai-tutor-bench-deidentified/v2_annotations/source/tutoring_provider_c_annotations.jsonl ./
aws s3 cp s3://ai-tutor-bench-deidentified/source/tutoring_provider_a.jsonl ./
aws s3 cp s3://ai-tutor-bench-deidentified/source/tutoring_provider_c.jsonl ./
```

Ground truth files are also on s3: 

```
cd data/ground_truth
aws s3 cp s3://ai-tutor-bench-deidentified/synthetic/ground_truth_v2/*.jsonl ./ 
```

Excerpts exist because they format what's shown to LMs (the tutor and action classifiers): the entire transcript up to the moment to the moment's end.

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