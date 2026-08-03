# Categorizing tutor actions into action types

You are categorizing the actions a human tutor takes during one-on-one K-12
math tutoring, for moments where the tutor was **tutoring K-12 math --
freely mixing scaffolding (making the task more accessible) and pushing for
rigor (raising cognitive demand and independence), often within the same
moment**. You are given a fixed list of action types lettered A to M, each
with a definition and examples. Your task is to assign each unlabeled tutor
action to the single letter whose action type it best fits.

Categorize by the action's pedagogical **function**, not the math topic or
surface wording. For example, "The tutor asks what 2 + 2 is" and "The tutor
asks the student to do the addition problem" are the same action type.

When an action could fit more than one type, categorize by its primary
**mechanism** -- *how* the tutor acts -- not an incidental outcome. A
correction delivered by explaining why an answer is wrong is an
*explanation*; restating the problem in different words is an *alternative
representation*; asking the student to rate confidence in their answer is
*self-assessment*, not a check-in.

Some statements carry a stance prefix like "The tutor scaffolds by X" or
"pushes for rigor by X" -- judge them by the underlying move X, not the
"scaffolds/pushes" framing. These actions freely mix scaffolding and rigor.
Both poles are first-class. Critically, scaffolding-style guiding/funneling
questions that lead the student toward a specific answer are a DIFFERENT
action type from rigor-style prompts that ask the student to justify,
explain, or reason independently -- never merge them. Use "M" (Other) only
when the action genuinely fits no other type (off-task, technical issues,
non-actions, or a bare stance with no concrete move).

## Action types

${categories_block}

## Instructions

Below are numbered actions (numbered from 1). For each, assign exactly one
category letter (A to M). Return one entry per action in an `assignments`
array, each giving the action's `id` (its number) and its `category`
letter:

```json
{"assignments": [{"id": 1, "category": "B"}, {"id": 2, "category": "A"}, {"id": 3, "category": "M"}]}
```

## Actions to categorize

${statements_block}
