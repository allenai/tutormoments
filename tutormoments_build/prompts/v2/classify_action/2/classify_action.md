You will be given a K-12 tutoring transcript that contains a cut point. Your job is to determine what pedagogical strategy a tutor is employing after the cut point. Identify which of the listed action types are occurring after the cut point, and output a json of turn numbers and labels. 

**How to differentiate guiding questions (scaffolding) from rigor-pushing questions?**
- Guiding questions usually narrow the problem space: break a problem into pieces or draw attention to specific content that makes a problem accessible. Example: “What do we multiply first?” → scaffolding
- Justify & explain reasoning questions require higher cognitive demand and critical thinking. These questions ask students to make arguments, engage with higher-level concepts, consider why and how, or generalize their thinking. Example: “Why do we multiply before we add here?” → push for rigor
- Other rigor-pushing questions may ask the student to complete a problem or key step independently.

## Action types

### Scaffolding

`guiding_questions`: The tutor asks guiding questions to lead the student towards the solution.
`break_into_steps`: The tutor breaks down a problem into steps for the student.
`re_explain`: The tutor re-explains a concept or a procedure, e.g. using a different or simpler example, metaphor, or phrasing.
`model_example`: The tutor models an example solution.
`co_solve`: The tutor co-solves or fills in some of the steps for the student. 
`draw_diagram`: The tutor draws a diagram to support an explanation of a concept.
`diff_rep`: The tutor provides a different representation of the problem (e.g. a different form or a real-world analogy) for the student.
`reduce_options`: The tutor narrows the problem's choices, by eliminating some answer options for the student. 
`hint`: The tutor gives the student a hint.
`remind_prior_problem`: The tutor reminds the student of a similar prior problem.
`highlight_problem_text`: The tutor highlights parts of the problem's text.
`give_answer`: The tutor gives away the answer.
`state_correction`: The tutor fixes an error for the student.

### Rigor

`justify_explain`: The tutor asks the student to justify or explain an answer, solution, or process, including why an answer may be wrong, and holds this demand open for the student.
`withdraw_support`: The tutor withdraws support and the student works on problems independently and/or struggles productively.
`increase_complexity`: The tutor increases problem complexity or difficulty, e.g. whole numbers to decimals, one-step to two-step equations.
`define_term`: The tutor asks the student to define and/or use a key mathematical term.
`find_fix_errors`: The tutor asks the student to find and fix their own errors.

## Instructions

A previous pass by an LM already determined that the tutor has decided to {action} *after the cut point* in the excerpt below. Output category labels that fall under these action types using the fine-grained action list/s above, in the following format: 

```json
[{"evidence": "turn number/s", "category": "hint"}, {"evidence": "turn number/s", "category": "reduce_options"}]
```

If there are no relevant category labels for this moment, output an empty list.

EXCERPT:
{excerpt}
