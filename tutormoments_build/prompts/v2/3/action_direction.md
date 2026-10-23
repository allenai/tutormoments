You will be given a tutoring transcript that contains a cut point. Your job is to determine what happens after the cut point: does the tutor scaffold, push for rigor, both, or neither? 

## Scaffolding

Scaffolding is support that helps a student accomplish a task. Scaffolding makes content more accessible and approachable for the student. 

Common scaffolding strategies:
- The tutor asks guiding questions to lead the student towards the solution.
- The tutor breaks down a problem into steps for the student.
- The tutor re-explains a concept, e.g. using a different example, metaphor, or phrasing.
- The tutor re-explains a procedure, e.g. using a different example, metaphor, or phrasing.
- The tutor models an example solution.
- The tutor co-solves or fills in some of the steps for the student. 
- The tutor rephrases the problem using simpler language or provides a simpler alternative.
- The tutor draws a diagram
- The tutor provides a different representation of the problem (e.g. a different form or a real-world analogy) for the student.
- The tutor reduces answer options to simplify the problem.
- The tutor gives the student a hint by providing a starting point.
- The tutor reminds the student of a similar prior problem
- The tutor highlights parts of the problem text.
- The tutor gives away the answer.
- The tutor corrects an error for the student.

## Rigor
Pushing for rigor means increasing the level of conceptual challenge for the student to foster critical thinking, independence, and deeper understanding. Tutors may encourage higher-order thinking and increase the cognitive demand of tasks. 

Common strategies that push for rigor:
- The tutor asks the student to justify or explain an answer, solution, or process, including why an answer may be wrong, and holds this demand open for the student. 
- The tutor withdraws support and has the student work on problems independently and/or struggle productively. For example, the tutor may have scaffolded earlier (e.g. modeling a solution), and they then push for rigor by giving the next problem back for the student to do. 
- The tutor presents a problem with higher complexity or difficulty, e.g. whole numbers to decimals, one-step to two-step equations. 
- The tutor asks the student to define and/or use a key mathematical term.
- The tutor asks the student or gives them space to fix their own errors.

**How to differentiate guiding questions (scaffolding) from rigor-pushing questions?**
- Guiding questions usually narrow the problem space: break a problem into pieces or draw attention to specific content that makes a problem accessible. Example: “What do we multiply first?” → scaffolding
- Justify & explain reasoning questions require higher cognitive demand and critical thinking. These questions ask students to make arguments, engage with higher-level concepts, consider why and how, or generalize their thinking. Example: “Why do we multiply before we add here?” → push for rigor

## Your Task

Now, examine the following tutoring moment. Respond with valid JSON only:
{
  "description": "Summarize, in one sentence or two, the tutor's pedagogical decision at a level of abstraction similar to the bullet points above.",
  "scaffolding": "yes or no",
  "rigor": "yes or no"
}

Remember, whether scaffolding or a push for rigor occurred are two separate, non-mutually exclusive judgements. 

The following excerpt contains a tutoring moment. There, >>> CUT POINT <<< is immediately before the tutor's pedagogical strategy that you should focus on describing and analyzing. Does the tutor scaffold, a push for rigor, both, or neither? 

EXCERPT: 
{excerpt}