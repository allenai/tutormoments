You will be given a tutoring transcript that contains a cut point. Your job is to determine the first pedagogical strategy a tutor decides to employ after the cut point, and whether that strategy relates to scaffolding, a push for rigor, both, or neither. 

## Scaffolding

Scaffolding is support that helps a student accomplish a task. Scaffolding makes content more accessible to a student. Appropriate scaffolding should make a task more approachable, while still requiring the student to do core thinking. 

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
- The tutor states a correction for the student.

### Rigor
Pushing for rigor means increasing the level of conceptual challenge for the student to foster critical thinking, independence, and deeper understanding. Tutors may encourage higher-order thinking and increase the cognitive demand of tasks when the student’s behavior suggests they are ready for it.

Common strategies that push for rigor:
- The tutor asks the student to justify or explain an answer, solution, or process, including why an answer may be wrong.
- The tutor withdraws support and has the student work on problems independently and/or struggle productively.
- The tutor increases problem complexity or difficulty, e.g. whole numbers to decimals, one-step to two-step equations.
- The tutor asks the student to define and/or use a key mathematical term.
- The tutor asks the student to find and fix their own errors.

# Your Task

Now, examine the following tutoring moment. Respond with valid JSON only:
{
  "description": "A sentence or two summarizing the tutor's pedagogical decision at a level of abstraction similar to the bullet points above.",
  "scaffolding": "yes or no",
  "rigor": "yes or no"
}

The following excerpt contains a tutoring moment. There, >>> CUT POINT <<< is the turn before the tutor's pedagogical strategy that you should focus on. If there are multiple strategies that occur after the cut point, focus on the classifying the first one.

EXCERPT: 
{excerpt}