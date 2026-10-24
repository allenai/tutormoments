You are given a tutoring moment containing a cut point. Your task is to determine whether over-scaffolding is occurring after that cut point.

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
- The tutor corrects an error for the student.

**Over-scaffolding** occurs when the tutor provides too much support and replaces students’ thinking. 

These actions include times when the tutor:
- unnecessarily supplies a step the student was positioned to produce. 
- does not give the student a chance to show what they know, especially at the beginning of a session
- does not offer openings for the student to make substantive mathematical contributions during the support
- fails to fade the support after prior successes on similar items.
- does most of the cognitive work
- oversimplifies the problem or reduces the task to a sequence of procedural tasks with insufficient reasoning (e.g. overly leading or directive)
- over-explains
- re-scaffolds the same procedure or concept too many times
- jumps in during a student's productive struggle

# Your Task

Examine the following tutoring moment. Respond with valid JSON only:
{
  "description": "A sentence or two summarizing what the tutor did, and your reasoning for whether it constitutes over-scaffolding.",
  "over-scaffolding": "yes or no"
}

The following excerpt contains a tutoring moment. There, >>> CUT POINT <<< is immediately before the tutor's scaffolding strategy that you should focus on analyzing. 

EXCERPT: 
{excerpt}
