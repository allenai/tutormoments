You are classifying a teaching coach's analysis of a tutoring moment.

You will be given one or more descriptions of a tutor's actions during a tutoring moment. Your job is to determine whether the moment involves the tutor employing pedagogical strategies that relate to scaffolding or those that relate to pushing for rigor. 

**Scaffolding**

Scaffolding is pedagogical support that helps a student accomplish a task.

- The tutor breaks down a problem into steps for the student.
- The tutor draws a diagram and uses a real-world analogy to help the student.
- The tutor rephrases the problem using simpler language or provides a simpler alternative. 
- The tutor asks guiding questions based on sub-steps to lead the student towards the solution. 
- The tutor fills in some of the steps for the student or provides a starting point.
- The tutor models an example solution. 
- The tutor gives the student a hint, e.g. reminding the student of a similar prior problem, or highlighting parts of the problem text.
- The tutor explains a concept or procedure. 
- The tutor presents a different representation or form of the problem to guide the student. 
- The tutor co-solves the problem with the student.
- The tutor reduces answer options to simplify the problem. 
- The tutor gives away definitions, answers, or key steps to the problem.   

**Rigor**

Rigor increases the level of conceptual challenge for the student. 

- The tutor has the student work on problems and/or struggle productively without support. 
- The tutor asks the student to justify, verbalize, or explain an answer, solution, or process. 
- The tutor increases problem complexity, e.g. whole numbers to decimals, one-step to two-step equations.
- The tutor asks the student to define and/or use a key vocabulary term.
- The tutor asks the student to find and fix their own error.
- The tutor asks the student to explain why an answer is wrong.

Though pushing for rigor can also involve questions (e.g. probing for reasoning), asking pinpointed, guiding questions is scaffolding, not a push for rigor. 

**Neither**

- The tutor zooms in on the screen. 
- The tutor moves on. 
- The tutor presents the student with a game. 
- The tutor engages in off-topic conversation. 
- The tutor reads out the problem. 
- The tutor types in the student's answer. 
- The tutor clicks on the screen.
- The tutor confirms the student's answer. 

# Your Task

Now, examine the following actions taken by a tutor. Using the exemplars above as guidance, determine if rigor and scaffolding strategies are present. Return "yes" or "no" in a JSON object, e.g., {"scaffolding": "no", "rigor": "yes"}. If the list of actions conflicts on whether rigor occurred or not, or whether scaffolding occurred or not, lean "yes".  

ACTIONS: 
{action_list}

# Output Format

Respond with valid JSON only:
{
  "scaffolding": "",
  "rigor": ""
}