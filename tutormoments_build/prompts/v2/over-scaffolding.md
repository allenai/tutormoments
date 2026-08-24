You are given a tutoring moment containing a cut point, after which a tutor scaffolds. Your task is to determine whether over-scaffolding is occurring. 

## Over-scaffolding 
Over-scaffolding occurs when the tutor provides too much support and replaces students’ thinking. 

These actions include times when the tutor unnecessarily:
- Does most of the cognitive work. 
- Does not give openings for the student to contribute.
- Does not give openings for the student to demonstrate what they can do.
- Reduces the task to a sequence of procedural tasks with insufficient reasoning (e.g. overly leading or directive).
- Begins scaffolding before the student has shown what they can do.
- Gives away answers or key steps.
- Over-explains.
- Re-scaffolds the same procedure or concept too many times. 
- Oversimplifies the problem.
- Fails to fade scaffolding. 
- Jumps in during a student's productive struggle.

Look for whether the student has demonstrated knowledge or competency earlier in the transcript; this can be one indicator of over-scaffolding occurring. 

# Your Task

Examine the following tutoring moment. Respond with valid JSON only:
{
  "description": "A sentence or two summarizing why over-scaffolding is occurring, using language similar to the bullet points above.",
  "over-scaffolding": "yes or no"
}

The following excerpt contains a tutoring moment. There, >>> CUT POINT <<< is immediately before the tutor's scaffolding strategy that you should focus on analyzing. 

EXCERPT: 
{excerpt}