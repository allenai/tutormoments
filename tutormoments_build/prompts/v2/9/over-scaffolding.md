You are given a tutoring moment containing a cut point. Your task is to determine whether the tutor over-scaffolds after the cut point.

Scaffolding is support that helps a student accomplish a task and makes content more accessible to a student. Appropriate scaffolding should still require the student to do core thinking. 

Over-scaffolding happens when the tutor:
- unnecessarily supplies a step the student was positioned to produce. 
- does not give the student a chance to show what they know, especially at the beginning of a session
- does not offer openings for the student to make substantive mathematical contributions during the support
- fails to fade the support after prior successes on similar items.
- does most of the cognitive work
- oversimplifies the problem or reduces the task to a sequence of procedural tasks with insufficient reasoning (e.g. overly leading or directive)
- over-explains
- re-scaffolds the same procedure or concept too many times
- jumps in during a student's productive struggle

An interactive sequence, even if it is lengthy, in which the tutor allows the student to meaningfully engage, is not over-scaffolding. Ask whether, overall, the tutor over-scaffolds right after the cut point. 

Note that the ordering of screen interactions and dialogue in the transcript do not always indicate the exact sequence of events. Sometimes a screen interaction or update happens simultaneously with the dialogue turn that precedes it. 

# Your Task

If teachers believe that this moment is appropriate for a push for rigor and the tutor doesn't push for rigor, over-scaffolding is almost always occurring. If teachers believe that this moment is appropriate for scaffolding, it is possible that over-scaffolding is still occurring, if the tutor provides too much support and replaces students' thinking after the cut point.

For the excerpt below, teachers describe the situation before the cut point as the following: 
{descriptions}

Teachers tend to believe that this situation {is_appropriate_for}. 

A previous pass by an LM determined that the tutor is {action} after the cut point.

Examine the following tutoring moment. Respond with valid JSON only:
{
  "description": "A sentence or two summarizing what the tutor did, and your reasoning for whether it constitutes over-scaffolding.",
  "over-scaffolding": "yes or no"
}

The following excerpt contains a tutoring moment. There, >>> CUT POINT <<< is immediately before the tutor's scaffolding strategy that you should focus on analyzing. 

EXCERPT: 
{excerpt}
