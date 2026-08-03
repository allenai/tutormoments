You are restructuring a teaching coach's analysis of a tutoring moment.

Given a description of a tutor's actions in a tutoring interaction, extract textual clauses or spans that indicate or suggest the presence of over-scaffolding. Examples of textual spans that convey the presence of over-scaffolding: 
- "The tutor ended up explaining the answer to the student, which allowed the student to just check out from their learning."
- "This appears to be a bit of over scaffolding."
- "The tutor did way too much scaffolding."
- "The tutor is doing a lot of teaching or talking at the student but the student is not given an opportunity to apply these skills at all."
- "The tutor almost added too much support at this point."
- "The tutor is still doing a lot of the academic work here."
- "Seemed like maybe over-scaffolding."
- "The tutor was giving away more than necessary." 

Focus on descriptions that indicate or suggest **excessive** or unwarranted levels of support, not necessary support or the mere presence of scaffolding at all. Under-rigor or ineffective strategies do not neccessarily entail over-scaffolding. Do not speculate; rely on the description's judgement. 

# Your task

Look at the following description of a tutoring interaction, and return ONLY a valid JSON array of strings representing spans that suggest or indicate that the tutor is over-scaffolding. If the description does not include any indication of over-scaffolding, return an empty list. 

Situation: {situation}
Action: {action}
Result: {result}