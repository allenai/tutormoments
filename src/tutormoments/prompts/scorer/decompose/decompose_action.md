You are restructuring a teaching coach's analysis of a tutoring moment.

Given a description of a tutor's actions in a tutoring interaction, decompose it into short, standalone, atomic facets. Focus ONLY on extracting actions the tutor did or did not do. Exclude events and actions that are hypothetical (e.g. things the description says should have occurred).  

# Examples

INPUT
The tutor sees that the student knows the correct answer and then asks the student to explain why the answer is correct.
OUTPUT
["The tutor sees that the student knows the correct answer.", "The tutor asks the student to explain why the answer is correct."]

INPUT
The tutor breaks the question down into smaller more manageable chunks and asks prompting questions.
OUTPUT
["The tutor breaks the question down into smaller more mangeable chunks.", "The tutor asks prompting questions."]

INPUT
Encouraging, reminding, waiting
OUTPUT
["The tutor is encouraging.", "The tutor is reminding.", "The tutor is waiting."]

INPUT
The tutor's strategy is to again tell the student they are adding 10 when it should be 5.
OUTPUT
["The tutor's strategy is to again tell the student they are adding 10 when it should be 5."]

INPUT
The tutor didn't push the student. They should've asked probing questions. 
OUTPUT
["The tutor didn't push the student."]

INPUT
The tutor told the student to compare this to an input output table. If the tutor had told the student to skip count instead of input-output, it would've been more effective.
OUTPUT
["The tutor told the student to compare this to an input output table."]

# Your task

Now, look at the following description of a tutor's actions, and return ONLY a valid JSON array of strings representing standalone facets that correspond to something the tutor did. If the description is entirely about hypothetical or non-occurring events/actions, return an empty list. 

Action description:
{action}