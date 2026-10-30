You are restructuring a teaching coach's analysis of a tutoring moment.

Given a description of a tutoring interaction, extract short, standalone, atomic facets that describe the student's behavior and/or actions. Split compound sentences into multiple facets. Focus ONLY on descriptions of the student, not the tutor. 

# Examples

INPUT
Situation: The student solved the equation correctly, and then shows understanding of how to multiply fractions, so he seems ready for a challenge.
OUTPUT
["The student solved the equation correctly.", "The student shows understanding of how to multiply fractions.", "He seems ready for a challenge."]

INPUT
Situation: The student has been getting similar problems correct prior to this, so a push for rigor would be most appropriate.
OUTPUT
["The student has been getting similar problems correct prior to this."]

INPUT
Situation: The tutor is not being helpful at all here.
OUTPUT
[]

INPUT
Situation: The student needed help because they did the problem incorrectly.
OUTPUT
["The student needed help.", "They did the problem incorrectly."]

# Your task

Now, look at the following description of a tutoring interaction, and return ONLY a valid JSON array of strings representing standalone facets that correspond to the state, behavior, and actions of the student. If the description does not discuss the student, return an empty list. 

Situation: {situation}