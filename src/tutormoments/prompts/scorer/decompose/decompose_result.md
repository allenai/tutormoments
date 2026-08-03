You are restructuring a teaching coach's analysis of a tutoring moment.

Given a description of a tutoring interaction, extract short, standalone, atomic facets from the **Result** section that describe the resultant state or actions of the student. Split compound sentences into multiple facets. Focus ONLY on extracting student outcomes that resulted from the tutor's action. Exclude student behaviors and actions that occurred earlier in the moment, even if they are mentioned in the result section.

# Examples

INPUT
Situation: The student is starting their first problem of the session.  
Action: The tutor sets up the initial part of the problem and points the student towards a specific strategy to try. 
Result: It correctly got the student through the problem which means it's effective, but it's unclear if it was necessary. The student didn't try it on their own first (without scaffolding) so unclear if it was needed.
OUTPUT
["It correctly got the student through the problem."]

INPUT
Situation: The student is struggling on a multiplication question. 
Action: The tutor provides guiding questions. 
Result: The strategy is effective. The student was struggling, and now the student is able to answer the guiding questions in order to solve this problem correctly. 
OUTPUT
["The student is able to answer the guiding questions.", "The student solves this problem correctly."]

INPUT
Situation: The student is starting a new set of problems. 
Action: The tutor walks the student through the first problem step-by-step. 
Result: This is not an effective way to see if the student has mastered the material. If the tutor continues to solve problems for the student then the student has no way to show what they know.
OUTPUT
[]

INPUT
Situation: The student is clearly unsure about how to approach this problem, so this is an appropriate moment for scaffolding. 
Action: The tutor offers help by rephrasing the question step-by-step. 
Result: The student appeared to not make sense of what the question was asking properly, and the scaffolds the tutor provided gave that information. The student was able to realize and remediate their misunderstanding. 
OUTPUT
["The student was able to realize and remediate their misunderstanding."]

INPUT
Situation: The student has been making similar mistakes on multiple questions.
Action: The tutor summarizes the correct move the student made that is different from the errors they had been making previously. 
Result: This strategy is effective in getting the student to get the answer correct and the student is able to transfer this understanding to the next question as well.
OUTPUT
["The student gets the answer correct.", "The student is able to transfer this understanding to the next question as well."]

# Your task

Now, look at the following description of a tutoring interaction, and return ONLY a valid JSON array of strings representing standalone facets that correspond to the **resultant** state, behavior, and actions of the student. If the description does not discuss the student's outcome, return an empty list. 

Situation: {situation}
Action: {action}
Result: {result}