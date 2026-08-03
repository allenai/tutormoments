You are classifying a teaching coach's analysis of a tutoring moment.

Given a situation description, determine one of the following: 
1) Is this an appropriate moment for scaffolding? yes / no / unclear / no_mention
2) Is this an appropriate moment to push for rigor? yes / no / unclear / no_mention

Note that rigor and scaffolding are not mutually exclusive. Your response should only consider very clear, direct statements in the situation description that convey whether the situation is appropriate or not for scaffolding or rigor. 

# Examples

The tutor should be pushing for rigor. 
OUTPUT: {"scaffolding": "no_mention", "rigor": "yes"}

Here is an appropriate time to push. 
OUTPUT: {"scaffolding": "no_mention", "rigor": "yes"}  

This is a good time to scaffold. 
OUTPUT: {"scaffolding": "yes", "rigor": "no_mention"}  

The student makes an error in calculation. 
OUTPUT: {"scaffolding": "no_mention", "rigor": "no_mention"}   

This is not an appropriate time to scaffold. 
OUTPUT: {"scaffolding": "no", "rigor": "no_mention"}   

The tutor is providing the student with additional tools. 
OUTPUT: {"scaffolding": "no_mention", "rigor": "no_mention"}   

This is not a place for scaffolding or rigor. 
OUTPUT: {"scaffolding": "no", "rigor": "no"}

This is an appropriate place to insert scaffolds. 
OUTPUT: {"scaffolding": "yes", "rigor": "no_mention"}  

# Your Task 

Now, examine the following situation and decompose it into appropriateness judgements, following the earlier examples and instructions. 

Situation:
{situation}

# Output Format

Respond with valid JSON only:
{
  "scaffolding": "",
  "rigor": "",
}