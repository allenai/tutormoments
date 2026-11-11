# Guiding Sheet for Scaffolding & Rigor

*The guide given to the expert teacher annotators who produced `annotations/`. Reviewer comments, the annotators' feedback tab and internal notes have been removed; the guidance is otherwise as the annotators saw it. The guide was revised while annotation was under way as annotators surface questions and aligned on consensus interpretations.*

## Contents

- [Overview](#overview)
- [Cut Point Selection](#cut-point-selection)
  - [Examples of situations that precede cut points](#examples-of-situations-that-precede-cut-points)
- [Key Moment Boundaries](#key-moment-boundaries)
  - [Where does a moment start?](#where-does-a-moment-start)
  - [After the cut point, where does the moment end?](#after-the-cut-point-where-does-the-moment-end)
  - [Example moment, with boundaries](#example-moment-with-boundaries)
- [Scaffolding](#scaffolding)
  - [Over-scaffolding](#over-scaffolding)
- [Rigor](#rigor)
  - [Strong rigor](#strong-rigor)
  - [Weak rigor](#weak-rigor)

## Overview

This sheet is a guide for what to look for and describe during your annotations. You may describe more observations than what this sheet covers, but please consider the behaviors/interactions in this sheet when making judgements.

**Goal.** You are reading through real, human, one-on-one tutoring sessions, and identifying decision points where the tutor had an opportunity to either scaffold or push for rigor.

**Reading tip.** For both first- and second-passes, have a good understanding of the whole transcript, especially places where a student is demonstrating (or not demonstrating) knowledge and competency. These student behaviors relate to scaffolding / rigor judgements. Careful consideration of session-level student capability and problem/topic progress could help with improving annotation alignment!

**What are your annotations used for?** We are building a framework to measure whether LLMs can balance one of the hardest trade-offs in education: when to step in and help a student and when to hold back and let the student do more of the work. Our evaluation tool takes the transcript up to a decision point you pick (the "cut point"), hands it to a language model, and has the model take over as the tutor in a simulated session – with the student played by another language model – to see what the LLM tutor does.

## Cut Point Selection

**What is a cut point?** A cut point is where it is clear that a tutor has to make a strategic decision about whether to scaffold or push for rigor. We use cut points to decide where to stop a transcript, have an AI tutor pick up the conversation, and see how it might've addressed the situation differently.

Cut points create challenges for the AI: how would a different tutor respond?

**Cut points should designate clear-cut situations.** The context before the cut point should provide enough situational information for a human or AI tutor to know how to respond (scaffolding or a push for rigor). It should be clear as to whether and why this is a scaffolding-appropriate or rigor-appropriate situation. Aim to identify cut points where most experienced teachers would agree on whether a good, adaptive tutor should scaffold or push for rigor. If the transcript or narration isn't clear as to what is happening, do not select anything and instead move on to other parts of the transcript.

It's possible that other pedagogical decisions are being made in transcripts (e.g. the tutor needing to correct their own error or redirect the student), but let's just focus on scaffolding and rigor pushes for now.

**Selection guidance:**

- In the UI, when you select a turn as a cut point, the cut is made after that turn.
- Cut points should precede a variety of scaffolding and rigor-pushing decisions, both good and bad, that may occur. Pick cut points before turns where the human tutor has done well (e.g. pushed for rigor when increased rigor was needed), and before turns in which they made suboptimal decisions (e.g. over-/under-scaffolding, weak rigor, missed opportunities).
- Most cut points will likely occur at a student's turn (to see how a tutor may respond), but you can also select to cut after a tutor's turn.
- Select approximately 5-8 of the clearest cut points per transcript, ideally scattered throughout the transcript.
- Prioritize rigor-appropriate situations, as these are less common than scaffolding-appropriate ones and also more difficult for LLM tutors. We would love to obtain a ~50/50 split of these two situation types across all transcripts.
- Ignore non-math-related discussions

### Examples of situations that precede cut points

Here's a non-exhaustive list of potential situations:

**Scaffolding-appropriate**

- Student gave an incorrect answer, and they likely cannot resolve the error on their own
- The student appears to be confused, e.g. verbally expressing confusion in response to a tutor's explanation or demonstrating unclear reasoning
- The student didn't know how to proceed or is stuck/struggling while working on a problem
- The student asks for help from the tutor, and help is warranted

**Rigor-appropriate**

- The student says they don't need help
- The student is struggling productively on a problem, and the tutor should continue to let them work through it on their own
- The student has been getting a sequence of related problems correct consistently
- The student has just received scaffolding from the tutor on a problem similar to the current one
- The student was able to answer a question correctly

Moments that are appropriate for diagnosis or assessing why the student made a mistake, and are unclear as to whether it's a scaffolding or rigor-pushing moment, are orthogonal to our current project.

## Key Moment Boundaries

After selecting a cut point, you'll draw key moment boundaries. Key moments include the situation, the cut point, the tutor's first strategy decision, and the student's response. If you're annotating a preselected moment, you'll have the option to skip confusing moments or redraw moment boundaries.

Moments are short! (see examples below)

### Where does a moment start?

The turns between the start and the cut point typically contain the "Situation", or conversation turns that trigger the decision point.

**Guidelines.** For scaffolding-appropriate moments, the key moment start may precede turns that indicate that the student is confused, struggling, or has a misconception. For rigor-appropriate moments, situation turns may include those that indicate the student is performing well or has just received help and is ready to be more independent. It's possible that some context for situation appropriateness may extend beyond the problem (e.g. the student consistently demonstrating proficiency over multiple problems signaling they're ready for more rigor). In that case, just mark the start as the start of the current problem and describe in your annotation your reasoning.

**Does the AI tutor get the entire transcript up to the cut point?** Yes! The AI tutor sees everything. The moment start boundary is mostly for us researchers to understand your annotations and for our LM categorizing tutors' actions to focus on approximately where a moment resides.

### After the cut point, where does the moment end?

**Guidelines.** The rest of the moment should wrap around the tutor's first strategy, attempt, or "Action" decision after the cut point, and the student's response to that decision. The checkboxes and text box in the "Action" section of the UI should focus on describing tutor actions after the cut point and before the end of the moment.

**What is a strategy or action decision?** A scaffolding or rigor pushing move listed in the "Scaffolding" and "Rigor" sections below. It may play out across several back-and-forth student-tutor turns (see the example moment below). It's possible that during a scaffolding/rigor-appropriate situation, the tutor may have done neither action; you'll be able to indicate this in our UI.

**Avoid letting a moment go too far beyond the tutor's first post-cut decision.** Key moments may not cover the entire problem solving arc. For example, if a tutor initially pushes for rigor but then starts explaining (scaffolding), the moment should end before any explaining (a second action decision) begins. If you wish to express that the second action wasn't appropriate (e.g. the push for rigor should have been carried forward), you could draw a second cut point. If you find yourself writing, "first the tutor did X and then they did Y", consider if the moment should actually end after X.

Following these moment boundary guidelines helps us improve teacher-teacher agreement and makes your descriptions easier for us to map back to the original transcript. 😄

### Example moment, with boundaries

```text
——————— Start ———————
Tutor: What does 4 over 4 mean?
Student: Four fourths.
Tutor: What happens when you have the same number over another number? What value does it equal?
Student: Um… I don't know.
- - - - - - - - - - - - -  Cut point - - - - - - - - - - - - - -
Tutor: Think of it like a pizza cut into four pieces. If you have all four, what do you have?
Student: One pizza?
Tutor: Exactly! Similarly, what would 5 pieces out of a pizza cut into 5 equal?
Student: Also one pizza?
Tutor: Yes, so what's 4 over 4?
Student: One.
——————— End  ———————
```

In this example, the cut point occurs after the student expresses confusion ("Situation"). The rest of the moment wraps around the tutor's first strategy decision ("Action"), which is scaffolding by "re-explaining a concept or procedure using a metaphor" and "guiding questions".

```text
——————— Start ———————
[first problem of the session]
Tutor: Okay. You remember how to add and subtract fractions, right?
Student: I think I remember… I can try giving you the answer.
Tutor: Okay.
Student: [screen interaction] The student writes "1/2" again and then "-1" again, underlining the problem.
Student: Am I correct?
- - - - - - - - - - - - -  Cut point - - - - - - - - - - - - - -
Tutor: [pause]
Tutor: Um, do you want to explain why you think it's one half?
Student: Cuz the the 10 minus oh wait, wait, wait.
Student: Okay. It's probably negative one because you cannot do this.
——————— End  ———————
```

In this example, the cut point occurs after the student expresses some intention do the problem on their own. It's the first problem of the session, so the tutor could see what the student can do. It's an appropriate place to push for rigor, and the tutor does so by withholding information that the student's solution is incorrect, and instead asking for justification / explanation from the student. There is possibly an alternative cut point after "I can try giving you the answer" to see if the tutor decides to let the student work independently. The cut point shown above is slightly more interesting, to see instead if the tutor directly answers "Am I correct" with yes/no or instead raises the challenge by letting the student self-correct.

## Scaffolding

**Definition.** Scaffolding is support that helps a student accomplish a task. Scaffolding makes content more accessible to a student. Appropriate scaffolding should make a task more approachable, while still requiring the student to do core thinking.

**Common scaffolding strategies:**

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

### Over-scaffolding

**Definition.** Over-scaffolding occurs when the tutor provides too much support and replaces students' thinking. These actions include times when the tutor unnecessarily:

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

## Rigor

**Definition.** Pushing for rigor means increasing the level of conceptual challenge for the student to foster critical thinking, independence, and deeper understanding. Tutors may encourage higher-order thinking and increase the cognitive demand of tasks when the student's behavior suggests they are ready for it. Rigor requires a balance of conceptual understanding, skills and procedures, and real-world application [see "rigor" [here](https://www.calstate.edu/csu-system/why-the-csu-matters/graduation-initiative-2025/co-requisite-mathematics-summit/Handouts/Common%20Core%20Key%20Shifts.pdf)]. Rigor ≠ difficulty. A question might be difficult because it's overly complicated, tedious, hard to parse, inappropriately above grade level, without being rigorous.

**Common strategies that push for rigor:**

- The tutor asks the student to justify or explain an answer, solution, or process. This may occur when a student gets something right (a push for demonstrating deeper understanding) or when they get an answer wrong (with the intention for the student to independently repair their mistake).
- The tutor withdraws support and has the student work on problems independently and/or struggle productively.
- The tutor increases problem complexity or difficulty, e.g. whole numbers to decimals, one-step to two-step equations.
- The tutor asks the student to define and/or use a key mathematical term.
- The tutor asks the student to find and fix their own errors.

When a tutor decides to push for rigor after a cut point, we'll ask you to determine whether the push for rigor is strong, weak, or unclear. Unclear cases are those where it is difficult to tell if the rigor push was strong or weak, or if it's somewhere in between.

### Strong rigor

**Definition.** Strong rigor means that the student is expected to demonstrate and communicate deep knowledge of the topic. It likely involves moving up into higher levels of Bloom's taxonomy (into applying, analyzing, synthesizing, evaluating) or higher [Depth of Knowledge](https://www.webbalign.org/dok-definitions-for-math) levels (extended thinking, strategic thinking) and away from simple recall or calculation.

For rigor pushes involving questions:

- Questions that require a student to explain why something works.
- Questions that require students to think critically about their response or possible responses.
- Questions that require students to make inferences about how a concept can be applied in a new way or to a new situation
- Questions that push students to self-identify mistakes

Non-question rigor pushes (e.g. withdrawing support, increasing problem complexity) may be strong if it's adequately adapting to the student and producing a desirable level of challenge.

### Weak rigor

**Definition.** Weak rigor may occur if it seems like the tutor intends to push for rigor, but is falling short, e.g. the student is not being pressed to a more challenging level of thinking. This may mean that the student may be able to answer a question correctly without conceptual understanding, or without the ability to apply that knowledge in other situations.

For rigor pushes involving questions (e.g. justify, explain, analyze, define, or find & fix error):

- Questions that are likely to be answered by guessing.
- Questions that can be answered by following leading prompts from the tutor.
- Questions that can be answered by recall or copying steps.
- Questions that require only basic knowledge of the topic.
- Questions that are essentially rhetorical.

If the tutor scaffolds (e.g. asking guiding questions) during a rigor-appropriate moment, select that an (over-)scaffolding action is present.

If a tutor first pushes for rigor but then switches to scaffolding too soon, these are two separate pedagogical decisions in sequence – two different cut points, and potentially one good decision followed by a bad one.

**How to differentiate guiding questions (scaffolding) from rigor-pushing questions?**

Here's some suggestions:

- Guiding questions usually narrow the problem space: break a problem into pieces or draw attention to specific content that makes a problem accessible. Example: "What do we multiply first?" → scaffolding
- Justify & explain reasoning questions require higher cognitive demand and critical thinking. These questions ask students to make arguments, engage with higher-level concepts, consider why and how, or generalize their thinking. Example: "Why do we multiply before we add here?" → push for rigor
