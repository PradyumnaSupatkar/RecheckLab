# v0.4.3 frozen evaluation recovery state

- Frozen selection: 60 questions, byte-identical and in the original order from v0.4.2.
- New v0.4.3 evaluation calls: none.
- Question 1: original v0.4.2 initial attempt 1 is preserved as incomplete. Its response chunks and terminal metadata were unavailable in v0.4.2 and remain unknown.
- Questions 2–60: no inference attempts have been recorded; these 59 questions remain untouched.
- Any v0.4.3 retry of question 1 is a new call with attempt number 2, stored separately. No retry is automatic.

See `../v0.4.3_math_l4_l5_development_2026-11-16_01/V043_OPERATIONS.md` for how to run or resume the frozen evaluation.
