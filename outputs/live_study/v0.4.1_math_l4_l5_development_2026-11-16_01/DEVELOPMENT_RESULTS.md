# MATH L4/L5 development gate

**Status: FAIL**. The gate requires at least 9/10 parseable initial and revision answers, successful exact-grader validation, all initial-only gate features compatible, no call failures, and safe RAM/VRAM.

- Questions/calls: 10/10 and 20/20.
- Parseable initial/revision answers: 4/10 and 5/10.
- Parseable but incorrect initial/revision answers: 0 and 0.
- HELP/HURT/NO_CHANGE: 1/0/9.
- Gate features valid before revision calls: 10/10; completion-rule rechecks: 6.
- Minimum free RAM/VRAM: 3.51 GiB / 1466 MiB.

The exact grader passed its synthetic checks; all 10 initial-only gate feature vectors were available before revision calls; all 20 calls completed; and RAM/VRAM stayed above the safety floors. The sole failed criterion was answer extraction reliability (4/10 initial and 5/10 revision parseable, below 9/10 each). The preselected 60-question evaluation was not started.
