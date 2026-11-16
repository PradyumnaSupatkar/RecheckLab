# Five-question generation preflight

Status: COMPLETE. Generation calls attempted: 20/20.

| Configuration | Calls OK | Valid FINAL | Correct | Length stops | Avg actual tokens | Avg seconds |
|---|---:|---:|---:|---:|---:|---:|
| initial | 5/5 | 3/5 | 2/5 | 3 | 1961.4 | 93.81 |
| A | 5/5 | 4/5 | 2/5 | 1 | 1251.8 | 48.84 |
| B | 5/5 | 3/5 | 2/5 | 2 | 1416.8 | 56.14 |
| C | 5/5 | 3/5 | 3/5 | 2 | 1519.2 | 60.58 |

Accuracy counts and paired oracles are diagnostic only. Invalid completed answers are scored incorrect. Actual tokens include prompt plus generated tokens, including thinking. Only saved visible initial content enters all three independent rechecks.

No existing protocol or artifact was modified; see integrity.json. No 50-question study ran.

Problem: none
