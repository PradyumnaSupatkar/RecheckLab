# Development-only completion correction

**Separate run:** `v0.3.3_development_thinking_off_2026-11-16_01`

This is a five-question development check on a disjoint seeded sample. It is not pooled with the eight-question pilot and is not final evaluation evidence. The only predeclared change was `thinking=true` to `thinking=false`; model digest, prompts, 1024-token call budgets, 4096 context, temperature, seeds, decoder settings, parser, and grader were unchanged.

- Calls: 10/10 successful; 0 failed.
- Parseable final answers: initial 5/5 (100%, exact 95% interval 47.8%-100.0%); revision 5/5 (100%, exact interval 47.8%-100.0%); pooled 10/10 (100%, exact interval 69.2%-100.0%).
- Mean call wall time: 22.6 seconds; total measured call time 226.4 seconds across five initial/revision pairs. Mean generated output tokens 461.7; mean prompt-plus-output tokens 893.1.

All ten outputs parsed under the fixed explicit-final parser. The original pilot had 2/8 parseable initial responses and 6/8 parseable revisions, but questions differed, so this development sample does not isolate a causal effect or establish expected completion reliability.
