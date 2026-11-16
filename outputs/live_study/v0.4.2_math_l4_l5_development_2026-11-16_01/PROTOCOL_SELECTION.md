# v0.4.2 protocol selection

The four-call 8K/16K development benchmark used the first two question IDs in the already frozen v0.4.1 development selection, with no answer keys loaded. All four calls at each context stopped naturally and parsed. Mean generation speed was higher at 8K, free RAM was higher, and `ollama ps` showed a smaller runner allocation; 8K is therefore frozen for the 10-question rerun. Selection did not use correctness.

`num_predict=-1` is sent explicitly and verified empirically: all eight context-benchmark calls returned `done_reason=stop`, including outputs above 512 tokens. The model's own stop behavior is used; no stop-string override is sent. `num_ctx=8192` is far below the model-reported architecture context of 262,144.

The final prompts and settings are in `EXPERIMENT_MANIFEST.json`. The 60 evaluation question selection and keys were copied byte-for-byte from v0.4.1 and remain untouched; only the new v0.4.2 protocol manifest was created before evaluation inference. Development extraction must reach 9/10 initial and revision parseability before the evaluation run can begin.
