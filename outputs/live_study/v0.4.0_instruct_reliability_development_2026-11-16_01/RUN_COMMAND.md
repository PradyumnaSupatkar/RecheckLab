# Frozen execution command

After reviewing the pre-inference manifests, run exactly:

```powershell
python outputs/live_study/v0.4.0_instruct_reliability_development_2026-11-16_01/run_instruct_study.py --run
```

This executes the 10-question development gate first. It starts the 40-question collection only if the predeclared extraction and resource criteria pass.
