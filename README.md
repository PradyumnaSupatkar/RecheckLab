# RecheckLab

RecheckLab studies whether one specified LLM revision is likely to help, hurt, or leave an initial answer unchanged, and whether its expected value justifies the extra inference cost. Skipping retains the initial answer; it does not certify correctness.

## Scientific objective and current evidence

The v0.2 offline demo replays published SEVRA Qwen3-4B responses. It trains a compact outcome predictor and compares policies on fixed partitions. It did not generate the included recorded answers or reproduce the source paper's evaluation. The target-test recorded-response replay reports 76.13% initial accuracy; the frozen gate reaches 76.67% while rechecking 3.41%, whereas always rechecking reaches 72.85%. This retrospective result is not independent live evidence. Classification accuracy is affected by class imbalance, and HURT detection is weak. The audit found no verified new improvement.

An optional live mode makes local Ollama calls under a different protocol. Current live checks establish operation only; they do not validate predictive accuracy or transfer. See [research summary](research/RESULTS_AND_LIMITATIONS.md), [data provenance](DATA_SOURCES.md), [research protocol](RESEARCH_PROTOCOL.md), and [IEEE audit](IEEE_RESEARCH_AUDIT.md).

## Architecture

- `rechecklab/`: data loading, feature extraction, outcome prediction, policy evaluation, live collection, protocol validation, and reporting.
- `app.py`: Streamlit demo and results interface.
- `scripts/`: reproducible preparation, verification, training, collection, and reporting entry points.
- `tests/`: unit and contract tests.
- `research/`: concise, reviewed summaries; frozen detailed run outputs remain local and are not altered.

## Setup

Requires Python 3.11 or 3.12 and the dependencies in `requirements.txt`. Install a compatible Python version first; tests have **not** been executed because no Python interpreter is currently available in the working environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The repository contains source code, tests, dependency specifications, protocols, and research documentation. Datasets, trained model binaries, and generated experiment outputs are intentionally excluded. Consequently, the recorded-response demo requires obtaining or regenerating its dataset and model artifacts before it can run. Follow the pinned provenance and preparation instructions in `DATA_SOURCES.md` and the research protocols. Do not commit private keys or credentials.

The live demo requires a compatible Ollama installation, the specified model (`qwen3:4b`), and the model/protocol conditions documented in the live scripts. Live mode makes actual local inference calls; it is distinct from the recorded-response replay. Current live checks establish operation only, not predictive accuracy or transfer.
## Reproducibility and limitations

Version-control source, tests, dependency specifications, protocols, dataset provenance, baseline checksums, and compact verified summaries. Bulk datasets, fitted model binaries, logs, and generated run trees are intentionally excluded; obtain/prepare data using the documented scripts and pinned source revisions. Preserve protocol hashes and split manifests for any future reviewed run as small research metadata when they are sanitized and verified.

Current evidence is retrospective for the public recorded-response study. Live experiments are independent collections but remain small/incomplete and use a changed protocol. Results do not establish deployment safety, calibrated probabilities, broad-domain performance, or a validated live benefit. Do not treat the held-out CommonsenseQA validation subset as an official test leaderboard result.

## License and data

No project license is asserted here. Public datasets retain their original terms and attribution; see DATA_SOURCES.md and the source dataset cards.

