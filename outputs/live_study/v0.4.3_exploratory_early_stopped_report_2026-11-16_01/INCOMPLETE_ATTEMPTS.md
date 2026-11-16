# Incomplete attempts and question accounting

The frozen selection contains 60 questions. 43 have fully completed initial/revision pairs; 47 distinct selected questions have at least one recorded attempt; 13 were never attempted.

Incomplete initial calls: 4. Incomplete revision calls: 0. Journal events classified as interrupted: 3. Timeout-related incomplete calls: 2.

## Incomplete calls

| Question ID | Phase | Attempt | Terminal status | Chunks | Last persisted elapsed (s) | Notes |
|---|---|---:|---|---:|---:|---|
| math:test:intermediate_algebra:intermediate_algebra/823 | initial | 1 | legacy_interruption_without_chunk_journal | 0 |  | v0.4.2 client disconnected; the response body and any partial stream were not persisted |
| math:test:precalculus:precalculus/490 | initial | 1 | watchdog_timeout | 18557 | 2699.859000000004 | 2700s call watchdog exceeded |
| math:test:number_theory:number_theory/18 | initial | 1 | timeout | 12179 | 1620.125 | Canceled after the 420-second amendment was issued; the call had already run about 1620 seconds. Partial chunks are preserved; no completion metadata was returned. |
| math:test:prealgebra:prealgebra/166 | initial | 1 | user_cancelled | 189 | 24.5 | User explicitly stopped the evaluation; partial stream preserved and no retry authorized. |

## Question 1 history

Question 1 is `math:test:intermediate_algebra:intermediate_algebra/823`. Its only recorded attempt is the prior v0.4.2 initial attempt 1, already marked interrupted in the inherited journal. v0.4.3 did not retry it; no initial/revision pair or policy decision exists for it. The partial historical response and token metadata were unavailable in the original evidence. No Question 1 attempt 2 was run.

## Runtime amendment and stop

The original manifest remains unchanged at its original 2700-second watchdog. The first long call (precalculus/490) terminated at that watchdog. A later runtime amendment set the collector watchdog to 420 seconds for future calls; when that amendment was issued, the already-running number_theory/18 call had exceeded 420 seconds and was explicitly canceled, with its last persisted elapsed time at 1620.125 seconds. The amended runner began the next initial call (prealgebra/166), which was explicitly canceled by the later stop instruction after 189 chunks. No retry was scheduled, and no Question 1 attempt 2 was run.

## Unattempted frozen questions

- `math:test:number_theory:number_theory/278`
- `math:test:intermediate_algebra:intermediate_algebra/115`
- `math:test:intermediate_algebra:intermediate_algebra/740`
- `math:test:prealgebra:prealgebra/634`
- `math:test:intermediate_algebra:intermediate_algebra/206`
- `math:test:precalculus:precalculus/389`
- `math:test:prealgebra:prealgebra/5`
- `math:test:precalculus:precalculus/488`
- `math:test:algebra:algebra/595`
- `math:test:prealgebra:prealgebra/386`
- `math:test:intermediate_algebra:intermediate_algebra/69`
- `math:test:algebra:algebra/878`
- `math:test:intermediate_algebra:intermediate_algebra/233`
