| Dataset / policy | Final accuracy | Recheck | Avg. tokens | Total tokens saved | HELP captured | HURT incurred |
|---|---:|---:|---:|---:|---:|---:|
| MATH source test / Never recheck | 74.7500% | 0.000% | 3182.02 | 44.64% | 0.00% | 0 |
| MATH source test / Always recheck | 90.2500% | 100.000% | 5747.44 | 0.00% | 100.00% | 3 |
| MATH source test / Completion rule | 90.0000% | 26.750% | 4402.42 | 23.40% | 95.38% | 1 |
| MATH source test / Frozen ML gate | 90.0000% | 27.250% | 4412.82 | 23.22% | 95.38% | 1 |
| MATH source test / Oracle (hindsight) | 91.0000% | 16.250% | 3707.84 | 35.49% | 100.00% | 0 |
| CommonsenseQA target test / Never recheck | 76.1255% | 0.000% | 2243.60 | 52.47% | 0.00% | 0 |
| CommonsenseQA target test / Always recheck | 72.8513% | 100.000% | 4719.98 | 0.00% | 100.00% | 34.5 |
| CommonsenseQA target test / Completion rule | 76.6712% | 3.411% | 2334.85 | 50.53% | 38.10% | 0 |
| CommonsenseQA target test / Frozen ML gate | 76.6712% | 3.411% | 2334.85 | 50.53% | 38.10% | 0 |
| CommonsenseQA target test / Oracle (hindsight) | 77.5580% | 1.432% | 2291.70 | 51.45% | 100.00% | 0 |
