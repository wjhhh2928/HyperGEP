# Evidence by figure

See the detailed [panel mapping](../reports/PAPER_EXPERIMENT_MAPPING.tsv) and [experiment status](../reports/EXPERIMENT_STATUS.tsv). For every corrected fit, the run directory includes `config_resolved.yaml`, `run_manifest.json`, `split_indices.tsv`, `predictions.csv`, `metrics.csv`, `training_log.csv`, preprocessing state and a checkpoint index or estimator snapshot.

| Figure | Train and analyze | Corrected evidence needed |
|---|---|---|
| 2 and S1–S3 | `experiments.run_main` and `experiments.run_baselines` on MaizeGEP, then `scripts.aggregate_results` | Same held-out records, raw data and authorized split |
| 2g–h and S5 | Residual statistics and diagnostic plots from `scripts.aggregate_results` | Independent pairs and a prespecified cluster-aware inference method |
| 3 | G2F main and baselines with the same runner | Paper G2F cohort and all comparator evidence |
| 4a–b | `experiments.run_ablation` | Matched splits and compute budget, one change per ablation |
| 4c | `experiments.run_genomic_comparators` | DNNGP/DeepGS topology reconstructions. Review original framework, cohort and training differences |
| 4d–e | `experiments.run_negative_transfer`, then `scripts.aggregate_results` | Per-fold matched STL, MTL and HyperGEP predictions |
| 4f–g | `experiments.run_sensitivity` | Fixed, prespecified K values with comparable outer folds |
| 5 and S4 | `scripts.meta_dependencies` and `experiments.run_loeo` | Complete fold-level weight tables, environment geocodes and held-out performance |
| 6a–h | `experiments.run_interpretability`, then `scripts.attribution_panels` | Held-out gradients and marker coordinates. The recorded corrected sampling scheme differs from the historic cohorts |
| S6 | `experiments.run_interpretability --analysis all` | Training-fold importance and variance-matched controls evaluated on held-out records. Reconcile the original cohort and matching definition |

The analysis scripts produce diagnostic charts and tidy evidence for panel reconstruction. Paper styling, full panel-by-panel regeneration and inference for site/genotype-dependent residuals require validated evidence. For attribution, the conditional bootstrap resamples records within each genotype across fixed checkpoints. Within-genotype environment TV comparisons use the same outer checkpoint. The corrected composition comparison uses sorted sample IDs when acquisition order is unavailable.
