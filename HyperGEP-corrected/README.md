# HyperGEP

HyperGEP combines maize genotypes and environmental covariates to predict agronomic traits. The corrected evaluation candidate uses a training-environment hypergraph, a convolutional SNP token encoder, environment-to-SNP attention, and a prediction-space Meta-Fusion model. Outer-fold training includes all preprocessing, independent trait fitting and cross-fitted Meta-Fusion. See [Model](docs/MODEL.md).

The manuscript studies 11 traits in MaizeGEP and Y, PH and GM in Genomes to Fields (G2F). The corrected code accepts ID-aligned raw-value data with a hash manifest. See [Data](docs/DATA.md).

## Install and run a CPU demo

Use Python 3.11 for the proposed release environment.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python -m pip install -r requirements.txt pytest
python -m pytest -q
python -m experiments.run_main --dataset demo --fold 0
python -m scripts.aggregate_results artifacts/demo --demo --output-dir artifacts/demo/summary
```

The demo trains on generated inputs. Real-data sanity runs and full paper-candidate runs use separate result roots.

## Paper-candidate runs

Inspect the task plan before allocating compute. The paper configuration is a proposed correction protocol. Final settings and input provenance require manuscript-team review.

```bash
python -m experiments.run_main --dataset maizegep --data-dir data --dry-run
python -m experiments.run_main --dataset g2f --data-dir data --fold 0 --device cpu --resume
python -m experiments.run_baselines --dataset maizegep --variant RandomForest --data-dir data --fold 0
python -m experiments.run_genomic_comparators --dataset g2f --data-dir data --fold 0
python -m experiments.run_ablation --dataset maizegep --variant wo_CrossAttention --data-dir data --fold 0
python -m experiments.run_negative_transfer --dataset g2f --data-dir data --fold 0
python -m experiments.run_sensitivity --dataset maizegep --variant k9 --data-dir data --fold 0
python -m experiments.run_loeo --dataset maizegep --min-test-samples 80 --data-dir data --fold 0
python -m experiments.run_interpretability --dataset maizegep --data-dir data --fold 0
python -m scripts.verify_release --structure-only
```

Install `environment/comparators.txt` for XGBoost, LightGBM and TabNet. A CUDA build of PyTorch and `--device cuda:0` enable a single selected GPU. Estimate formal runtime from one measured fold with the final input and settings before scheduling the full matrix. See [Experiments and resources](docs/EXPERIMENTS.md) and [Code Ocean](docs/CODE_OCEAN.md).

Results are nested by model version, dataset, experiment, variant, seed, fold and content-derived run ID. Each completed run has predictions, metrics, configurations, split membership, preprocessing, checkpoint snapshots and artifact checksums. Completed runs are reused after verification. Use `--resume` with the original arguments to continue interrupted runs. A different configuration or data hash produces a separate run.

The [figure map](docs/REPRODUCIBILITY.md) links analysis to evidence. [Known limitations](docs/KNOWN_LIMITATIONS.md) and the [release blockers](reports/RELEASE_BLOCKERS.md) describe the formal release status.

## Citation and license

See [CITATION.cff](CITATION.cff) for the manuscript author list. The public repository README identifies the historical code as MIT. Confirm the corrected release's license notice and the data redistribution terms with the project owner. [Public HyperGEP repository](https://github.com/wjhhh2928/HyperGEP)
