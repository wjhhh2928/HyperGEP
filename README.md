# HyperGEP
# 🌽 HyperGEP: Decoupled Hypergraph Neural Network for Joint Gene-Environment ModelingOfficial PyTorch implementation of the framework described in "Decoupled integration of genomic and environmental information improves multi-trait prediction in maize".
HyperGEP provides a highly scalable computational solution for dissecting crop environmental adaptation and accelerating climate-resilient design breeding. By addressing the gradient conflicts across pleiotropic traits induced by early feature coupling, this decoupled two-stage deep learning framework accurately captures non-stationary spatial environmental gradients and resolves complex genotype-environment (G $\times$ E) interactions.

<p align="center">
    <img src="image/Fig. 1.png" alt="Model" width="85%">
</p>

# 🌟 Key Methodological Advances
Hypergraph Attention Network: Extracts higher-order climatic topologies and non-stationary spatial environmental gradients, operating in parallel with the genotypic encoder.
Decoupled Parallel Encoding: Utilizes independent modules for genotype and environment feature extraction, effectively preventing gradient conflicts across pleiotropic traits.
Predictive-Space Meta-Attention: Dynamically allocates task-specific weights to fuse distinct representations, mitigating task conflict and preventing negative transfer.
Structural Robustness: Demonstrates exceptional resilience against environmental noise, rigorously validated through strict leave-one-environment-out (LOEO) cross-validation across 11 agronomic traits.
Physical Interpretability: Gradient-based saliency mapping physically co-localizes the highest-weighted predictors with established developmental genes (e.g., GID1L2 and SPL) and identifies novel unannotated epistatic loci.

# 📂 Repository Structure
Based on the project architecture, the codebase is organized as follows:


# 🗂️ Datasets
The framework has been rigorously evaluated on large-scale maize cohorts:

MaizeGEP: Large-scale maize genotype-environment-phenotype dataset.

Genomes to Fields (G2F): Extensive multi-environment maize trials.

(Note: Please ensure you comply with the respective data usage agreements when downloading these datasets from their official repositories.)

# 🎓 Citation
If this framework assists your research in quantitative genetics or computational biology, please cite our manuscript:
Zhang Q., Wang, J., Zhang, Y., Li, B., Zhang, B., Zhao, X., Wang, Y., Piao, X., Zhang, B., & Wang, K..Decoupled integration of genomic and environmental information improves multi-trait prediction in maize.

# 📄 License
This project is licensed under the MIT License.
