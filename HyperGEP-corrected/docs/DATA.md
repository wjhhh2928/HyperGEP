# Data contract

Place three CSV files and one JSON manifest in `data/maizegep/` or `data/g2f/`. Phenotypes contain unique `sample_id`, `genotype_id`, `environment_id`, optional `site_id`, and the configured trait names. Repeated genotype/environment observations require unique source sample IDs. Genotypes contain one row per genotype with ordered numeric SNP dosages. Environment features contain one row per environment. Supply features available at the declared prediction time. Preserve the original observation units and leave missing labels blank. Tables may use TSV by configuring their paths.

The loader validates unique keys, checks complete joins, preserves leading zeros and retains unlabeled records for prediction. It rejects unmatched identifier references. Each run records counts and duplicate genotype/environment pairs. Already normalized files require source verification and a raw-data recovery decision.

`provenance.json` requires `dataset_version`, `origin`, `license`, `preprocessing` set to `raw`, `row_alignment_verified` set to true, `label_provenance`, `environment_availability`, and `sha256` containing the hashes of the three tables by asset type. Preserve source checksums, sizes, marker assembly/layout and import recipes in the asset version record. For regional attribution, supply a reviewed `markers.csv` with `marker_id`, `chromosome`, `position` and `reference_build`.

Obtain the exact research release and its filtering recipe from the manuscript team. The provider links describe acquisition channels and do not identify the exact paper cohort. [MaizeGEP portal](http://user.ebreed.cn:9992/scb/), [G2F Resources](https://www.genomes2fields.org/resources/), [G2F competition archive](https://doi.org/10.25739/tq5e-ak26)

The separate local G2F smoke adapter selects a deterministic raw-value excerpt, records the provenance of the used genotype prefix and independently hashes staged files. It cannot establish the version of the paper input or redistribution rights.
