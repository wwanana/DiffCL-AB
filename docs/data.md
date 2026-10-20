# Data provenance and conventions

Files larger than 384 KiB are distributed as deterministic gzip streams split into small Git-tracked parts under `assets/`. `python -m scripts.prepare_assets` reconstructs the original paths before analysis. `assets/manifest.json` records the uncompressed byte count and SHA-256 plus each part's SHA-256. The reconstructed bytes are identical to the audited deposited files; compression does not transform sequences, labels, coordinates, or tensors.

## Source and curation

`data/source/` contains the CSVs recovered from the supplied research project, not newly queried measurements. Source-level assay values are retained. `scripts.prepare_data` rebuilds the processed datasets and writes a machine-readable report. `source_row` identifies the original CSV row; `record_id` identifies a row in the model-valid processed dataset. Neither identifier is a model feature.

| Dataset | Source rows | Processed rows | PDB complexes | Repeated-input rows | Conflicting-label input groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| AB645 | 645 | 645 | 29 | 6 | 3 |
| AB1101 | 1,100 | 1,100 | 32 | 12 | 6 |
| S1131 | 1,131 | 1,130 | 112 | 0 | 0 |

Repeated-input counts include all members of repeated four-sequence input groups. Labels are not averaged or silently deduplicated. Ordinary CV keeps each repeated-input group in one fold.

AB1101's 645 single-substitution records are matched to AB645 using PDB, mutation annotation, ΔΔG, and stable occurrence order for repeated keys. The eight sequence-component fields are reconciled from the corresponding AB645 rows; 27 AB1101 records change at the sequence-input level. No ΔΔG label or multiple-substitution record is modified. The affected original row identifiers are in `data/processed/preparation.json`.

S1131 source row 311 has no observable WT/MT sequence change and is excluded from the model-valid input. It remains available in `data/processed/S1131_excluded.csv`. Remaining records require finite labels, nonempty partner sequences, equal WT/MT lengths within each component, and at least one encoded residue substitution.

The core loader removes non-letter characters and converts sequences to uppercase. AB component order is light chain followed by heavy chain for the antibody role, and antigen A followed by antigen B for the antigen role. Empty optional chains are allowed. Component boundaries are retained when building mutation-site and neighborhood masks; the global encoder processes the concatenated partner sequence. For S1131, `a` and `b` are general protein-partner roles.

## Experimental labels and mutation depth

The supplied AB-Bind convention assigns ΔΔG = +8 kcal mol⁻¹ to designated nonbinders. There are 27 such labels in AB645, 50 in AB1101, and 23 in the 455-record MutD test set. The archived benchmark retains them. `analyze_predictions.py` reports both the complete test set and the 432-record subset excluding these assignments. The subset is a diagnostic sensitivity analysis, not a replacement benchmark or a model-selection set.

The mutation-depth split follows the source mutation annotations. Within the 455 annotation-defined multiple-substitution records, 275 have different annotation and encoded-component counts; 157 encode only one distinct WT/MT residue change. Most count differences occur in repeated-chain complexes (1T83 and 3WJJ); additional discrepancies occur in 1JTG and 2NY7. The release retains the deposited memberships and sequences to preserve the evaluated benchmark. The analysis exports both counts and separately summarizes records with at least two encoded changes. Annotation-defined multiple substitutions should not be described as uniformly multiple distinct changes in the model's sequence inputs.

## Homology resources

`data/homology/AB1101/` preserves the direct MMseqs2 hit table from the supplied low-homology mutation-depth experiment. Its FASTA is reconstructed losslessly from the deposited sequence manifest (983 unique component sequences). It covers both curated AB datasets. The same hit table can be filtered at 30% for partitioning and at 90% for contrastive masking; these thresholds have separate purposes.

`data/homology/S1131/` is generated from the current model-valid S1131 dataset using the released MMseqs2 builder. `provenance.json` records the executable version, exact command, and hashes. Regenerating searches with another MMseqs2 version can change borderline relationships, so the deposited hit tables and manifests are the fixed reference for this release. Direct-hit filtering is not a guarantee that every remote evolutionary relationship has been detected.

## Checkpoint and structure assets

`checkpoints/mutd_seed42.pt` is the supplied trainable-state checkpoint, containing 137 tensors and 5,820,801 parameters. Its SHA-256 is `0eb1a5acb0b419cf3daaeef606c874693f1d67b65f728c6a79c6a77c3d6e6183`. Frozen ESM weights are obtained separately. The archived checkpoint retains its original configuration metadata; the portable runner overrides machine-specific paths. Its companion normalization file is recalculated from the deposited 645-record refit CSV and is verified during the release audit.

The three structure files retain their PDB fixed-column coordinates. Case selection and prediction tables are deposited in `data/cases/`. The distance audit checks WT residue identities and all 22 reported shortest heavy-atom distances against these coordinates. Distances are geometric separations, not automatically assigned hydrogen bonds. No experimentally determined mutant conformations or mechanistic energy decompositions are claimed.

The cited AB-Bind, SKEMPI, BindProfX, ESM2, MMseqs2, Mamba, and PDB resources retain their respective attribution and distribution terms. Dataset integrity checks establish correspondence to the deposited inputs; they do not independently remeasure binding affinity or remove heterogeneity among source assays.
