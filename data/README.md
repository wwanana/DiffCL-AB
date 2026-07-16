# Dataset layout

Place the processed datasets in the following locations unless you override the paths in the launch scripts:

```text
data/
├── sigmul_data/
│   └── AB1101_single.csv
└── identity_data/
    ├── csv_AB645/
    │   ├── AB645_train.csv
    │   └── AB645_test.csv
    └── csv_S1131/
        ├── S1131_train.csv
        └── S1131_test.csv
```

The model expects wild-type/mutant antibody and antigen sequences plus a numerical ΔΔG label. Preserve the column names expected by `utils/data_utils.py` or `utils/sig_mul_utils.py`.

AB645 and AB1101 originate from AB-Bind; S1131 is derived from SKEMPI. Verify and comply with the original dataset licenses and citation requirements before redistribution. If redistribution is not permitted, provide download/preprocessing instructions instead of committing the CSV files.
