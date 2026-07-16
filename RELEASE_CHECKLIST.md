# GitHub release checklist

## Mandatory before publishing

- [ ] Confirm the final repository name and project title.
- [ ] Replace every placeholder in `CITATION.cff`.
- [ ] Verify the upstream ProtAttBA code license and choose a compatible license.
- [ ] Verify whether processed AB645, AB1101, and S1131 CSV files may be redistributed.
- [ ] Remove ESM2 weights, checkpoints, experiment results, and temporary files.
- [ ] Search for API keys, tokens, passwords, usernames, email addresses, and server paths.
- [ ] Replace hard-coded `/root/...` paths with command-line arguments or relative paths.
- [ ] Replace `argparse type=bool` with a real `str2bool` parser.
- [ ] Run `python -m compileall` on the release directory.
- [ ] Run `python debug_forward.py` in a clean environment.
- [ ] Run at least one fold or a reduced-size CV smoke test.
- [ ] Export the actual environment to `requirements-lock.txt`.
- [ ] Confirm that README commands match the final script argument names.
- [ ] Add the final paper citation and DOI after acceptance.

## Recommended

- [ ] Add a model architecture figure under `assets/`.
- [ ] Add a small non-sensitive example input file.
- [ ] Add expected output examples.
- [ ] Add GitHub issue and pull-request templates.
- [ ] Create a tagged release matching the paper version.
