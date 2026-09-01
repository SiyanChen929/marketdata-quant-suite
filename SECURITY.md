# Security and data handling

Report a suspected secret or data-license exposure privately to the repository owner. Do not open a public issue containing the value.

## Rules

- Read `MARKETDATA_TOKEN` and optional specialist-source credentials from the environment.
- Keep `.env`, key files, vendor responses, licensed event inputs, and account identifiers outside Git.
- Point `QUANT_DATA_HOME` to a directory outside this repository.
- Commit only source, tests, schemas, synthetic fixtures, small aggregate results, and reproducibility instructions.
- Rotate a credential immediately if it enters a commit or shared archive. Removing the current file does not remove it from Git history.
- Treat live-order integration as a separate reviewed boundary. This suite does not submit orders.
