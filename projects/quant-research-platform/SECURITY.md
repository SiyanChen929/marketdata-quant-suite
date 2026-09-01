# Security and data handling

- Load credentials from environment variables or an operating-system secret store.
- Never commit `.env`, API-key files, raw vendor data, positions, broker exports, or logs.
- Run a secret scan against the staged index before the first push and every release.
- Treat MarketData, index-provider, exchange, and broker datasets as licensed inputs; publish code and synthetic fixtures unless redistribution rights are explicit.
- If a credential may have been copied, committed, shared, or backed up externally, rotate it before publication.

Report security issues privately to the repository owner. Do not open a public issue containing credentials or account details.
