# quant-marketdata

`quant-marketdata` is the shared data boundary for the MarketData Quant Suite.
It provides an injectable MarketData.app client, a checksum-verified external
Parquet store, and one canonical long-form bar schema:

```text
date, symbol, open, high, low, close, volume, source, finality
```

Confirmed history and provisional observations are physically separated. A
formal write cannot contain a provisional row, and a same-day New York daily
candle cannot be labeled confirmed before 16:15 ET.

## Configuration

Credentials and storage paths come only from environment variables:

```bash
export MARKETDATA_TOKEN="your-token"
export QUANT_DATA_HOME="/absolute/path/to/external/quant-data"
```

No token is included in request hashes, manifests, errors, or persisted data.
`QUANT_DATA_HOME` should remain outside every Git checkout.

## Stable API

```python
from quant_marketdata import MarketDataClient, MarketDataStore, wide_close

store = MarketDataStore()
client = MarketDataClient(store=store)

daily = client.get_daily_bars(
    "AAPL",
    "2024-01-01",
    "2024-12-31",
    finality="confirmed",
)

panel = client.get_bulk_daily_bars(
    ["AAPL", "MSFT"],
    "2024-01-01",
    "2024-12-31",
)
closes = wide_close(panel)

five_minute = client.get_stock_bars(
    "AAPL",
    "2024-06-03",
    "2024-06-03",
    resolution="5",
    finality="provisional",
)

chain = client.get_option_chain(
    "AAPL",
    date="2024-06-03",
    side="call",
    min_open_interest=100,
)

confirmed = store.read_bars(
    ["AAPL", "MSFT"],
    start="2024-01-01",
    end="2024-12-31",
    finality="confirmed",
    resolution="D",
)
```

Daily requests explicitly set both `adjustsplits=true` and
`adjustdividends=true`; the adjustment policy is recorded in the request
manifest. Pass `refresh=True` to bypass an exact-request cache object.

## External layout

```text
$QUANT_DATA_HOME/
  raw/marketdata/confirmed/request-cache/
  raw/marketdata/provisional/request-cache/
  lake/confirmed/bars/resolution=D/year=YYYY/bars.parquet
  staging/provisional/bars/resolution=5/year=YYYY/bars.parquet
  manifests/marketdata/confirmed.json
  manifests/marketdata/provisional.json
```

Request keys are deterministic SHA-256 hashes of provider, dataset, symbol,
date range, resolution, adjustment policy, finality, and schema version.
Bar resolutions also occupy separate Parquet partitions, so daily, hourly,
and minute observations cannot overwrite one another.
Parquet objects and canonical partitions are written through same-directory
temporary files and atomic replacement. Manifests carry SHA-256 checksums and
are verified on reads.

Option chains use the same injected transport, environment-only token, request
hashing, atomic writes, and checksum verification. They remain reference tables
under `raw/marketdata/options/request-cache` and never enter the bar lake.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The test suite uses synthetic payloads and an injected HTTP session. It makes no
network requests and needs no real credentials.
