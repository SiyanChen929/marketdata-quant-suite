from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from lfm_helpers import make_bars, panel, synthetic_bars  # noqa: F401 - fixtures
from llm_factor_mining.data import PANEL_FIELDS, Panel, PanelError, load_confirmed_panel, panel_from_bars
from quant_marketdata import MarketDataStore
from quant_marketdata.exceptions import DataContractError


def test_panel_shapes_and_alignment(synthetic_bars: pd.DataFrame) -> None:
    panel = panel_from_bars(synthetic_bars)
    assert panel.close.shape == (80, 12)
    for name in PANEL_FIELDS:
        frame = panel.field(name)
        assert frame.index.equals(panel.dates) and frame.columns.equals(panel.close.columns)
        assert frame.dtypes.eq("float64").all()
    assert panel.symbols == tuple(sorted(panel.symbols))
    assert panel.sources == ("synthetic",)
    assert panel.finality == "confirmed"
    first = synthetic_bars.iloc[0]
    assert panel.close.loc[pd.Timestamp(first["date"]), first["symbol"]] == first["close"]
    with pytest.raises(KeyError):
        panel.field("returns")


def test_missing_bars_become_nan(synthetic_bars: pd.DataFrame) -> None:
    dropped = synthetic_bars.drop(index=[5, 6]).reset_index(drop=True)
    panel = panel_from_bars(dropped)
    assert int(panel.close.isna().to_numpy().sum()) == 2
    assert int((~panel.available).to_numpy().sum()) == 2


def test_rejects_non_confirmed_rows(synthetic_bars: pd.DataFrame) -> None:
    mixed = synthetic_bars.copy()
    mixed.loc[3, "finality"] = "provisional"
    with pytest.raises(PanelError, match="confirmed"):
        panel_from_bars(mixed)
    with pytest.raises(PanelError, match="confirmed"):
        panel_from_bars(make_bars(10, 3, finality="provisional"))


def test_rejects_missing_labels_intraday_and_bad_frames(synthetic_bars: pd.DataFrame) -> None:
    with pytest.raises(PanelError, match="finality"):
        panel_from_bars(synthetic_bars.drop(columns="finality"))
    with pytest.raises(PanelError, match="source"):
        panel_from_bars(synthetic_bars.drop(columns="source"))
    intraday = synthetic_bars.copy()
    intraday["date"] = pd.to_datetime(intraday["date"]) + pd.Timedelta(hours=10)
    with pytest.raises(PanelError, match="daily"):
        panel_from_bars(intraday)
    weekly = synthetic_bars.copy()
    weekly.attrs["marketdata_resolution"] = "W"
    with pytest.raises(PanelError, match="daily"):
        panel_from_bars(weekly)
    broken = synthetic_bars.copy()
    broken.loc[0, "high"] = broken.loc[0, "low"] / 2.0
    with pytest.raises(DataContractError):
        panel_from_bars(broken)
    with pytest.raises(PanelError):
        panel_from_bars(synthetic_bars.iloc[:0])
    with pytest.raises(PanelError):
        panel_from_bars("not a frame")  # type: ignore[arg-type]


def test_content_hash_is_deterministic_and_sensitive(synthetic_bars: pd.DataFrame) -> None:
    first = panel_from_bars(synthetic_bars)
    shuffled = synthetic_bars.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert panel_from_bars(shuffled).content_sha256 == first.content_sha256
    changed = synthetic_bars.copy()
    changed.loc[10, "volume"] += 1.0
    assert panel_from_bars(changed).content_sha256 != first.content_sha256
    assert first.manifest_sha256 is None
    assert len(first.content_sha256) == 64


def test_manifest_hash_is_kept_from_attrs(synthetic_bars: pd.DataFrame) -> None:
    tagged = synthetic_bars.copy()
    tagged.attrs["marketdata_manifest_sha256"] = "ab" * 32
    panel = panel_from_bars(tagged)
    assert panel.manifest_sha256 == "ab" * 32
    assert panel.provenance()["manifest_sha256"] == "ab" * 32


def test_load_confirmed_panel_from_store(tmp_path, synthetic_bars: pd.DataFrame) -> None:
    store = MarketDataStore(tmp_path)
    store.write_bars(synthetic_bars, finality="confirmed")
    store.write_bars(make_bars(10, 2, seed=1, finality="provisional"), finality="provisional")
    panel = load_confirmed_panel(store, ["S000", "S001", "s002", "ZZZ"], "2031-01-02", "2031-02-28")
    assert panel.symbols == ("S000", "S001", "S002")
    assert panel.dates.max() <= pd.Timestamp("2031-02-28")
    manifest = tmp_path / "manifests" / "marketdata" / "confirmed.json"
    assert panel.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert panel.metadata["missing_symbols"] == ["ZZZ"]
    provenance = panel.provenance()
    assert provenance["finality"] == "confirmed" and provenance["n_symbols"] == 3
    with pytest.raises(PanelError, match="no bars"):
        load_confirmed_panel(store, ["ZZZ"], "2031-01-02", "2031-02-28")


def test_panel_rejects_misaligned_fields(panel: Panel) -> None:
    with pytest.raises(PanelError, match="aligned"):
        Panel(
            open=panel.open.iloc[:-1],
            high=panel.high,
            low=panel.low,
            close=panel.close,
            volume=panel.volume,
            sources=panel.sources,
            content_sha256=panel.content_sha256,
        )
    with pytest.raises(PanelError, match="confirmed"):
        Panel(
            open=panel.open,
            high=panel.high,
            low=panel.low,
            close=panel.close,
            volume=panel.volume,
            sources=panel.sources,
            content_sha256=panel.content_sha256,
            finality="provisional",
        )
    reversed_close = panel.close.iloc[::-1]
    with pytest.raises(PanelError, match="increasing"):
        Panel(
            open=reversed_close,
            high=reversed_close,
            low=reversed_close,
            close=reversed_close,
            volume=reversed_close,
            sources=panel.sources,
            content_sha256="x",
        )
    assert np.isfinite(panel.close.to_numpy()).all()
