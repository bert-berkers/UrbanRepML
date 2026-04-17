"""Build data pipeline for hierarchical cluster-brush visualization (Wave 1).

Purpose: Load netherlands ring_agg res9 embeddings, cluster with MiniBatchKMeans,
aggregate labels upward via H3 hierarchy, and emit a self-contained viz.html
artifact with embedded JSON for deck.gl rendering.

Lifetime: temporary
Stage: 3

Usage:
    uv run python scripts/one_off/cluster_brush_viz/build.py

Outputs (all written next to this script, never into data/):
    - viz.html: Wave 1 minimal shell (one flat H3HexagonLayer at res7)
    - data.json (sidecar, only if inline payload would exceed 12 MB)
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import h3  # ALLOWED: hierarchy traversal only (cell_to_parent), per .claude/rules/srai-spatial.md
import numpy as np
import pandas as pd

# Project infra
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.paths import StudyAreaPaths  # noqa: E402
from stage3_analysis.visualization.clustering_utils import (  # noqa: E402
    apply_pca_reduction,
    perform_minibatch_clustering,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STUDY_AREA = "netherlands"
MODEL = "ring_agg"
SOURCE_RES = 9
YEAR = "20mix"
TARGET_RESOLUTIONS = [5, 6, 7]
K = 10
PCA_COMPONENTS_DEFAULT = 32
PCA_COMPONENTS_FALLBACK = 64
PCA_VARIANCE_FLOOR = 0.9
INLINE_BYTES_BUDGET = 12 * 1024 * 1024  # 12 MB

# tab10 RGB tuples (matplotlib default categorical palette)
TAB10 = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy_nats(counts: list[int]) -> float:
    """Shannon entropy in nats from a list of class counts (non-normalized)."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h


def _hex_bounds_wgs84(rows: list[dict]) -> dict:
    """Compute a coarse WGS84 bounding box from a sample of hex centers.

    Uses h3.cell_to_latlng on a stride of the input (hierarchy-adjacent
    introspection via a single-cell centroid call; not a tessellation or
    neighborhood op). For the Wave 1 shell we only need approximate view
    bounds — deck.gl's H3HexagonLayer renders hex geometry natively.
    """
    if not rows:
        return {"minLon": 3.0, "maxLon": 7.5, "minLat": 50.7, "maxLat": 53.8}
    stride = max(1, len(rows) // 2000)
    lats, lons = [], []
    for row in rows[::stride]:
        lat, lon = h3.cell_to_latlng(row["hex"])
        lats.append(lat)
        lons.append(lon)
    return {
        "minLon": float(min(lons)),
        "maxLon": float(max(lons)),
        "minLat": float(min(lats)),
        "maxLat": float(max(lats)),
    }


def _aggregate_to_parent_res(
    res9_hexes: np.ndarray,
    res9_labels: np.ndarray,
    target_res: int,
) -> list[dict]:
    """Group res9 labels by their parent at target_res; return rows.

    Each row: {"hex": parent_hex, "cluster": majority_label,
               "n_children": int, "entropy": float}
    """
    print(f"  [res{target_res}] computing parents for {len(res9_hexes):,} res9 hexes...")
    t0 = time.time()
    # Compute parents in a tight loop; h3-py is C-backed so this is fast.
    parents = np.empty(len(res9_hexes), dtype=object)
    for i, hid in enumerate(res9_hexes):
        parents[i] = h3.cell_to_parent(hid, target_res)

    # Group-by parent -> majority vote + entropy
    # Use pandas for vectorized groupby of python-object hex strings.
    df = pd.DataFrame({"parent": parents, "label": res9_labels})
    rows: list[dict] = []
    grouped = df.groupby("parent", sort=False)
    n_groups = len(grouped)
    for parent_hex, sub in grouped:
        counts = Counter(sub["label"].tolist())
        majority_label, _ = counts.most_common(1)[0]
        n_children = int(sub.shape[0])
        entropy = _shannon_entropy_nats(list(counts.values()))
        rows.append({
            "hex": str(parent_hex),
            "cluster": int(majority_label),
            "n_children": n_children,
            "entropy": round(float(entropy), 4),
        })
    dt = time.time() - t0
    print(
        f"  [res{target_res}] {n_groups:,} parent hexes, aggregation {dt:.1f}s"
    )
    return rows


# ---------------------------------------------------------------------------
# Viz shell
# ---------------------------------------------------------------------------

_VIZ_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Cluster Brush Viz — Wave 1 (data pipeline)</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; width: 100%;
                background: #111; color: #ddd; font-family: system-ui, sans-serif; }}
  #deck {{ position: absolute; inset: 0; }}
  #title {{ position: absolute; top: 12px; left: 16px; z-index: 10;
            font-size: 14px; letter-spacing: 0.02em; opacity: 0.85;
            pointer-events: none; text-shadow: 0 1px 2px #000; }}
  #subtitle {{ position: absolute; top: 32px; left: 16px; z-index: 10;
               font-size: 11px; opacity: 0.55; pointer-events: none;
               text-shadow: 0 1px 2px #000; }}
</style>
</head>
<body>
<div id="deck"></div>
<div id="title">Cluster Brush Viz — Wave 1 (data pipeline)</div>
<div id="subtitle">res7 MiniBatchKMeans k={k_val}, tab10 palette — {n_res7} hexes</div>
<script src="https://unpkg.com/deck.gl@9.1.9/dist.min.js"></script>
<script>
{data_block}

const TAB10 = {tab10_json};
const res7Data = DATA.layers["7"];

const {{Deck, MapView}} = deck;
const {{H3HexagonLayer}} = deck;

const layer = new H3HexagonLayer({{
  id: 'res7-clusters',
  data: res7Data,
  pickable: false,
  stroked: false,
  filled: true,
  extruded: false,
  highPrecision: false,
  getHexagon: d => d.hex,
  getFillColor: d => TAB10[d.cluster] || [180, 180, 180],
  opacity: 0.85,
}});

new Deck({{
  parent: document.getElementById('deck'),
  views: new MapView({{repeat: false}}),
  initialViewState: {{
    longitude: 5.3,
    latitude: 52.2,
    zoom: 7,
    pitch: 0,
    bearing: 0,
  }},
  controller: true,
  layers: [layer],
  style: {{background: '#111'}},
}});
</script>
</body>
</html>
"""


def _render_viz_html(data_payload: dict, out_dir: Path) -> dict:
    """Write viz.html (and sidecar data.json if payload too large).

    Returns dict with chosen mode + sizes for stdout reporting.
    """
    data_json_str = json.dumps(data_payload, separators=(",", ":"))
    data_bytes = len(data_json_str.encode("utf-8"))
    n_res7 = len(data_payload["layers"]["7"])
    k_val = data_payload["k"]
    tab10_json = json.dumps(TAB10)

    if data_bytes > INLINE_BYTES_BUDGET:
        # Sidecar mode: write data.json, have viz.html fetch it
        sidecar = out_dir / "data.json"
        sidecar.write_text(data_json_str, encoding="utf-8")
        data_block = (
            "let DATA = null;\n"
            "// Sidecar mode (payload > 12 MB): fetched at runtime\n"
            "async function _boot() {\n"
            "  const resp = await fetch('./data.json');\n"
            "  DATA = await resp.json();\n"
            "  _init();\n"
            "}\n"
            "_boot();\n"
            "function _init() {\n"
        )
        # close _init at end
        html = _VIZ_HTML_TEMPLATE.format(
            k_val=k_val,
            n_res7=n_res7,
            data_block=data_block,
            tab10_json=tab10_json,
        ).replace(
            "new Deck({",
            "new Deck({",
        )
        # Append closing brace for _init
        html = html.replace(
            "</script>\n</body>",
            "}\n</script>\n</body>",
        )
        mode = "sidecar"
    else:
        data_block = f"const DATA = {data_json_str};"
        html = _VIZ_HTML_TEMPLATE.format(
            k_val=k_val,
            n_res7=n_res7,
            data_block=data_block,
            tab10_json=tab10_json,
        )
        mode = "inline"

    viz_path = out_dir / "viz.html"
    viz_path.write_text(html, encoding="utf-8")
    return {
        "mode": mode,
        "data_bytes": data_bytes,
        "viz_bytes": viz_path.stat().st_size,
        "viz_path": viz_path,
        "sidecar_path": (out_dir / "data.json") if mode == "sidecar" else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    paths = StudyAreaPaths(STUDY_AREA)
    src = paths.fused_embedding_file(MODEL, SOURCE_RES, YEAR)
    print(f"[load] {src}")
    if not src.exists():
        # No fallbacks (per memory/feedback_no_fallbacks.md): fail loud.
        raise FileNotFoundError(
            f"ring_agg res{SOURCE_RES} {YEAR} parquet not found at {src}. "
            "No fallback — refusing to silently substitute another embedding."
        )

    df = pd.read_parquet(src)
    print(f"[load] shape={df.shape} index={df.index.name!r}")

    # Sanity: region_id is the index, all columns are numeric features
    if df.index.name != "region_id":
        # Try common alternates if upstream schema drifts
        for cand in ("region_id", "h3", "hex"):
            if cand in df.columns:
                df = df.set_index(cand)
                break
        else:
            raise ValueError(
                f"Expected region_id index or an h3-like column; got index={df.index.name!r} "
                f"cols[:5]={list(df.columns[:5])}"
            )
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"[load] dropping non-feature columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    hex_ids = df.index.astype(str).to_numpy()
    X = df.to_numpy(dtype=np.float32, copy=False)
    print(f"[load] {X.shape[0]:,} hexes × {X.shape[1]}D features")

    # PCA
    n_comp = PCA_COMPONENTS_DEFAULT
    X_pca, pca = apply_pca_reduction(X, n_components=n_comp)
    var_retained = float(pca.explained_variance_ratio_.sum())
    if var_retained < PCA_VARIANCE_FLOOR:
        print(
            f"[pca] variance {var_retained:.3f} < {PCA_VARIANCE_FLOOR}; "
            f"bumping to {PCA_COMPONENTS_FALLBACK}"
        )
        n_comp = PCA_COMPONENTS_FALLBACK
        X_pca, pca = apply_pca_reduction(X, n_components=n_comp)
        var_retained = float(pca.explained_variance_ratio_.sum())

    print(f"[pca] n_components={n_comp} variance_retained={var_retained:.3f}")

    # Clustering
    clusters = perform_minibatch_clustering(X_pca, [K], standardize=False)
    labels = clusters[K].astype(np.int64)
    uniq, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(uniq, counts)}
    print(f"[cluster] k={K} distribution={dist}")

    # Hierarchical aggregation
    layers: dict[str, list[dict]] = {}
    for r in TARGET_RESOLUTIONS:
        rows = _aggregate_to_parent_res(hex_ids, labels, r)
        layers[str(r)] = rows
        # distribution at this resolution
        lbl_counter = Counter(row["cluster"] for row in rows)
        print(
            f"[agg]  res{r}: {len(rows):,} hexes, "
            f"cluster_dist={dict(sorted(lbl_counter.items()))}"
        )

    bounds = _hex_bounds_wgs84(layers[str(max(TARGET_RESOLUTIONS))])
    print(f"[bounds] {bounds}")

    payload = {
        "k": K,
        "source": {
            "study_area": STUDY_AREA,
            "model": MODEL,
            "resolution": SOURCE_RES,
            "year": YEAR,
            "pca_components": n_comp,
            "pca_variance_retained": round(var_retained, 4),
        },
        "bounds": bounds,
        "layers": layers,
    }

    out = _render_viz_html(payload, _SCRIPT_DIR)
    print(
        f"[write] viz.html mode={out['mode']} "
        f"data_bytes={out['data_bytes']:,} ({out['data_bytes']/1e6:.2f} MB) "
        f"viz_bytes={out['viz_bytes']:,} ({out['viz_bytes']/1e6:.2f} MB)"
    )
    if out["sidecar_path"] is not None:
        print(f"[write] sidecar: {out['sidecar_path']}")
    print(f"[done] total={time.time() - t_start:.1f}s")
    print(f"[done] open in browser: {out['viz_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
