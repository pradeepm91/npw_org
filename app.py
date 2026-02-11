import datetime as _dt
import gc
import json
import math
from collections import defaultdict, deque
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from pyvis.network import Network

DATA_DIR = Path("Rels_Data")
REL_ZIP = DATA_DIR / "CSV_RELATIONSHIPS.zip"
ATTR_ACTIVE_ZIP = DATA_DIR / "CSV_ATTRIBUTES_ACTIVE.zip"
ATTR_CLOSED_ZIP = DATA_DIR / "CSV_ATTRIBUTES_CLOSED.zip"
ATTR_BRANCHES_ZIP = DATA_DIR / "CSV_ATTRIBUTES_BRANCHES.zip"
TRANSFORM_ZIP = DATA_DIR / "CSV_TRANSFORMATIONS.zip"
LOCAL_CACHE_DIR = DATA_DIR / "_cache"
RELS_CACHE_SCHEMA_VERSION = 2

st.set_page_config(page_title="NPW Org-Chart Explorer", layout="wide")

# ----------------------------
# Utility functions
# ----------------------------

def _to_int(series: pd.Series, default: int = 0) -> pd.Series:
    # Keep integer columns compact for Streamlit Cloud memory limits.
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(np.int32)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("#").strip() for c in df.columns]
    return df


def _read_zip_csv(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as f:
            return pd.read_csv(f, low_memory=False)


@st.cache_data(show_spinner=False, max_entries=1)
def load_attributes(active_zip: Path, closed_zip: Path) -> pd.DataFrame:
    df_active = _read_zip_csv(active_zip)
    df_closed = _read_zip_csv(closed_zip)
    df = pd.concat([df_active, df_closed], ignore_index=True)
    df = _normalize_columns(df)

    keep_cols = [
        "ID_RSSD",
        "NM_SHORT",
        "NM_LGL",
        "BROAD_REG_CD",
        "BHC_IND",
        "SLHC_IND",
        "IHC_IND",
        "EST_TYPE_CD",
        "DOMESTIC_IND",
        "DT_START",
        "DT_END",
        "DT_EXIST_CMNC",
        "DT_EXIST_TERM",
        "BANK_CNT",
        "CITY",
        "STATE_ABBR_NM",
        "ENTITY_TYPE",
        "CHTR_TYPE_CD",
        "ORG_TYPE_CD",
        "ID_RSSD_HD_OFF",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Cast types
    df["ID_RSSD"] = _to_int(df["ID_RSSD"])
    for c in [
        "BROAD_REG_CD",
        "BHC_IND",
        "SLHC_IND",
        "IHC_IND",
        "EST_TYPE_CD",
        "DT_START",
        "DT_END",
        "DT_EXIST_CMNC",
        "DT_EXIST_TERM",
        "BANK_CNT",
        "CHTR_TYPE_CD",
        "ORG_TYPE_CD",
        "ID_RSSD_HD_OFF",
    ]:
        if c in df.columns:
            df[c] = _to_int(df[c])

    for c in ["NM_SHORT", "NM_LGL", "CITY", "STATE_ABBR_NM", "ENTITY_TYPE", "DOMESTIC_IND"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    return df


@st.cache_data(show_spinner=False, max_entries=1)
def load_branches(branches_zip: Path) -> pd.DataFrame:
    if not branches_zip.exists():
        return pd.DataFrame()
    df = _read_zip_csv(branches_zip)
    df = _normalize_columns(df)
    keep_cols = [
        "ID_RSSD",
        "ID_RSSD_HD_OFF",
        "NM_SHORT",
        "NM_LGL",
        "DT_START",
        "DT_END",
        "EST_TYPE_CD",
        "BROAD_REG_CD",
        "CITY",
        "STATE_ABBR_NM",
        "DOMESTIC_IND",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df["ID_RSSD"] = _to_int(df["ID_RSSD"])
    if "ID_RSSD_HD_OFF" in df.columns:
        df["ID_RSSD_HD_OFF"] = _to_int(df["ID_RSSD_HD_OFF"])
    for c in ["DT_START", "DT_END", "EST_TYPE_CD", "BROAD_REG_CD"]:
        if c in df.columns:
            df[c] = _to_int(df[c])
    for c in ["NM_SHORT", "NM_LGL", "CITY", "STATE_ABBR_NM", "DOMESTIC_IND"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()
    return df


@st.cache_data(show_spinner=False, max_entries=1)
def load_relationships(rel_zip: Path) -> pd.DataFrame:
    df = _read_zip_csv(rel_zip)
    df = _normalize_columns(df)
    keep_cols = [
        "ID_RSSD_PARENT",
        "ID_RSSD_OFFSPRING",
        "D_DT_START",
        "D_DT_END",
        "DT_START",
        "DT_END",
        "RELN_LVL",
        "CTRL_IND",
        "REG_IND",
        "PCT_EQUITY",
        "PCT_EQUITY_FORMAT",
        "PCT_EQUITY_BRACKET",
        "PCT_OTHER",
        "OTHER_BASIS_IND",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    for c in [
        "ID_RSSD_PARENT",
        "ID_RSSD_OFFSPRING",
        "DT_START",
        "DT_END",
        "RELN_LVL",
        "CTRL_IND",
        "REG_IND",
        "OTHER_BASIS_IND",
    ]:
        if c in df.columns:
            df[c] = _to_int(df[c])

    if "PCT_EQUITY" in df.columns:
        df["PCT_EQUITY"] = pd.to_numeric(df["PCT_EQUITY"], errors="coerce")
    if "PCT_OTHER" in df.columns:
        df["PCT_OTHER"] = pd.to_numeric(df["PCT_OTHER"], errors="coerce")

    for c in ["PCT_EQUITY_FORMAT", "PCT_EQUITY_BRACKET"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    for c in ["D_DT_START", "D_DT_END"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    return df


@st.cache_data(show_spinner=False, max_entries=1)
def expand_relationships_yearly(
    rels: pd.DataFrame,
    min_year: int,
    max_year: int,
) -> tuple[pd.DataFrame, dict]:
    df = rels.copy()
    stats: dict[str, int] = {"raw_rows": int(len(df))}

    # Basic data checks and cleanup
    bad_ids = (df["ID_RSSD_PARENT"] <= 0) | (df["ID_RSSD_OFFSPRING"] <= 0)
    stats["dropped_bad_ids"] = int(bad_ids.sum())
    df = df[~bad_ids]

    bad_dates = (df["DT_START"] <= 0) | (df["DT_END"] <= 0)
    stats["dropped_bad_dates"] = int(bad_dates.sum())
    df = df[~bad_dates]

    reversed_dates = df["DT_START"] > df["DT_END"]
    stats["dropped_reversed_dates"] = int(reversed_dates.sum())
    df = df[~reversed_dates]

    self_loops = df["ID_RSSD_PARENT"] == df["ID_RSSD_OFFSPRING"]
    stats["dropped_self_loops"] = int(self_loops.sum())
    df = df[~self_loops]

    # Build year interval primarily from D_DT_START / D_DT_END as requested.
    # Fallback to DT_START / DT_END when D_DT values are missing.
    def _year_from_d_dt(s: pd.Series) -> pd.Series:
        if s is None:
            return pd.Series(0, index=df.index, dtype=int)
        yr = pd.to_numeric(s.astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")
        return yr.fillna(0).astype(int)

    y_start_ddt = _year_from_d_dt(df["D_DT_START"]) if "D_DT_START" in df.columns else pd.Series(0, index=df.index)
    y_end_ddt = _year_from_d_dt(df["D_DT_END"]) if "D_DT_END" in df.columns else pd.Series(0, index=df.index)
    y_start_dt = (df["DT_START"] // 10000).astype(int)
    y_end_dt = (df["DT_END"] // 10000).astype(int)

    y_start = y_start_ddt.where(y_start_ddt > 0, y_start_dt)
    y_end = y_end_ddt.where(y_end_ddt > 0, y_end_dt)
    y_end = y_end.replace(9999, max_year)

    y_start = y_start.clip(lower=min_year, upper=max_year).astype(int)
    y_end = y_end.clip(lower=min_year, upper=max_year).astype(int)
    df["YEAR_START"] = y_start
    df["YEAR_END"] = y_end

    valid_span = df["YEAR_START"] <= df["YEAR_END"]
    stats["dropped_out_of_range"] = int((~valid_span).sum())
    df = df[valid_span]

    span = (df["YEAR_END"] - df["YEAR_START"] + 1).astype(int)
    stats["clean_rows"] = int(len(df))
    stats["expanded_rows_pre_dedup"] = int(span.sum())

    rep_idx = np.repeat(np.arange(len(df), dtype=np.int64), span.to_numpy())
    years = np.concatenate(
        [np.arange(s, e + 1, dtype=np.int16) for s, e in zip(df["YEAR_START"].to_numpy(), df["YEAR_END"].to_numpy())]
    ).astype(int)

    exp = df.iloc[rep_idx].copy()
    exp["YEAR"] = years

    # Keep latest relationship state if multiple rows land in same year for same edge.
    exp = exp.sort_values(["YEAR", "ID_RSSD_PARENT", "ID_RSSD_OFFSPRING", "RELN_LVL", "DT_START"])
    exp = exp.drop_duplicates(["YEAR", "ID_RSSD_PARENT", "ID_RSSD_OFFSPRING", "RELN_LVL"], keep="last")

    # Keep only columns required downstream and downcast types for memory safety.
    keep_cols = [
        "YEAR",
        "ID_RSSD_PARENT",
        "ID_RSSD_OFFSPRING",
        "DT_START",
        "RELN_LVL",
        "CTRL_IND",
        "REG_IND",
        "PCT_EQUITY",
        "PCT_EQUITY_FORMAT",
        "PCT_EQUITY_BRACKET",
        "OTHER_BASIS_IND",
    ]
    keep_cols = [c for c in keep_cols if c in exp.columns]
    exp = exp[keep_cols].copy()
    for c in ["YEAR", "ID_RSSD_PARENT", "ID_RSSD_OFFSPRING", "DT_START"]:
        if c in exp.columns:
            exp[c] = exp[c].astype(np.int32, copy=False)
    for c in ["RELN_LVL", "CTRL_IND", "REG_IND", "OTHER_BASIS_IND"]:
        if c in exp.columns:
            exp[c] = exp[c].astype(np.int16, copy=False)
    for c in ["PCT_EQUITY_FORMAT", "PCT_EQUITY_BRACKET"]:
        if c in exp.columns:
            exp[c] = exp[c].astype("category")

    stats["expanded_rows"] = int(len(exp))

    return exp, stats


def load_or_build_relationships_yearly(
    rels: pd.DataFrame,
    min_year: int,
    max_year: int,
    rel_zip: Path,
) -> tuple[pd.DataFrame, dict]:
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parq = LOCAL_CACHE_DIR / f"rels_yearly_{min_year}_{max_year}.parquet"
    meta = LOCAL_CACHE_DIR / f"rels_yearly_{min_year}_{max_year}.json"

    src_mtime_ns = int(rel_zip.stat().st_mtime_ns) if rel_zip.exists() else 0
    src_size = int(rel_zip.stat().st_size) if rel_zip.exists() else 0

    if parq.exists() and meta.exists():
        try:
            meta_obj = json.loads(meta.read_text(encoding="utf-8"))
            if (
                int(meta_obj.get("schema_version", -1)) == RELS_CACHE_SCHEMA_VERSION
                and int(meta_obj.get("source_mtime_ns", -1)) == src_mtime_ns
                and int(meta_obj.get("source_size", -1)) == src_size
                and int(meta_obj.get("min_year", -1)) == int(min_year)
                and int(meta_obj.get("max_year", -1)) == int(max_year)
            ):
                cached = pd.read_parquet(parq)
                return cached, dict(meta_obj.get("stats", {}))
        except Exception:
            pass

    exp, stats = expand_relationships_yearly(rels, min_year, max_year)
    try:
        exp.to_parquet(parq, index=False, compression="zstd")
        meta_obj = {
            "schema_version": RELS_CACHE_SCHEMA_VERSION,
            "source_mtime_ns": src_mtime_ns,
            "source_size": src_size,
            "min_year": int(min_year),
            "max_year": int(max_year),
            "stats": stats,
        }
        meta.write_text(json.dumps(meta_obj), encoding="utf-8")
    except Exception:
        pass
    return exp, stats


@st.cache_data(show_spinner=False, max_entries=1)
def load_transformations(trans_zip: Path) -> pd.DataFrame:
    if not trans_zip.exists():
        return pd.DataFrame()
    df = _read_zip_csv(trans_zip)
    df = _normalize_columns(df)
    keep_cols = [
        "ID_RSSD_PREDECESSOR",
        "ID_RSSD_SUCCESSOR",
        "DT_TRANS",
        "TRNSFM_CD",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    for c in ["ID_RSSD_PREDECESSOR", "ID_RSSD_SUCCESSOR", "DT_TRANS", "TRNSFM_CD"]:
        if c in df.columns:
            df[c] = _to_int(df[c])
    df = df[df["DT_TRANS"] > 0]
    return df


TOP20_BHCS = [
    {"rank": 1, "name": "JPMORGAN CHASE & CO.", "rssd_id": 1039502, "location": "NEW YORK, NY", "assets": 4560205000},
    {"rank": 2, "name": "BANK OF AMERICA CORPORATION", "rssd_id": 1073757, "location": "CHARLOTTE, NC", "assets": 3403716000},
    {"rank": 3, "name": "CITIGROUP INC.", "rssd_id": 1951350, "location": "NEW YORK, NY", "assets": 2642475000},
    {"rank": 4, "name": "WELLS FARGO & COMPANY", "rssd_id": 1120754, "location": "SAN FRANCISCO, CA", "assets": 2062977000},
    {"rank": 5, "name": "GOLDMAN SACHS GROUP, INC., THE", "rssd_id": 2380443, "location": "NEW YORK, NY", "assets": 1807982000},
    {"rank": 6, "name": "MORGAN STANLEY", "rssd_id": 2162966, "location": "NEW YORK, NY", "assets": 1364806000},
    {"rank": 7, "name": "U.S. BANCORP", "rssd_id": 1119794, "location": "MINNEAPOLIS, MN", "assets": 695357000},
    {"rank": 8, "name": "CAPITAL ONE FINANCIAL CORPORATION", "rssd_id": 2277860, "location": "MCLEAN, VA", "assets": 661876810},
    {"rank": 9, "name": "PNC FINANCIAL SERVICES GROUP, INC., THE", "rssd_id": 1069778, "location": "PITTSBURGH, PA", "assets": 568770103},
    {"rank": 10, "name": "TRUIST FINANCIAL CORPORATION", "rssd_id": 1074156, "location": "CHARLOTTE, NC", "assets": 543851000},
    {"rank": 11, "name": "TD GROUP US HOLDINGS LLC", "rssd_id": 3606542, "location": "MOUNT LAUREL, NJ", "assets": 519713109},
    {"rank": 12, "name": "CHARLES SCHWAB CORPORATION, THE", "rssd_id": 1026632, "location": "WESTLAKE, TX", "assets": 465255000},
    {"rank": 13, "name": "BANK OF NEW YORK MELLON CORPORATION, THE", "rssd_id": 3587146, "location": "NEW YORK, NY", "assets": 455321000},
    {"rank": 14, "name": "STATE STREET CORPORATION", "rssd_id": 1111435, "location": "BOSTON, MA", "assets": 371070000},
    {"rank": 15, "name": "AMERICAN EXPRESS COMPANY", "rssd_id": 1275216, "location": "NEW YORK, NY", "assets": 297550000},
    {"rank": 16, "name": "BMO FINANCIAL CORP.", "rssd_id": 1245415, "location": "CHICAGO, IL", "assets": 286206001},
    {"rank": 17, "name": "HSBC NORTH AMERICA HOLDINGS INC.", "rssd_id": 3232316, "location": "NEW YORK, NY", "assets": 236412715},
    {"rank": 18, "name": "FIRST CITIZENS BANCSHARES, INC.", "rssd_id": 1075612, "location": "RALEIGH, NC", "assets": 233488000},
    {"rank": 19, "name": "UNITED SERVICES AUTOMOBILE ASSOCIATION", "rssd_id": 1447376, "location": "SAN ANTONIO, TX", "assets": 231410000},
    {"rank": 20, "name": "CITIZENS FINANCIAL GROUP, INC.", "rssd_id": 1132449, "location": "PROVIDENCE, RI", "assets": 223156482},
]
JPM_DEFAULT_ID = 1039502


def build_lineage_chain(target_id: int, trans: pd.DataFrame) -> list[dict]:
    chain = []
    current = int(target_id)
    visited = set()
    while True:
        if current in visited:
            break
        visited.add(current)
        subset = trans[trans["ID_RSSD_SUCCESSOR"] == current]
        if subset.empty:
            break
        subset = subset.sort_values(["DT_TRANS", "ID_RSSD_PREDECESSOR"])
        row = subset.iloc[-1]
        pred = int(row["ID_RSSD_PREDECESSOR"])
        dt_trans = int(row["DT_TRANS"])
        trn = int(row.get("TRNSFM_CD", 0))
        if pred == current:
            break
        chain.append({"pred": pred, "succ": current, "dt_trans": dt_trans, "trnsfm_cd": trn})
        current = pred
    return chain


def resolve_lineage_root(target_id: int, asof: int, trans: pd.DataFrame) -> tuple[int, list[dict]]:
    if trans.empty:
        return target_id, []
    chain = build_lineage_chain(target_id, trans)
    if not chain:
        return target_id, []

    # Walk backward using the closest future transformation relative to asof.
    # This avoids jumping to an arbitrary predecessor when multiple merger rows exist.
    root = target_id
    progressed = True
    while progressed:
        progressed = False
        sub = trans[(trans["ID_RSSD_SUCCESSOR"] == root) & (trans["DT_TRANS"] > asof)]
        if not sub.empty:
            sub = sub.sort_values(["DT_TRANS", "ID_RSSD_PREDECESSOR"])
            pred = int(sub.iloc[0]["ID_RSSD_PREDECESSOR"])
            if pred != root:
                root = pred
                progressed = True
    return root, chain


def inject_collapse_js(
    html: str,
    children_map: dict[int, list[int]],
    label_base: dict[int, str],
    root_id: int,
    show_labels: bool,
) -> str:
    children_js = {str(k): [int(x) for x in v] for k, v in children_map.items()}
    label_js = {str(k): v for k, v in label_base.items()}
    script = f"""
    <script>
    (function() {{
      var childrenMap = {json.dumps(children_js)};
      var labelBase = {json.dumps(label_js)};
      var rootId = {int(root_id)};
      var showLabels = {str(bool(show_labels)).lower()};
      var initialized = false;

      function boot() {{
        var nw = window.network || (typeof network !== "undefined" ? network : null);
        if (!nw || !nw.body || !nw.body.data || !nw.body.data.nodes || !nw.body.data.edges) {{
          return false;
        }}
        if (initialized) return true;
        initialized = true;

        var nodes = nw.body.data.nodes;
        var edges = nw.body.data.edges;
        var originalNodes = nodes.get();
        var originalEdges = edges.get();
        var nodeById = {{}};
        for (var ni = 0; ni < originalNodes.length; ni++) {{
          nodeById[originalNodes[ni].id] = originalNodes[ni];
        }}
        var expanded = {{}};
        var visibleIds = new Set([rootId]);

        function getChildren(id) {{
          var key = String(id);
          return childrenMap[key] || [];
        }}

        function hasChildren(id) {{
          return getChildren(id).length > 0;
        }}

        function computeLabel(id) {{
          var base = labelBase[String(id)] || "";
          if (!showLabels && id !== rootId) {{
            base = "";
          }}
          if (!base) {{
            return id === rootId ? ("RSSD " + id) : "";
          }}
          return base;
        }}

        function upsertNode(id) {{
          var baseNode = nodeById[id];
          if (!baseNode) return;
          var nodeCopy = Object.assign({{}}, baseNode);
          nodeCopy.label = computeLabel(id);
          return nodeCopy;
        }}

        function computeVisibleSet() {{
          var vis = new Set([rootId]);
          var q = [rootId];
          while (q.length > 0) {{
            var parent = q.shift();
            if (!expanded[parent]) continue;
            var kids = getChildren(parent);
            for (var i = 0; i < kids.length; i++) {{
              var c = kids[i];
              if (!vis.has(c)) {{
                vis.add(c);
                q.push(c);
              }}
            }}
          }}
          return vis;
        }}

        function rebuildGraph(preserveView) {{
          visibleIds = computeVisibleSet();

          var visibleNodes = [];
          visibleIds.forEach(function(id) {{
            var row = upsertNode(id);
            if (row) visibleNodes.push(row);
          }});

          var visibleEdges = [];
          for (var i = 0; i < originalEdges.length; i++) {{
            var e = originalEdges[i];
            if (visibleIds.has(e.from) && visibleIds.has(e.to)) {{
              visibleEdges.push(e);
            }}
          }}

          nodes.clear();
          edges.clear();
          if (visibleNodes.length) nodes.add(visibleNodes);
          if (visibleEdges.length) edges.add(visibleEdges);
          nw.redraw();
          if (!preserveView) {{
            nw.fit({{ nodes: [rootId], animation: false }});
          }}
        }}

        function drawPlusMinus(ctx) {{
          var ids = nodes.getIds();
          for (var i = 0; i < ids.length; i++) {{
            var id = ids[i];
            if (!hasChildren(id)) continue;
            var bnode = nw.body.nodes[id];
            if (!bnode) continue;

            var bg = (bnode.options && bnode.options.color && bnode.options.color.background) || "#1f77b4";
            var glyphColor = "#ffffff";
            var hex = String(bg).trim().replace("#", "");
            if (/^[0-9a-fA-F]{{6}}$/.test(hex)) {{
              var r = parseInt(hex.substring(0, 2), 16);
              var g = parseInt(hex.substring(2, 4), 16);
              var b = parseInt(hex.substring(4, 6), 16);
              var luma = 0.299 * r + 0.587 * g + 0.114 * b;
              glyphColor = luma > 150 ? "#222222" : "#ffffff";
            }}

            var symbol = expanded[id] ? "-" : "+";
            var nodeSize = (bnode.options && bnode.options.size) ? bnode.options.size : 16;
            var glyphPx = Math.max(10, Math.min(16, Math.round(nodeSize * 0.5)));

            ctx.save();
            ctx.font = "bold " + glyphPx + "px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.lineWidth = 2;
            ctx.strokeStyle = glyphColor === "#ffffff" ? "#111111" : "#ffffff";
            ctx.fillStyle = glyphColor;
            ctx.strokeText(symbol, bnode.x, bnode.y + 0.5);
            ctx.fillText(symbol, bnode.x, bnode.y + 0.5);
            ctx.restore();
          }}
        }}

        nw.on("afterDrawing", function(ctx) {{
          drawPlusMinus(ctx);
        }});

        nw.on("click", function(params) {{
          if (!params.nodes || params.nodes.length !== 1) return;
          var id = params.nodes[0];
          if (!hasChildren(id)) return;
          expanded[id] = !expanded[id];
          rebuildGraph(true);
        }});

        rebuildGraph(false);
        return true;
      }}

      var tries = 0;
      var timer = setInterval(function() {{
        tries += 1;
        if (boot() || tries > 200) {{
          clearInterval(timer);
        }}
      }}, 50);
    }})();
    </script>
    """
    return html.replace("</body>", script + "</body>")


@st.cache_data(show_spinner=False, max_entries=1)
def build_org_catalog(attrs: pd.DataFrame) -> pd.DataFrame:
    df = attrs.copy()
    if "EST_TYPE_CD" in df.columns:
        df = df[df["EST_TYPE_CD"] == 1]
    df = df.sort_values(["ID_RSSD", "DT_START"])
    df = df.drop_duplicates("ID_RSSD", keep="last")

    name_short = df.get("NM_SHORT", pd.Series("", index=df.index)).replace("", np.nan)
    name_legal = df.get("NM_LGL", pd.Series("", index=df.index)).replace("", np.nan)
    name = name_short.fillna(name_legal).fillna("Unknown").astype(str).str.strip()

    df["_name"] = name
    df["_label"] = name + " (RSSD " + df["ID_RSSD"].astype(str) + ")"
    return df.set_index("ID_RSSD", drop=False)


@st.cache_data(show_spinner=False, max_entries=1)
def build_entity_name_lookup(attrs: pd.DataFrame) -> dict[int, str]:
    cols = [c for c in ["ID_RSSD", "NM_SHORT", "NM_LGL", "DT_START", "DT_END"] if c in attrs.columns]
    df = attrs[cols].copy()
    if "ID_RSSD" not in df.columns:
        return {}
    if "DT_START" not in df.columns:
        df["DT_START"] = 0
    if "DT_END" not in df.columns:
        df["DT_END"] = 0

    nm_short = df.get("NM_SHORT", pd.Series("", index=df.index)).astype(str).str.strip()
    nm_lgl = df.get("NM_LGL", pd.Series("", index=df.index)).astype(str).str.strip()
    name = nm_short.where(nm_short != "", nm_lgl)
    name = name.fillna("").astype(str).str.strip()
    df["_name"] = name
    df["_has_name"] = (df["_name"] != "").astype(int)

    df = df.sort_values(["ID_RSSD", "_has_name", "DT_END", "DT_START"], ascending=[True, False, False, False])
    df = df.drop_duplicates("ID_RSSD", keep="first")

    out = {}
    for _, r in df.iterrows():
        rid = int(r["ID_RSSD"])
        nm = str(r.get("_name", "")).strip()
        if not nm or nm.lower() in {"unknown", "nan", "none"}:
            nm = f"RSSD {rid}"
        out[rid] = nm
    return out


def build_new_subsidiaries_table(
    new_nodes_full: set[int],
    attrs_asof: pd.DataFrame,
    attrs_asof_loose: pd.DataFrame,
    attrs_best_effort: pd.DataFrame,
    name_lookup: dict[int, str],
) -> pd.DataFrame:
    rows = []
    ids = sorted(int(x) for x in new_nodes_full)
    for nid in ids:
        src = "asof_filtered"
        row = None
        if nid in attrs_asof.index:
            row = attrs_asof.loc[nid]
        elif nid in attrs_asof_loose.index:
            row = attrs_asof_loose.loc[nid]
            src = "asof_unfiltered"
        elif nid in attrs_best_effort.index:
            row = attrs_best_effort.loc[nid]
            src = "nearest_history"
        else:
            src = "relationships_only"

        name_short = ""
        name_legal = ""
        broad = 0
        bhc_ind = 0
        slhc_ind = 0
        ihc_ind = 0
        city = ""
        state = ""
        if row is not None:
            name_short = str(row.get("NM_SHORT", "") or "").strip()
            name_legal = str(row.get("NM_LGL", "") or "").strip()
            broad = int(row.get("BROAD_REG_CD", 0) or 0)
            bhc_ind = int(row.get("BHC_IND", 0) or 0)
            slhc_ind = int(row.get("SLHC_IND", 0) or 0)
            ihc_ind = int(row.get("IHC_IND", 0) or 0)
            city = str(row.get("CITY", "") or "").strip()
            state = str(row.get("STATE_ABBR_NM", "") or "").strip()

        name = name_short or name_legal or name_lookup.get(nid, "")
        if not name or name.lower() in {"unknown", "nan", "none"}:
            name = f"RSSD {nid}"

        node_type = (
            "relationship_only"
            if src == "relationships_only"
            else classify_node_category(broad, bhc_ind, slhc_ind, ihc_ind, is_branch=False)
        )

        rows.append(
            {
                "rssd": nid,
                "name": name,
                "name_short": name_short,
                "name_legal": name_legal,
                "type": node_type,
                "city": city,
                "state": state,
                "attr_source": src,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["rssd", "name", "name_short", "name_legal", "type", "city", "state", "attr_source"])
    out = pd.DataFrame(rows)
    out = out.sort_values(["name", "rssd"])
    return out


@st.cache_data(show_spinner=False, max_entries=2, ttl=1200)
def select_attributes_asof(attrs: pd.DataFrame, asof: int, filter_existence: bool = True) -> pd.DataFrame:
    needed_cols = [
        "ID_RSSD",
        "NM_SHORT",
        "NM_LGL",
        "BROAD_REG_CD",
        "BHC_IND",
        "SLHC_IND",
        "IHC_IND",
        "EST_TYPE_CD",
        "DOMESTIC_IND",
        "DT_START",
        "DT_END",
        "DT_EXIST_CMNC",
        "DT_EXIST_TERM",
        "BANK_CNT",
        "CITY",
        "STATE_ABBR_NM",
        "ENTITY_TYPE",
        "CHTR_TYPE_CD",
        "ORG_TYPE_CD",
        "ID_RSSD_HD_OFF",
    ]
    needed_cols = [c for c in needed_cols if c in attrs.columns]
    df = attrs[needed_cols].copy()
    valid = (df["DT_START"] <= asof) & (df["DT_END"] >= asof)
    if filter_existence:
        exist_ok = (df["DT_EXIST_CMNC"] == 0) | (df["DT_EXIST_CMNC"] <= asof)
        term_ok = (
            (df["DT_EXIST_TERM"] == 0)
            | (df["DT_EXIST_TERM"] == 99991231)
            | (df["DT_EXIST_TERM"] >= asof)
        )
        valid &= exist_ok & term_ok
    df = df[valid]
    df = df.sort_values(["ID_RSSD", "DT_START"])
    df = df.drop_duplicates("ID_RSSD", keep="last")
    return df.set_index("ID_RSSD", drop=False)


@st.cache_data(show_spinner=False, max_entries=6, ttl=1200)
def select_attributes_best_effort_for_ids(
    attrs: pd.DataFrame,
    asof: int,
    rssd_ids: tuple[int, ...],
) -> pd.DataFrame:
    if not rssd_ids:
        return pd.DataFrame()

    ids = [int(x) for x in rssd_ids]
    needed_cols = [
        "ID_RSSD",
        "NM_SHORT",
        "NM_LGL",
        "BROAD_REG_CD",
        "BHC_IND",
        "SLHC_IND",
        "IHC_IND",
        "DT_START",
        "DT_END",
        "CITY",
        "STATE_ABBR_NM",
        "ENTITY_TYPE",
    ]
    needed_cols = [c for c in needed_cols if c in attrs.columns]
    df = attrs[attrs["ID_RSSD"].isin(ids)][needed_cols].copy()
    if df.empty:
        return df.set_index(pd.Index([], name="ID_RSSD"))

    active_now = (df["DT_START"] <= asof) & (df["DT_END"] >= asof)
    started = df["DT_START"] <= asof
    # Rank rows by relevance to as-of:
    # 0 active on asof, 1 most recent historical row, 2 earliest future row.
    df["_rank"] = np.where(active_now, 0, np.where(started, 1, 2)).astype(np.int8)
    df["_pref_start"] = np.where(df["_rank"] == 2, df["DT_START"], -df["DT_START"]).astype(np.int64)
    broad = df["BROAD_REG_CD"] if "BROAD_REG_CD" in df.columns else pd.Series(0, index=df.index)
    bhc = df["BHC_IND"] if "BHC_IND" in df.columns else pd.Series(0, index=df.index)
    slhc = df["SLHC_IND"] if "SLHC_IND" in df.columns else pd.Series(0, index=df.index)
    ihc = df["IHC_IND"] if "IHC_IND" in df.columns else pd.Series(0, index=df.index)
    df["_has_type"] = ((broad > 0) | bhc.isin([1, 2]) | (slhc == 1) | (ihc == 1)).astype(np.int8)

    sort_cols = ["ID_RSSD", "_rank", "_pref_start", "_has_type", "DT_END"]
    sort_asc = [True, True, True, False, False]
    df = df.sort_values(sort_cols, ascending=sort_asc)
    df = df.drop_duplicates("ID_RSSD", keep="first")
    return df.set_index("ID_RSSD", drop=False)


@st.cache_data(show_spinner=False, max_entries=3, ttl=1200)
def select_relationships_asof(
    rels: pd.DataFrame,
    asof: int,
    ctrl_inds=(1,),
    reg_inds=(1, 2),
    reln_lvls=(1,),
) -> pd.DataFrame:
    df = rels.copy()
    year = asof // 10000
    if "YEAR" in df.columns:
        valid = df["YEAR"] == year
    else:
        valid = (df["DT_START"] <= asof) & (df["DT_END"] >= asof)
    if "CTRL_IND" in df.columns and ctrl_inds:
        valid &= df["CTRL_IND"].isin(ctrl_inds)
    if "REG_IND" in df.columns and reg_inds:
        valid &= df["REG_IND"].isin(reg_inds)
    if "RELN_LVL" in df.columns and reln_lvls:
        valid &= df["RELN_LVL"].isin(reln_lvls)
    df = df[valid]
    sort_cols = ["ID_RSSD_PARENT", "ID_RSSD_OFFSPRING", "RELN_LVL", "DT_START"]
    if "YEAR" in df.columns:
        sort_cols = ["YEAR"] + sort_cols
    df = df.sort_values(sort_cols)
    dedup_cols = ["ID_RSSD_PARENT", "ID_RSSD_OFFSPRING", "RELN_LVL"]
    if "YEAR" in df.columns:
        dedup_cols = ["YEAR"] + dedup_cols
    df = df.drop_duplicates(dedup_cols, keep="last")
    return df


@st.cache_data(show_spinner=False, max_entries=1)
def get_year_bounds(attrs: pd.DataFrame, rels: pd.DataFrame) -> tuple[int, int]:
    def _valid_min_max(series: pd.Series) -> tuple[int, int]:
        s = series[(series > 0) & (series < 99990000)]
        if s.empty:
            return 19000000, 19000000
        return int(s.min()), int(s.max())

    a_min, a_max = _valid_min_max(attrs["DT_START"])
    r_min, r_max = _valid_min_max(rels["DT_START"])

    e_min, e_max = _valid_min_max(attrs["DT_END"])
    r_e_min, r_e_max = _valid_min_max(rels["DT_END"])

    min_dt = min(a_min, r_min)
    max_dt = max(a_max, r_max, e_max, r_e_max)
    min_year = max(1900, min_dt // 10000)

    current_year = _dt.date.today().year
    max_year = max_dt // 10000 if max_dt > 19000000 else current_year
    max_year = min(max_year, current_year)
    if max_year < min_year:
        max_year = min_year
    return min_year, max_year


def classify_bank(broad_reg_cd: int) -> str:
    if broad_reg_cd == 1:
        return "bank"
    if broad_reg_cd == 2:
        return "other_depository"
    if broad_reg_cd == 3:
        return "nonbank"
    if broad_reg_cd == 4:
        return "inactive"
    return "unknown"


def classify_node_category(
    broad_reg_cd: int,
    bhc_ind: int,
    slhc_ind: int,
    ihc_ind: int,
    is_branch: bool = False,
) -> str:
    if is_branch:
        return "branch"
    base = classify_bank(int(broad_reg_cd or 0))
    # Bank color takes priority even if flagged as holding/intermediate.
    if base == "bank":
        return "bank"
    if base in {"nonbank", "unknown"}:
        if int(ihc_ind or 0) == 1:
            return "intermediate_holding_company"
        if int(bhc_ind or 0) in {1, 2} or int(slhc_ind or 0) == 1:
            return "holding_company"
    return base


def bank_color(label: str) -> str:
    return {
        "bank": "#1f77b4",
        "other_depository": "#17becf",
        "nonbank": "#ff7f0e",
        "holding_company": "#2ca02c",
        "intermediate_holding_company": "#8c564b",
        "relationship_only": "#bdbdbd",
        "inactive": "#7f7f7f",
        "unknown": "#c7c7c7",
        "branch": "#9467bd",
    }.get(label, "#c7c7c7")


def get_descendants(
    rels: pd.DataFrame,
    root_id: int,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> tuple[set[int], dict[int, int], dict[int, list[int]], bool]:
    children = defaultdict(list)
    for p, c in zip(rels["ID_RSSD_PARENT"], rels["ID_RSSD_OFFSPRING"]):
        children[p].append(c)

    visited = set([root_id])
    depths = {root_id: 0}
    q = deque([root_id])
    truncated = False

    while q:
        node = q.popleft()
        if max_depth is not None and depths[node] >= max_depth:
            continue
        for child in children.get(node, []):
            if child not in visited:
                visited.add(child)
                depths[child] = depths[node] + 1
                q.append(child)
                if max_nodes is not None and len(visited) >= max_nodes:
                    truncated = True
                    q.clear()
                    break
    return visited, depths, children, truncated


def compute_attr_source_counts(
    node_ids: set[int],
    attrs_asof: pd.DataFrame,
    attrs_asof_loose: pd.DataFrame,
    attrs_best_effort: pd.DataFrame,
) -> dict[str, int]:
    counts = {
        "asof_filtered": 0,
        "asof_unfiltered": 0,
        "nearest_history": 0,
        "relationships_only": 0,
    }
    for nid in node_ids:
        if nid in attrs_asof.index:
            counts["asof_filtered"] += 1
        elif nid in attrs_asof_loose.index:
            counts["asof_unfiltered"] += 1
        elif nid in attrs_best_effort.index:
            counts["nearest_history"] += 1
        else:
            counts["relationships_only"] += 1
    return counts


def compute_subtree_sizes(
    children: dict[int, list[int]],
    root_id: int,
    allowed: set[int] | None = None,
) -> dict[int, int]:
    memo = {}

    def dfs(n: int, stack: set[int]) -> int:
        if n in memo:
            return memo[n]
        if n in stack:
            return 0
        stack.add(n)
        size = 1
        for c in children.get(n, []):
            if allowed is not None and c not in allowed:
                continue
            size += dfs(c, stack)
        stack.remove(n)
        memo[n] = size
        return size

    if allowed is None or root_id in allowed:
        dfs(root_id, set())
    return memo


def edge_label(row: pd.Series) -> str:
    fmt = str(row.get("PCT_EQUITY_FORMAT", "")).strip().lower()
    if fmt.startswith("exact") and pd.notna(row.get("PCT_EQUITY")):
        return f"{row.get('PCT_EQUITY'):.2f}% (Exact)"
    if fmt.startswith("bracket"):
        bracket = str(row.get("PCT_EQUITY_BRACKET", "")).strip()
        return f"{bracket} (Bracket)" if bracket else "Bracket"
    if fmt.startswith("other"):
        return "Other basis"
    return ""


def top_level_diagnostic(
    rels_yearly: pd.DataFrame,
    rssd_id: int,
    years: list[int],
    ctrl_inds=(1,),
    reg_inds=(1, 2),
    reln_lvls=(1,),
) -> pd.DataFrame:
    rows = []
    for y in years:
        d = rels_yearly[rels_yearly["YEAR"] == y]
        if ctrl_inds:
            d = d[d["CTRL_IND"].isin(ctrl_inds)]
        if reg_inds:
            d = d[d["REG_IND"].isin(reg_inds)]
        if reln_lvls:
            d = d[d["RELN_LVL"].isin(reln_lvls)]
        parents = set(d["ID_RSSD_PARENT"].unique())
        kids = set(d["ID_RSSD_OFFSPRING"].unique())
        roots = parents - kids
        rows.append(
            {
                "year": y,
                "as_parent_edges": int((d["ID_RSSD_PARENT"] == rssd_id).sum()),
                "as_offspring_edges": int((d["ID_RSSD_OFFSPRING"] == rssd_id).sum()),
                "is_top_level_root": rssd_id in roots,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=1)
def build_root_universe(
    rels_yearly: pd.DataFrame,
    ctrl_inds=(1,),
    reg_inds=(1, 2),
    direct_reln_lvls=(1,),
    all_reln_lvls=(1, 2, 3, 4),
) -> tuple[list[int], dict]:
    df = rels_yearly
    common = pd.Series(True, index=df.index)
    if "CTRL_IND" in df.columns and ctrl_inds:
        common &= df["CTRL_IND"].isin(ctrl_inds)
    if "REG_IND" in df.columns and reg_inds:
        common &= df["REG_IND"].isin(reg_inds)

    direct_mask = common.copy()
    if "RELN_LVL" in df.columns and direct_reln_lvls:
        direct_mask &= df["RELN_LVL"].isin(direct_reln_lvls)

    all_mask = common.copy()
    if "RELN_LVL" in df.columns and all_reln_lvls:
        all_mask &= df["RELN_LVL"].isin(all_reln_lvls)

    direct_df = df[direct_mask]
    all_df = df[all_mask]
    parents = set(direct_df["ID_RSSD_PARENT"].astype(int).tolist())
    offspring_any = set(all_df["ID_RSSD_OFFSPRING"].astype(int).tolist())
    root_ids = sorted(list(parents - offspring_any))

    stats = {
        "direct_edges_all_years": int(len(direct_df)),
        "all_level_edges_all_years": int(len(all_df)),
        "candidate_root_count": int(len(root_ids)),
    }
    return root_ids, stats


def lineage_seed_ids(root_id: int, trans: pd.DataFrame) -> set[int]:
    seeds = {int(root_id)}
    if trans.empty:
        return seeds
    chain = build_lineage_chain(int(root_id), trans)
    for ev in chain:
        seeds.add(int(ev["pred"]))
        seeds.add(int(ev["succ"]))
    return seeds


@st.cache_data(show_spinner=False, max_entries=1)
def build_bhc_year_panel(
    rels_yearly: pd.DataFrame,
    root_id: int,
    seed_ids: tuple[int, ...],
    min_year: int,
    max_year: int,
    ctrl_inds=(1,),
    reg_inds=(1, 2),
    reln_lvls=(1,),
    all_reln_lvls=(1, 2, 3, 4),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = rels_yearly
    if "YEAR" in df.columns:
        df = df[(df["YEAR"] >= min_year) & (df["YEAR"] <= max_year)]

    common = pd.Series(True, index=df.index)
    if "CTRL_IND" in df.columns and ctrl_inds:
        common &= df["CTRL_IND"].isin(ctrl_inds)
    if "REG_IND" in df.columns and reg_inds:
        common &= df["REG_IND"].isin(reg_inds)

    all_mask = common.copy()
    if "RELN_LVL" in df.columns and all_reln_lvls:
        all_mask &= df["RELN_LVL"].isin(all_reln_lvls)
    all_levels_df = df[all_mask]

    seeds = {int(x) for x in seed_ids} if seed_ids else {int(root_id)}
    children = defaultdict(set)
    for p, c in zip(
        all_levels_df["ID_RSSD_PARENT"].astype(int),
        all_levels_df["ID_RSSD_OFFSPRING"].astype(int),
    ):
        children[p].add(c)

    reachable = set(seeds)
    q = deque(seeds)
    while q:
        node = q.popleft()
        for child in children.get(node, ()):
            if child not in reachable:
                reachable.add(child)
                q.append(child)

    in_panel = (
        all_levels_df["ID_RSSD_PARENT"].isin(reachable)
        & all_levels_df["ID_RSSD_OFFSPRING"].isin(reachable)
    )
    panel_all_levels = all_levels_df[in_panel].copy()

    panel_direct = panel_all_levels
    if "RELN_LVL" in panel_direct.columns and reln_lvls:
        panel_direct = panel_direct[panel_direct["RELN_LVL"].isin(reln_lvls)].copy()

    stats = {
        "seed_ids": int(len(seeds)),
        "reachable_nodes": int(len(reachable)),
        "panel_all_levels_rows": int(len(panel_all_levels)),
        "panel_direct_rows": int(len(panel_direct)),
        "years_covered": int(panel_direct["YEAR"].nunique()) if "YEAR" in panel_direct.columns else 0,
    }
    return panel_direct, panel_all_levels, stats


def truncate_label(text: str, max_len: int) -> str:
    s = str(text or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 3)].rstrip() + "..."


def format_node_label(text: str, max_len: int, line_len: int) -> str:
    t = truncate_label(text, max_len)
    if len(t) <= line_len:
        return t
    split_at = t.rfind(" ", 0, line_len + 1)
    if split_at < max(4, line_len // 2):
        split_at = line_len
    left = t[:split_at].rstrip()
    right = t[split_at:].lstrip()
    if not right:
        return left
    return left + "\n" + right


def build_pyvis_graph(
    root_id: int,
    rels_asof: pd.DataFrame,
    attrs_asof: pd.DataFrame,
    new_nodes: set[int],
    include_branches: bool,
    branches_asof: pd.DataFrame,
    max_depth: int | None,
    max_nodes: int | None,
    show_labels: bool,
    show_edge_labels: bool,
    name_map: dict[int, str] | None = None,
    focus_nodes: set[int] | None = None,
    include_path: bool = True,
    base_reachable: set[int] | None = None,
    base_children: dict[int, list[int]] | None = None,
    font_size: int = 13,
    edge_font_size: int = 11,
    arrow_scale: float = 0.6,
    graph_height: int = 720,
) -> tuple[Network, dict]:
    # Collect nodes reachable from root
    if base_reachable is None or base_children is None:
        reachable, depths, children, truncated = get_descendants(
            rels_asof, root_id, max_depth=max_depth, max_nodes=max_nodes
        )
    else:
        reachable = set(base_reachable)
        children = base_children
        truncated = False

    # Optional focus filtering
    if focus_nodes:
        focus_nodes = set(focus_nodes) & reachable
        if focus_nodes:
            focus_set = set()

            def add_descendants(start_id: int):
                q = deque([start_id])
                while q:
                    node = q.popleft()
                    if node in focus_set:
                        continue
                    focus_set.add(node)
                    for child in children.get(node, []):
                        if child in reachable:
                            q.append(child)

            for f in focus_nodes:
                add_descendants(f)

            if include_path:
                parents = defaultdict(list)
                for p, c in zip(rels_asof["ID_RSSD_PARENT"], rels_asof["ID_RSSD_OFFSPRING"]):
                    if p in reachable and c in reachable:
                        parents[c].append(p)

                q = deque(list(focus_nodes))
                while q:
                    node = q.popleft()
                    for par in parents.get(node, []):
                        if par not in focus_set:
                            focus_set.add(par)
                            q.append(par)

            reachable = focus_set

    # Attach branches if requested
    branch_edges = []
    if include_branches and not branches_asof.empty:
        for _, r in branches_asof.iterrows():
            head = int(r.get("ID_RSSD_HD_OFF", 0) or 0)
            bid = int(r.get("ID_RSSD", 0) or 0)
            if head and bid and head in reachable:
                if max_nodes is not None and len(reachable) >= max_nodes:
                    break
                branch_edges.append((head, bid))
                reachable.add(bid)

    # Subtree sizes for node sizing
    sizes = compute_subtree_sizes(children, root_id, allowed=reachable)

    net = Network(
        height=f"{graph_height}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1b1b1b",
        cdn_resources="remote",
    )
    net.set_options(
        json.dumps(
            {
                "layout": {
                    "hierarchical": {
                        "enabled": True,
                        "direction": "UD",
                        "sortMethod": "directed",
                        "nodeSpacing": 200,
                        "levelSeparation": 150,
                    }
                },
                "physics": {"enabled": False},
                "nodes": {"font": {"size": font_size}},
                "edges": {
                    "font": {"size": edge_font_size, "align": "middle"},
                    "arrows": {"to": {"enabled": True, "scaleFactor": arrow_scale}},
                    "smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4},
                },
                "interaction": {
                    "hover": True,
                    "navigationButtons": True,
                    "zoomView": True,
                    "dragNodes": True,
                },
            }
        )
    )

    label_base = {}
    # Add nodes
    new_nodes = set(new_nodes) & set(reachable)
    for node_id in sorted(reachable):
        has_attr_row = False
        if node_id in attrs_asof.index:
            has_attr_row = True
            row = attrs_asof.loc[node_id]
            raw_name = row.get("NM_SHORT") or row.get("NM_LGL") or ""
            name = str(raw_name).strip()
            broad = int(row.get("BROAD_REG_CD", 0) or 0)
            bhc_ind = int(row.get("BHC_IND", 0) or 0)
            slhc_ind = int(row.get("SLHC_IND", 0) or 0)
            ihc_ind = int(row.get("IHC_IND", 0) or 0)
            city = row.get("CITY", "")
            state = row.get("STATE_ABBR_NM", "")
            entity_type = row.get("ENTITY_TYPE", "")
            if (not name or name.lower() in {"unknown", "nan", "none"}) and name_map and node_id in name_map:
                name = str(name_map.get(node_id, "")).strip()
        elif name_map and node_id in name_map:
            name = str(name_map.get(node_id, "")).strip()
            broad = 0
            bhc_ind = 0
            slhc_ind = 0
            ihc_ind = 0
            city = ""
            state = ""
            entity_type = ""
        else:
            name = ""
            broad = 0
            bhc_ind = 0
            slhc_ind = 0
            ihc_ind = 0
            city = ""
            state = ""
            entity_type = ""

        if not name or str(name).strip().lower() in {"unknown", "nan", "none"}:
            name = f"RSSD {node_id}"

        if has_attr_row:
            node_category = classify_node_category(
                broad,
                bhc_ind,
                slhc_ind,
                ihc_ind,
                is_branch=(include_branches and node_id in branches_asof.index),
            )
        elif include_branches and node_id in branches_asof.index:
            node_category = "branch"
        else:
            node_category = "relationship_only"

        color = bank_color(node_category)
        border_color = "#d62728" if node_id in new_nodes else "#333333"
        border_width = 3 if node_id in new_nodes else 1
        size = sizes.get(node_id, 1)
        size = min(60, 8 + math.log1p(size) * 8)

        type_label = node_category.replace("_", " ").title()
        if node_category == "relationship_only":
            type_label = "Relationship-only (no attributes row)"

        title = (
            f"<b>{name}</b><br>"
            f"RSSD: {node_id}<br>"
            f"Type: {type_label}<br>"
        )
        if entity_type:
            title += f"Entity: {entity_type}<br>"
        if city or state:
            title += f"Location: {city} {state}<br>"

        if node_id == root_id:
            label = format_node_label(name, 34, 16)
        elif show_labels:
            label = format_node_label(name, 22, 11)
        else:
            label = format_node_label(name, 16, 10)
        if not label:
            label = f"RSSD {node_id}"
        label_base[node_id] = label
        net.add_node(
            node_id,
            label=label,
            title=title,
            color={"background": color, "border": border_color},
            borderWidth=border_width,
            size=size,
            shape="dot",
        )

    # Add relationship edges
    for _, row in rels_asof.iterrows():
        p = int(row["ID_RSSD_PARENT"])
        c = int(row["ID_RSSD_OFFSPRING"])
        if p in reachable and c in reachable:
            lbl = edge_label(row) if show_edge_labels else ""
            net.add_edge(p, c, label=lbl)

    # Add branch edges
    for p, c in branch_edges:
        net.add_edge(p, c, label="branch")

    meta = {
        "children_map": children,
        "label_base": label_base,
        "root_id": root_id,
        "show_labels": show_labels,
        "reachable": reachable,
    }
    return net, meta


# ----------------------------
# App UI
# ----------------------------

st.title("NPW Org-Chart Explorer")

# Data checks
missing = [p for p in [REL_ZIP, ATTR_ACTIVE_ZIP, ATTR_CLOSED_ZIP] if not p.exists()]
if missing:
    st.error("Missing required data files in Rels_Data: " + ", ".join([str(p.name) for p in missing]))
    st.stop()

with st.spinner("Loading data..."):
    attrs = load_attributes(ATTR_ACTIVE_ZIP, ATTR_CLOSED_ZIP)
    rels = load_relationships(REL_ZIP)

min_year, max_year = get_year_bounds(attrs, rels)
min_year = max(min_year, 1990)
if max_year < min_year:
    max_year = min_year

with st.spinner("Preparing yearly relationship panel..."):
    rels_yearly, rels_quality = load_or_build_relationships_yearly(rels, min_year, max_year, REL_ZIP)
del rels
gc.collect()

# Top timeline control
year_options = list(range(min_year, max_year + 1))
if "selected_year" not in st.session_state:
    st.session_state["selected_year"] = max_year
if int(st.session_state["selected_year"]) < min_year or int(st.session_state["selected_year"]) > max_year:
    st.session_state["selected_year"] = max_year

st.subheader("Timeline")
yr_prev_col, yr_slider_col, yr_next_col, yr_badge_col = st.columns([0.8, 7.6, 0.8, 1.4], gap="small")
with yr_prev_col:
    if st.button("Prev", key="year_prev_btn", width="stretch"):
        st.session_state["selected_year"] = max(min_year, int(st.session_state["selected_year"]) - 1)

with yr_next_col:
    if st.button("Next", key="year_next_btn", width="stretch"):
        st.session_state["selected_year"] = min(max_year, int(st.session_state["selected_year"]) + 1)

with yr_slider_col:
    year_slider = st.select_slider(
        "As-of year",
        options=year_options,
        value=int(st.session_state["selected_year"]),
    )
    st.session_state["selected_year"] = int(year_slider)

with yr_badge_col:
    st.markdown(
        f"""
<div style="display:flex;align-items:center;justify-content:center;height:38px;width:38px;border-radius:999px;border:2px solid #2a4f84;color:#1d3557;font-weight:700;margin-top:26px;">
  {int(st.session_state["selected_year"])}
</div>
""",
        unsafe_allow_html=True,
    )

year = int(st.session_state["selected_year"])
asof = year * 10000 + 1231
prev_asof = (year - 1) * 10000 + 1231

# Sidebar controls
st.sidebar.header("Controls")

top_placeholder = st.sidebar.empty()
org_placeholder = st.sidebar.empty()

org_catalog = build_org_catalog(attrs)
entity_name_lookup = build_entity_name_lookup(attrs)
org_mode = st.sidebar.radio("Organization list", ["Top-level only", "Top-level holdcos only"], index=0)
domestic_only = st.sidebar.checkbox("Domestic only", value=False)

st.sidebar.subheader("Lineage")
lineage_mode = st.sidebar.checkbox("Lineage mode (auto predecessor)", value=True)

st.sidebar.subheader("Graph")

st.components.v1.html(
    """
    <script>
    (function() {
      const params = new URLSearchParams(window.location.search);
      const w = window.innerWidth || 1200;
      const h = window.innerHeight || 800;
      if (!params.get('w') || !params.get('h')) {
        params.set('w', w);
        params.set('h', h);
        window.location.search = params.toString();
      }
    })();
    </script>
    """,
    height=0,
)

def _get_query_param_int(key: str) -> int | None:
    try:
        params = st.query_params
        val = params.get(key)
        if isinstance(val, list):
            val = val[0] if val else None
    except Exception:
        params = st.experimental_get_query_params()
        val = params.get(key, [None])[0]
    try:
        return int(val) if val is not None and str(val).strip() else None
    except Exception:
        return None

screen_w = _get_query_param_int("w")
screen_h = _get_query_param_int("h")

auto_tune = st.sidebar.checkbox("Auto tune for screen size", value=True)

def _auto_graph_settings(w: int | None, h: int | None):
    if not w:
        return 3, 800, True, False, 13, 11, 0.6, 720
    if w < 900:
        return 2, 300, True, False, 11, 10, 0.45, int((h or 720) * 0.7)
    if w < 1200:
        return 3, 600, True, False, 12, 10, 0.5, int((h or 760) * 0.75)
    if w < 1600:
        return 3, 900, True, False, 13, 11, 0.55, int((h or 820) * 0.78)
    return 4, 1200, True, False, 14, 12, 0.65, int((h or 880) * 0.8)

if auto_tune:
    max_depth, max_nodes, show_labels, show_edge_labels, font_size, edge_font_size, arrow_scale, graph_height = _auto_graph_settings(
        screen_w, screen_h
    )
    max_depth = st.sidebar.slider("Max depth", min_value=1, max_value=6, value=max_depth, disabled=True)
    max_nodes = st.sidebar.slider("Max nodes", min_value=50, max_value=2000, value=max_nodes, step=50, disabled=True)
    show_labels = st.sidebar.checkbox("Show labels", value=show_labels, disabled=True)
    show_edge_labels = st.sidebar.checkbox("Show edge labels", value=show_edge_labels, disabled=True)
else:
    max_depth = st.sidebar.slider("Max depth", min_value=1, max_value=6, value=3)
    max_nodes = st.sidebar.slider("Max nodes", min_value=50, max_value=2000, value=800, step=50)
    show_labels = st.sidebar.checkbox("Show labels", value=True)
    show_edge_labels = st.sidebar.checkbox("Show edge labels", value=False)
    font_size = 13
    edge_font_size = 11
    arrow_scale = 0.6
    graph_height = 720

include_branches = st.sidebar.checkbox("Include branches", value=False)
filter_existence = st.sidebar.checkbox("Filter by existence dates", value=True)

with st.sidebar.expander("Relationship data checks", expanded=False):
    st.write(
        {
            "raw_rows": rels_quality.get("raw_rows", 0),
            "clean_rows": rels_quality.get("clean_rows", 0),
            "expanded_rows": rels_quality.get("expanded_rows", 0),
        }
    )
    st.caption(
        "Dropped rows: "
        f"bad_ids={rels_quality.get('dropped_bad_ids', 0)}, "
        f"bad_dates={rels_quality.get('dropped_bad_dates', 0)}, "
        f"reversed_dates={rels_quality.get('dropped_reversed_dates', 0)}, "
        f"self_loops={rels_quality.get('dropped_self_loops', 0)}, "
        f"out_of_range={rels_quality.get('dropped_out_of_range', 0)}"
    )

# As-of data
attrs_asof = select_attributes_asof(attrs, asof, filter_existence=filter_existence)
attrs_asof_loose = select_attributes_asof(attrs, asof, filter_existence=False)
trans = load_transformations(TRANSFORM_ZIP)

with st.spinner("Computing top-level organizations across all years..."):
    root_ids_all_years, root_universe_stats = build_root_universe(
        rels_yearly,
        ctrl_inds=(1,),
        reg_inds=(1, 2),
        direct_reln_lvls=(1,),
        all_reln_lvls=(1, 2, 3, 4),
    )

top20_ids_set = {int(x["rssd_id"]) for x in TOP20_BHCS}

root_ids_index = pd.Index(np.asarray(root_ids_all_years, dtype=np.int32), name="ID_RSSD")
roots_catalog = org_catalog.reindex(root_ids_index)
roots_asof = attrs_asof.reindex(root_ids_index)
roots = roots_asof.combine_first(roots_catalog)
if "ID_RSSD" in roots.columns:
    roots["ID_RSSD"] = (
        roots["ID_RSSD"]
        .fillna(pd.Series(root_ids_index.astype(np.int32), index=root_ids_index))
        .astype(np.int32)
    )
else:
    roots["ID_RSSD"] = root_ids_index.astype(np.int32)

if "EST_TYPE_CD" in roots.columns:
    roots = roots[roots["EST_TYPE_CD"] == 1]

holdco_mask = (
    roots.get("BHC_IND", pd.Series(False, index=roots.index)).isin([1, 2])
    | (roots.get("SLHC_IND", pd.Series(False, index=roots.index)) == 1)
    | (roots.get("IHC_IND", pd.Series(False, index=roots.index)) == 1)
)
if org_mode == "Top-level holdcos only":
    roots = roots[holdco_mask]
else:
    roots = roots[holdco_mask | roots["ID_RSSD"].isin(top20_ids_set)]

if domestic_only and "DOMESTIC_IND" in roots.columns:
    roots = roots[roots["DOMESTIC_IND"].str.upper() == "Y"]

if roots.empty:
    st.warning("No top-level organizations found with the current filters.")
    st.stop()

root_nm = roots.get("NM_SHORT", "").replace("", np.nan).fillna(roots.get("NM_LGL", ""))
root_nm = root_nm.fillna("").astype(str).str.strip()
root_nm = root_nm.mask(root_nm == "", "RSSD " + roots["ID_RSSD"].astype(str))
roots["_label"] = root_nm + " (RSSD " + roots["ID_RSSD"].astype(str) + ")"
roots = roots.sort_values("_label")

root_ids = roots["ID_RSSD"].tolist()
label_map = dict(zip(roots["ID_RSSD"].tolist(), roots["_label"].tolist()))
name_map = dict(zip(org_catalog["ID_RSSD"].tolist(), org_catalog["_name"].tolist()))
name_map.update(entity_name_lookup)

# Keep organization selection stable across year changes.
default_dropdown_id = JPM_DEFAULT_ID if JPM_DEFAULT_ID in root_ids else int(root_ids[0])
if "selected_dropdown_root" not in st.session_state or int(st.session_state["selected_dropdown_root"]) not in root_ids:
    st.session_state["selected_dropdown_root"] = int(default_dropdown_id)

selected_dropdown = org_placeholder.selectbox(
    "Top-level organization",
    options=root_ids,
    format_func=lambda x: label_map.get(x, str(x)),
    index=root_ids.index(int(st.session_state["selected_dropdown_root"])),
    key="selected_dropdown_root",
)

# Top 20 holdings radio (hardcoded)
top20_df = pd.DataFrame(TOP20_BHCS)
top20_ids = top20_df["rssd_id"].astype(int).tolist()
top20_labels = dict(zip(top20_df["rssd_id"].astype(int), top20_df["name"].astype(str)))

def _top_label(x: int) -> str:
    nm = top20_labels.get(x, "Unknown")
    return f"{nm}"

top_options = ["Use dropdown"] + top20_ids
default_top_choice = JPM_DEFAULT_ID if JPM_DEFAULT_ID in top20_ids else "Use dropdown"
if "top20_choice" not in st.session_state or st.session_state["top20_choice"] not in top_options:
    st.session_state["top20_choice"] = default_top_choice

top_choice = top_placeholder.radio(
    "Top 20 BHCs",
    options=top_options,
    format_func=lambda x: x if isinstance(x, str) else _top_label(x),
    index=top_options.index(st.session_state["top20_choice"]),
    key="top20_choice",
)

root_id = selected_dropdown
if isinstance(top_choice, int):
    root_id = top_choice

panel_seed_ids = tuple(sorted(lineage_seed_ids(int(root_id), trans)))

with st.spinner("Processing selected organization across all years..."):
    rels_root_yearly, _rels_root_yearly_all_levels, root_panel_stats = build_bhc_year_panel(
        rels_yearly,
        int(root_id),
        panel_seed_ids,
        min_year,
        max_year,
        ctrl_inds=(1,),
        reg_inds=(1, 2),
        reln_lvls=(1,),
        all_reln_lvls=(1, 2, 3, 4),
    )

rels_asof = select_relationships_asof(rels_root_yearly, asof)

graph_root_id = root_id
lineage_chain = []
if lineage_mode and not trans.empty:
    root_parent_count = int((rels_asof["ID_RSSD_PARENT"] == root_id).sum())
    root_child_count = int((rels_asof["ID_RSSD_OFFSPRING"] == root_id).sum())
    if (root_parent_count + root_child_count) == 0:
        candidate_root, lineage_chain = resolve_lineage_root(root_id, asof, trans)
        if candidate_root != root_id:
            cand_parent_count = int((rels_asof["ID_RSSD_PARENT"] == candidate_root).sum())
            cand_child_count = int((rels_asof["ID_RSSD_OFFSPRING"] == candidate_root).sum())
            if (cand_parent_count + cand_child_count) > 0:
                graph_root_id = candidate_root
                st.sidebar.info(
                    f"Lineage mode fallback: using predecessor {graph_root_id} for year {year}."
                )

with st.sidebar.expander("Selected organization panel", expanded=False):
    st.write(
        {
            "candidate_roots_all_years": root_universe_stats.get("candidate_root_count", 0),
            "seed_ids": root_panel_stats.get("seed_ids", 0),
            "reachable_nodes": root_panel_stats.get("reachable_nodes", 0),
            "rows_all_levels": root_panel_stats.get("panel_all_levels_rows", 0),
            "rows_direct": root_panel_stats.get("panel_direct_rows", 0),
            "years_covered": root_panel_stats.get("years_covered", 0),
        }
    )

branches_asof = pd.DataFrame()
if include_branches:
    branches = load_branches(ATTR_BRANCHES_ZIP)
    if not branches.empty:
        branches_asof = select_attributes_asof(branches, asof, filter_existence=False)

if graph_root_id not in attrs_asof_loose.index:
    st.warning(
        "Selected organization has no matching row in Attributes Active/Closed for this year. "
        "The chart will render from relationships only."
    )
elif graph_root_id not in attrs_asof.index and filter_existence:
    st.info(
        "Selected organization exists in attributes for this year but is filtered out by existence-date rules."
    )

# Base reachable set (used for focus options)
base_reachable, base_depths, base_children, base_truncated = get_descendants(
    rels_asof, graph_root_id, max_depth=max_depth, max_nodes=max_nodes
)
reachable_ids_tuple = tuple(sorted(int(x) for x in base_reachable))
attrs_best_effort = select_attributes_best_effort_for_ids(attrs, asof, reachable_ids_tuple)
attrs_graph = attrs_asof.combine_first(attrs_asof_loose)
if not attrs_best_effort.empty:
    attrs_graph = attrs_graph.combine_first(attrs_best_effort)
attr_source_counts_base = compute_attr_source_counts(base_reachable, attrs_asof, attrs_asof_loose, attrs_best_effort)
with st.sidebar.expander("Type mapping coverage", expanded=False):
    st.write(
        {
            "asof_filtered": int(attr_source_counts_base.get("asof_filtered", 0)),
            "asof_unfiltered": int(attr_source_counts_base.get("asof_unfiltered", 0)),
            "nearest_history": int(attr_source_counts_base.get("nearest_history", 0)),
            "relationships_only": int(attr_source_counts_base.get("relationships_only", 0)),
        }
    )

# Focus mode
st.sidebar.subheader("Select subsidiaries")
focus_mode = st.sidebar.checkbox("Select subsidiaries", value=True)
focus_nodes = set()
include_path = True

if focus_mode:
    focus_options = sorted(base_reachable)

    def _focus_label(x: int) -> str:
        if x in attrs_graph.index:
            row = attrs_graph.loc[x]
            nm = row.get("NM_SHORT") or row.get("NM_LGL") or ""
            nm = str(nm).strip() or name_map.get(x, "")
        else:
            nm = name_map.get(x, "")
        nm = str(nm or "").strip()
        if not nm or nm.lower() in {"unknown", "nan", "none"}:
            nm = f"RSSD {x}"
        return f"{nm} (RSSD {x})"

    selected = st.sidebar.multiselect(
        "Choose subsidiaries (subtrees)",
        options=focus_options,
        format_func=_focus_label,
    )
    focus_nodes = set(selected)
    include_path = st.sidebar.checkbox("Include path to root", value=True)

# Descendant comparison for new nodes (full panel, no depth/node truncation)
new_nodes_full = set()
if year > min_year:
    rels_prev = select_relationships_asof(rels_root_yearly, prev_asof)
    now_nodes_full = get_descendants(
        rels_asof, graph_root_id, max_depth=None, max_nodes=None
    )[0]
    prev_nodes_full = get_descendants(
        rels_prev, graph_root_id, max_depth=None, max_nodes=None
    )[0]
    new_nodes_full = now_nodes_full - prev_nodes_full
    if graph_root_id in new_nodes_full:
        new_nodes_full.remove(graph_root_id)

new_df = build_new_subsidiaries_table(
    new_nodes_full,
    attrs_asof,
    attrs_asof_loose,
    attrs_best_effort,
    name_map,
)
new_nodes = set(new_nodes_full)

# Build graph
with st.spinner("Building graph..."):
    net, meta = build_pyvis_graph(
        graph_root_id,
        rels_asof,
        attrs_graph,
        new_nodes,
        include_branches,
        branches_asof,
        max_depth=max_depth,
        max_nodes=max_nodes,
        show_labels=show_labels,
        show_edge_labels=show_edge_labels,
        name_map=name_map,
        focus_nodes=focus_nodes if focus_mode else None,
        include_path=include_path,
        base_reachable=base_reachable,
        base_children=base_children,
        font_size=font_size,
        edge_font_size=edge_font_size,
        arrow_scale=arrow_scale,
        graph_height=graph_height,
    )
    html = net.generate_html()
    html = inject_collapse_js(
        html,
        meta["children_map"],
        meta["label_base"],
        meta["root_id"],
        meta["show_labels"],
    )

# Legend
legend = """
<div style='display:flex; gap:14px; flex-wrap:wrap; align-items:center;'>
  <div><span style='display:inline-block;width:12px;height:12px;background:#1f77b4;border:1px solid #333;'></span> Bank</div>
  <div><span style='display:inline-block;width:12px;height:12px;background:#17becf;border:1px solid #333;'></span> Other Depository</div>
  <div><span style='display:inline-block;width:12px;height:12px;background:#ff7f0e;border:1px solid #333;'></span> Nonbank</div>
  <div><span style='display:inline-block;width:12px;height:12px;background:#2ca02c;border:1px solid #333;'></span> Holding Company</div>
  <div><span style='display:inline-block;width:12px;height:12px;background:#8c564b;border:1px solid #333;'></span> Intermediate Holding Company</div>
  <div><span style='display:inline-block;width:12px;height:12px;background:#bdbdbd;border:1px solid #333;'></span> Relationship-only (no attributes row)</div>
  <div><span style='display:inline-block;width:12px;height:12px;background:#7f7f7f;border:1px solid #333;'></span> Inactive</div>
  <div><span style='display:inline-block;width:12px;height:12px;background:#c7c7c7;border:1px solid #333;'></span> Unknown</div>
  <div><span style='display:inline-block;width:12px;height:12px;background:#9467bd;border:1px solid #333;'></span> Branch</div>
  <div><span style='display:inline-block;width:12px;height:12px;border:2px solid #d62728;'></span> New in selected year</div>
</div>
"""

main_col, side_col = st.columns([3.3, 1.4], gap="large")

with main_col:
    st.markdown(legend, unsafe_allow_html=True)
    st.info(
        "Click a node with a '+' to expand one level. Click '-' to collapse that node's subtree."
    )
    st.components.v1.html(html, height=graph_height, scrolling=True)
    if max_nodes is not None and len(net.nodes) >= max_nodes:
        st.warning(
            "Graph truncated due to the max nodes limit. "
            "Increase Max nodes or reduce depth to adjust readability."
        )

with side_col:
    st.subheader("New Subsidiaries")
    st.caption(
        "Computed from full descendants (no depth/node cap): entities in selected year but not prior year."
    )
    st.metric("Count vs prior year", len(new_nodes_full))
    if new_df.empty:
        st.info("No newly added subsidiaries for this year.")
    else:
        show_cols = ["rssd", "name", "type", "city", "state", "attr_source"]
        st.dataframe(
            new_df[show_cols],
            width="stretch",
            height=max(320, graph_height - 130),
        )
        missing_name_count = int(new_df["name"].astype(str).str.startswith("RSSD ").sum())
        if missing_name_count > 0:
            st.warning(
                f"{missing_name_count} new subsidiaries have no non-empty name in attributes; showing RSSD IDs."
            )

# Summary stats
st.subheader("Snapshot Summary")
col1, col2, col3 = st.columns(3)
col1.metric("As-of date", f"{year}-12-31")
col2.metric("Nodes in org chart", len(net.nodes))
col3.metric("New subsidiaries vs prior year", len(new_nodes_full))

with st.expander("New subsidiaries QA checks", expanded=False):
    st.caption(
        "New subsidiaries are computed from full descendants (no depth/node cap) using relationship dates, "
        "and enriched from Attributes Active + Attributes Closed."
    )
    if year <= min_year:
        st.write({"year": year, "prior_year": None, "new_subsidiaries": 0})
    else:
        st.write(
            {
                "year": year,
                "prior_year": year - 1,
                "new_subsidiaries": int(len(new_nodes_full)),
                "attr_source_asof_filtered": int((new_df["attr_source"] == "asof_filtered").sum()) if not new_df.empty else 0,
                "attr_source_asof_unfiltered": int((new_df["attr_source"] == "asof_unfiltered").sum()) if not new_df.empty else 0,
                "attr_source_nearest_history": int((new_df["attr_source"] == "nearest_history").sum()) if not new_df.empty else 0,
                "attr_source_relationships_only": int((new_df["attr_source"] == "relationships_only").sum()) if not new_df.empty else 0,
            }
        )

if lineage_mode and lineage_chain:
    with st.expander("Lineage details (transformations)"):
        rows = []
        for ev in lineage_chain:
            rows.append(
                {
                    "predecessor": ev["pred"],
                    "successor": ev["succ"],
                    "dt_trans": ev["dt_trans"],
                    "trnsfm_cd": ev["trnsfm_cd"],
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")

with st.expander("JPMorgan top-level diagnostic"):
    diag_years = sorted(set([1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024, year]))
    jpm_diag = top_level_diagnostic(rels_yearly, 1039502, diag_years)
    st.dataframe(jpm_diag, width="stretch")
