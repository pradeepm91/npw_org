# NPW Org-Chart Explorer

This app builds an interactive org chart from the NPW/NIC Bulk Data Download files in `Rels_Data`.

## What it does
- Dropdown to select a top-level BHC/SLHC/IHC.
- Year slider (as-of `YYYY-12-31`).
- Hierarchy graph with bank vs nonbank coloring.
- Highlights newly added subsidiaries versus prior year.

## Data expectations
Required files (already in this repo):
- `Rels_Data/CSV_RELATIONSHIPS.zip`
- `Rels_Data/CSV_ATTRIBUTES_ACTIVE.zip`
- `Rels_Data/CSV_ATTRIBUTES_CLOSED.zip`

Optional:
- `Rels_Data/CSV_ATTRIBUTES_BRANCHES.zip` (only used if “Include branches” is checked)

## Install and run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The graph uses `CTRL_IND = 1`, `RELN_LVL = 1` relationships by default (controlled, direct).
- As-of logic uses `DT_START <= asof <= DT_END` and existence filters from Attributes.
- Branches are excluded by default and attached via `ID_RSSD_HD_OFF` when enabled.

## Sharing with coauthors
Easiest options:
- Send the repo + instructions; they run `streamlit run app.py` locally.
- Host on Streamlit Community Cloud (requires a GitHub repo) and share the URL.

If you want a static HTML export or a different hosting setup, tell me.
