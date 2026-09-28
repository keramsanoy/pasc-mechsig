"""Build the Stage-0 cohort descriptive table ("Table 1").

Produces Overall / PASC+ / PASC- columns plus a standardized mean difference
(SMD) for every characteristic, mirroring the layout expected by the dataset
chapter. This is *additive, read-only* analysis over already-materialised
parquets -- it does not touch the locked feature/model modules or re-run any
pipeline.

Two tables are emitted, one per severity-window feature matrix:
  - table1_w0_90.csv  (primary; w0-90 strict matrix)
  - table1_w0_21.csv  (severity rows sourced from the acute w0-21 matrix)
Everything except the acute-window severity rows should agree between them, so
the pair doubles as a severity cross-check.

Race / ethnicity lives in no parquet; it is pulled live from the OMOP
``CDMPHI.person`` table (reusing the ``pasc.db.connect()`` connection) and cached to
``race_ethnicity_by_person.parquet`` so the tables rebuild offline afterwards.

Outputs (results/descriptives/):
  - table1_w0_90.csv / table1_w0_21.csv  -- characteristic, overall, pasc_pos, pasc_neg, smd
  - lab_missingness_by_window.csv        -- rows = analytes, cols = windows
  - scalars.json                         -- prose numbers (prevalence, IQRs, ...)
  - race_ethnicity_by_person.parquet     -- cached race buckets per person_id

Run:
  python scripts/make_table1.py 2>&1 | tee logs/table1_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: E402,F401  (makes `pasc` importable from a plain clone)
from pasc.config.paths import REPO_ROOT as ROOT, COHORT_DIR, DESCRIPTIVES_DIR as OUTDIR  # noqa: E402

PRIMARY_PARQUET = COHORT_DIR / "enhanced_w0_90_strict_all_patients.parquet"
ACUTE_PARQUET = COHORT_DIR / "enhanced_w0_21_strict_all_patients.parquet"

LABEL_COL = "label"
# Optional guard: the MSHS run modelled 168,345 patients / 2,102 cases (thesis 4.1).
# Set PASC_TABLE1_EXPECT="168345,2102" to refuse a matrix that does not match.
_expect = os.environ.get("PASC_TABLE1_EXPECT", "").strip()
EXPECTED_N, EXPECTED_N_POS = (tuple(int(x) for x in _expect.split(",")) if _expect else (None, None))

# Five mutually-exclusive SARS-CoV-2 variant-wave flags; wave 6 is the remainder.
WAVE_COLS = [
    "f_ext_wave_ancestral_1",
    "f_ext_wave_iota_alpha_2",
    "f_ext_wave_delta_3",
    "f_ext_wave_delta_omicron_4",
    "f_ext_wave_omicron_ba2_ba5_5",
]

# Windows to scan for lab presence/missingness.
LAB_WINDOWS = ["w0_21", "w0_30", "w0_60", "w0_90", "w30_60", "w30_90", "w60_90"]

# Race/ethnicity buckets (collapsed ethnicity-first; order = table row order).
RACE_BUCKETS = [
    "Non-Hispanic white",
    "Non-Hispanic black",
    "Hispanic / Latino",
    "Asian / other / unknown",
]

from pasc.config.omop import CDM_SCHEMA  # OMOP schema name; set OMOP_CDM_SCHEMA to override


# --------------------------------------------------------------------------- #
# Formatting + statistics                                                      #
# --------------------------------------------------------------------------- #
def fmt_continuous(series: pd.Series) -> str:
    """median (q1--q3) with NaNs dropped."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "NA"
    med = s.median()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return f"{med:.0f} ({q1:.0f}-{q3:.0f})"


def fmt_binary(series: pd.Series) -> str:
    """n (pct%) where series is a 0/1 indicator."""
    s = pd.to_numeric(series, errors="coerce")
    n = int(np.nansum(s.values))
    denom = int(s.notna().sum())
    pct = 100.0 * n / denom if denom else float("nan")
    return f"{n:,} ({pct:.1f}%)"


def smd(pos: pd.Series, neg: pd.Series, kind: str) -> float:
    """Standardized difference between PASC+ and PASC- groups.

    continuous: (m1 - m0) / sqrt((s1^2 + s0^2) / 2)
    binary:     (p1 - p0) / sqrt(pbar * (1 - pbar))
    |SMD| > 0.1 is the conventional "meaningful imbalance" flag.
    """
    p = pd.to_numeric(pos, errors="coerce").dropna()
    q = pd.to_numeric(neg, errors="coerce").dropna()
    if p.empty or q.empty:
        return float("nan")

    if kind == "cont":
        m1, m0 = p.mean(), q.mean()
        s1, s0 = p.var(ddof=1), q.var(ddof=1)
        denom = math.sqrt((s1 + s0) / 2.0)
        return float((m1 - m0) / denom) if denom > 0 else float("nan")

    # binary / proportion
    p1, p0 = p.mean(), q.mean()
    pbar = (p1 + p0) / 2.0
    denom = math.sqrt(pbar * (1.0 - pbar))
    return float((p1 - p0) / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Table 1 specification                                                        #
# --------------------------------------------------------------------------- #
def _union(*cols):
    """Row value = 1 if any of the named columns is truthy."""
    def _fn(df: pd.DataFrame) -> pd.Series:
        sub = df[list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0)
        return (sub.sum(axis=1) > 0).astype(int)
    return _fn


def _wave6(df: pd.DataFrame) -> pd.Series:
    """Post-surveillance wave: none of the five variant-wave flags set."""
    sub = df[WAVE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    return (sub.sum(axis=1) == 0).astype(int)


def build_spec() -> list[dict]:
    """Ordered Table-1 row specification.

    Each entry: label, kind ('cont'|'bin'), and either 'col' (a parquet column)
    or 'fn' (a callable df -> series). Race rows are inserted at runtime once
    the DB-derived buckets are joined (see ``insert_race_rows``).
    Rheumatologic disease uses ``f_comor_rheumatic`` (Charlson "rheumatic
    disease"); ``f_comor_autoimmune`` is the broader alternative and is not used.
    """
    return [
        {"label": "Age at index, years", "kind": "cont", "col": "f_age"},
        {"label": "Female", "kind": "bin", "col": "f_sex_female"},
        # -- race / ethnicity rows inserted here at runtime --
        {"label": "Wave 1 (ancestral)", "kind": "bin", "col": "f_ext_wave_ancestral_1"},
        {"label": "Wave 2 (Iota/Alpha)", "kind": "bin", "col": "f_ext_wave_iota_alpha_2"},
        {"label": "Wave 3 (Delta)", "kind": "bin", "col": "f_ext_wave_delta_3"},
        {"label": "Wave 4 (Delta->BA.1)", "kind": "bin", "col": "f_ext_wave_delta_omicron_4"},
        {"label": "Wave 5 (BA.2/BA.5)", "kind": "bin", "col": "f_ext_wave_omicron_ba2_ba5_5"},
        {"label": "Wave 6 (post-surveillance)", "kind": "bin", "fn": _wave6},
        {"label": "Hospitalised within 21 d", "kind": "bin", "col": "f_ext_hosp_acute"},
        {"label": "ICU / mechanical ventilation within 21 d", "kind": "bin", "col": "f_ext_icu_any"},
        {"label": "Obesity / BMI >= 30", "kind": "bin", "col": "f_comor_obesity"},
        {"label": "Diabetes mellitus", "kind": "bin",
         "fn": _union("f_comor_diabetes_uncomplicated", "f_comor_diabetes_complicated")},
        {"label": "Hypertension", "kind": "bin", "col": "f_comor_hypertension"},
        {"label": "Chronic lung disease", "kind": "bin", "col": "f_comor_chronic_pulm"},
        {"label": "Depression / anxiety", "kind": "bin",
         "fn": _union("f_comor_depression", "f_comor_anxiety")},
        {"label": "Rheumatologic disease", "kind": "bin", "col": "f_comor_rheumatic"},
        {"label": "Cancer (active/recent)", "kind": "bin", "col": "f_comor_cancer"},
        {"label": "Pre-index encounters, 1 y", "kind": "cont",
         "col": "f_eng_n_encounters_pre_index_1y"},
        # PCP and insurance rows dropped: f_eng_has_pcp_pre_index is unreliable
        # (provider.specialty_concept_id is ~99.8% NULL and the upstream PCP
        # concept IDs are wrong), and f_eng_insurance_* are all-zero because
        # payer_plan_period was empty in the MSHS extract (thesis 3.4.1), so only the encounter count is populated.
        {"label": "PASC positive (600588/600589, d90-540)", "kind": "bin", "col": LABEL_COL},
    ]


def insert_race_rows(spec: list[dict], race: pd.DataFrame | None) -> list[dict]:
    """Insert the four race/ethnicity rows after 'Female'.

    ``race`` is a per-person frame with a 'race_bucket' column, or None when the
    DB was unreachable (rows then carry a 'tbd' flag and render as 'TBD').
    """
    rows = []
    for b in RACE_BUCKETS:
        if race is None:
            rows.append({"label": f"Race/ethnicity: {b}", "kind": "bin", "tbd": True})
        else:
            ind = (race["race_bucket"] == b).astype(int)
            rows.append({
                "label": f"Race/ethnicity: {b}", "kind": "bin",
                "fn": (lambda s: (lambda df: s.reindex(df.index).fillna(0).astype(int)))(ind),
            })
    out = []
    for entry in spec:
        out.append(entry)
        if entry["label"] == "Female":
            out.extend(rows)
    return out


def _series_for(entry: dict, df: pd.DataFrame) -> pd.Series:
    if "fn" in entry:
        return entry["fn"](df)
    return df[entry["col"]]


def build_table1(df: pd.DataFrame, spec: list[dict], label_col: str = LABEL_COL) -> pd.DataFrame:
    """Pure builder: returns tidy frame characteristic/overall/pasc_pos/pasc_neg/smd."""
    pos_mask = pd.to_numeric(df[label_col], errors="coerce") == 1
    neg_mask = pd.to_numeric(df[label_col], errors="coerce") == 0

    records = []
    for entry in spec:
        label, kind = entry["label"], entry["kind"]
        if entry.get("tbd"):
            records.append({"characteristic": label, "overall": "TBD",
                            "pasc_pos": "TBD", "pasc_neg": "TBD", "smd": np.nan})
            continue

        s = _series_for(entry, df)
        s_pos, s_neg = s[pos_mask], s[neg_mask]
        fmt = fmt_continuous if kind == "cont" else fmt_binary
        records.append({
            "characteristic": label,
            "overall": fmt(s),
            "pasc_pos": fmt(s_pos),
            "pasc_neg": fmt(s_neg),
            "smd": round(smd(s_pos, s_neg, kind), 3),
        })
    return pd.DataFrame.from_records(records,
                                     columns=["characteristic", "overall", "pasc_pos", "pasc_neg", "smd"])


# --------------------------------------------------------------------------- #
# Race / ethnicity extractor (OMOP person table)                              #
# --------------------------------------------------------------------------- #
def _bucket_race(race_name: str | None, eth_name: str | None) -> str:
    """Collapse OMOP race/ethnicity concept names into the four table buckets.

    Ethnicity-first: Hispanic/Latino overrides race.
    """
    r = (race_name or "").lower()
    e = (eth_name or "").lower()
    if "hispanic" in e and "not hispanic" not in e:
        return "Hispanic / Latino"
    if "white" in r:
        return "Non-Hispanic white"
    if "black" in r or "african" in r:
        return "Non-Hispanic black"
    return "Asian / other / unknown"


def fetch_race_ethnicity(person_ids: list[int]) -> pd.DataFrame:
    """Pull race/ethnicity for the cohort from CDMPHI.person and map to buckets.

    Uses a HANA LOCAL TEMPORARY COLUMN TABLE with explicit commits so the JOIN
    sees the inserted ids.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pasc.db import connect
    hana_conn = connect()

    cur = hana_conn.cursor()
    try:
        try:
            cur.execute('DROP TABLE "#t1_race_ids"')
            cur.connection.commit()
        except Exception:
            cur.connection.rollback()

        cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE "#t1_race_ids" (person_id BIGINT)')
        cur.connection.commit()

        rows = [(int(pid),) for pid in person_ids]
        batch = 5000
        for i in range(0, len(rows), batch):
            cur.executemany('INSERT INTO "#t1_race_ids" (person_id) VALUES (?)', rows[i:i + batch])
        cur.connection.commit()

        sql = f"""
            SELECT t.person_id,
                   rc.concept_name AS race_name,
                   ec.concept_name AS ethnicity_name
            FROM "#t1_race_ids" t
            JOIN {CDM_SCHEMA}.person p ON p.person_id = t.person_id
            LEFT JOIN {CDM_SCHEMA}.concept rc ON rc.concept_id = p.race_concept_id
            LEFT JOIN {CDM_SCHEMA}.concept ec ON ec.concept_id = p.ethnicity_concept_id
        """
        cur.execute(sql)
        fetched = cur.fetchall()
        cols = [d[0].lower() for d in cur.description]
    finally:
        try:
            cur.execute('DROP TABLE "#t1_race_ids"')
            cur.connection.commit()
        except Exception:
            cur.connection.rollback()
        cur.close()

    df = pd.DataFrame(fetched, columns=cols)
    df["race_bucket"] = [
        _bucket_race(r, e) for r, e in zip(df.get("race_name"), df.get("ethnicity_name"))
    ]
    return df[["person_id", "race_name", "ethnicity_name", "race_bucket"]]


def load_or_build_race(person_ids: list[int]) -> pd.DataFrame | None:
    """Return cached race buckets if present, else query the DB and cache."""
    cache = OUTDIR / "race_ethnicity_by_person.parquet"
    if cache.exists():
        print(f"  Using cached race/ethnicity: {cache}")
        return pd.read_parquet(cache)
    try:
        race = fetch_race_ethnicity(person_ids)
    except Exception as exc:  # DB unreachable -> leave race rows as TBD
        print(f"  WARNING: race/ethnicity query failed ({exc}); race rows -> TBD")
        return None
    race.to_parquet(cache, index=False)
    print(f"  Wrote {len(race):,} race rows -> {cache}")
    return race


# --------------------------------------------------------------------------- #
# Lab missingness by window                                                    #
# --------------------------------------------------------------------------- #
def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def lab_missingness_by_window() -> pd.DataFrame:
    """Presence-rate -> missingness per analyte per window.

    Routine labs use ``f_lab_{w}_{analyte}_measured``; specialty labs use
    ``f_ind_enh_{w}_immuno_{ana|anti_ccp}_ordered``. ADAMTS13 has no flag in any
    window manifest and is reported as 'not recorded'.
    """
    analytes = {
        "CRP": ("lab", "crp"),
        "D-dimer": ("lab", "ddimer"),
        "Ferritin": ("lab", "ferritin"),
        "ANA": ("immuno", "ana"),
        "anti-CCP": ("immuno", "anti_ccp"),
        "ADAMTS13": ("absent", None),
    }
    records = {name: {} for name in analytes}
    for w in LAB_WINDOWS:
        path = COHORT_DIR / f"enhanced_{w}_strict_all_patients.parquet"
        if not path.exists():
            for name in analytes:
                records[name][w] = "no parquet"
            continue
        cols = _parquet_columns(path)
        for name, (kind, key) in analytes.items():
            if kind == "absent":
                records[name][w] = "not recorded"
                continue
            flag = (f"f_lab_{w}_{key}_measured" if kind == "lab"
                    else f"f_ind_enh_{w}_immuno_{key}_ordered")
            if flag not in cols:
                records[name][w] = "not recorded"
                continue
            s = pd.read_parquet(path, columns=[flag])[flag]
            miss = 100.0 * (1.0 - pd.to_numeric(s, errors="coerce").mean())
            records[name][w] = f"{miss:.1f}%"

    out = pd.DataFrame.from_dict(records, orient="index")
    out = out.reindex(columns=LAB_WINDOWS)
    out.index.name = "analyte"
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Prose scalars                                                                #
# --------------------------------------------------------------------------- #
def build_scalars(df: pd.DataFrame, race: pd.DataFrame | None, lab_miss: pd.DataFrame) -> dict:
    age = pd.to_numeric(df["f_age"], errors="coerce").dropna()
    enc = pd.to_numeric(df["f_eng_n_encounters_pre_index_1y"], errors="coerce").dropna()
    n = len(df)
    n_pos = int(pd.to_numeric(df[LABEL_COL], errors="coerce").sum())

    scalars = {
        "n_total": n,
        "n_pos": n_pos,
        "n_neg": n - n_pos,
        "pasc_prevalence_pct": round(100.0 * n_pos / n, 2),
        "prose_note": ("Dataset prose claims prevalence '~10.4%'; the true "
                       "single-code (600588/600589) prevalence is 1.2%. Flag for rewrite."),
        "age_median": round(float(age.median()), 1),
        "age_q1": round(float(age.quantile(0.25)), 1),
        "age_q3": round(float(age.quantile(0.75)), 1),
        "female_pct": round(100.0 * pd.to_numeric(df["f_sex_female"], errors="coerce").mean(), 1),
        "pre_index_encounters_p25": round(float(enc.quantile(0.25)), 1),
        "pre_index_encounters_p50": round(float(enc.quantile(0.50)), 1),
        "pre_index_encounters_p75": round(float(enc.quantile(0.75)), 1),
        "pre_index_encounters_p95": round(float(enc.quantile(0.95)), 1),
    }

    if race is not None:
        order = race.set_index("person_id").reindex(df["person_id"].values)["race_bucket"]
        dist = order.value_counts(normalize=True) * 100.0
        scalars["race_distribution_pct"] = {b: round(float(dist.get(b, 0.0)), 1) for b in RACE_BUCKETS}
    else:
        scalars["race_distribution_pct"] = {b: "TBD" for b in RACE_BUCKETS}

    def _row(analyte):
        r = lab_miss.set_index("analyte").loc[analyte]
        return {w: r[w] for w in LAB_WINDOWS}

    scalars["lab_missingness_routine"] = {a: _row(a) for a in ["CRP", "D-dimer", "Ferritin"]}
    scalars["lab_missingness_specialty"] = {a: _row(a) for a in ["ANA", "anti-CCP", "ADAMTS13"]}
    return scalars


# --------------------------------------------------------------------------- #
# Verification                                                                 #
# --------------------------------------------------------------------------- #
def verify(df: pd.DataFrame) -> None:
    n = len(df)
    n_pos = int(pd.to_numeric(df[LABEL_COL], errors="coerce").sum())
    n_neg = n - n_pos
    if EXPECTED_N is not None:
        assert n == EXPECTED_N, f"row count {n} != {EXPECTED_N} (PASC_TABLE1_EXPECT)"
        assert n_pos == EXPECTED_N_POS, f"n_pos {n_pos} != {EXPECTED_N_POS} (PASC_TABLE1_EXPECT)"
    assert n_pos + n_neg == n
    prev = 100.0 * n_pos / n
    print(f"  modelled population: n={n:,} cases={n_pos:,} prevalence={prev:.3f}%")

    wave = df[WAVE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    row_sums = wave.sum(axis=1)
    assert (row_sums <= 1).all(), "wave flags are not mutually exclusive"
    counts = [int(wave[c].sum()) for c in WAVE_COLS]
    wave6 = int((row_sums == 0).sum())
    assert sum(counts) + wave6 == n, "six wave counts do not sum to N"
    print(f"  [verify] N={n:,} n_pos={n_pos:,} prevalence={prev:.2f}%")
    print(f"  [verify] waves 1-5={counts} wave6={wave6:,} sum={sum(counts) + wave6:,}")


def assert_columns(df: pd.DataFrame, spec: list[dict]) -> None:
    needed = {LABEL_COL, "person_id", "f_age", *WAVE_COLS}
    for entry in spec:
        if "col" in entry:
            needed.add(entry["col"])
    missing = sorted(c for c in needed if c not in df.columns)
    if missing:
        raise SystemExit(f"FATAL: columns missing from parquet (schema drift?): {missing}")


# --------------------------------------------------------------------------- #
# PNG rendering                                                                #
# --------------------------------------------------------------------------- #
def render_table_png(table: pd.DataFrame, path: Path, title: str) -> None:
    """Render a Table-1 frame as a publication-style PNG.

    The 'smd' column is shown to 3 decimals and |SMD| > 0.1 cells are bolded to
    flag meaningful PASC+/PASC- imbalance.
    """
    headers = ["Characteristic", "Overall", "PASC+", "PASC\u2212", "SMD"]
    cells = []
    for _, r in table.iterrows():
        smd_val = r["smd"]
        smd_txt = "" if pd.isna(smd_val) else f"{smd_val:.3f}"
        cells.append([str(r["characteristic"]), str(r["overall"]),
                      str(r["pasc_pos"]), str(r["pasc_neg"]), smd_txt])

    n_rows = len(cells)
    fig_h = 1.1 + 0.34 * n_rows
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, loc="left")

    tbl = ax.table(cellText=cells, colLabels=headers, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)

    col_widths = [0.34, 0.18, 0.18, 0.18, 0.12]
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_width(col_widths[col])
        if col >= 1:
            cell.set_text_props(ha="right")
        if row == 0:  # header
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold",
                                ha="left" if col == 0 else "right")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#f4f6f8")
            # bold meaningful SMDs
            if col == 4 and cells[row - 1][4]:
                try:
                    if abs(float(cells[row - 1][4])) > 0.1:
                        cell.set_text_props(fontweight="bold", color="#b22222", ha="right")
                except ValueError:
                    pass

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading primary matrix: {PRIMARY_PARQUET}")
    df90 = pd.read_parquet(PRIMARY_PARQUET)

    base_spec = build_spec()
    assert_columns(df90, base_spec)
    verify(df90)

    # Race/ethnicity (DB-dependent; cached after first run).
    race = load_or_build_race(df90["person_id"].tolist())
    if race is not None:
        race_idx = race.set_index("person_id").reindex(df90["person_id"].values).reset_index(drop=True)
        race_for_build = race_idx.set_index(df90.index)
    else:
        race_for_build = None

    spec = insert_race_rows(base_spec, race_for_build)

    # Table 1 -- primary (w0-90).
    t90 = build_table1(df90, spec)
    out90 = OUTDIR / "table1_w0_90.csv"
    t90.to_csv(out90, index=False)
    print(f"  Wrote {out90} ({len(t90)} rows)")
    png90 = OUTDIR / "table1_w0_90.png"
    render_table_png(t90, png90, "Table 1. Cohort characteristics (w0-90 feature window)")
    print(f"  Wrote {png90}")

    # Table 1 -- acute (w0-21), severity rows re-sourced from the acute matrix.
    print(f"Reading acute matrix: {ACUTE_PARQUET}")
    df21 = pd.read_parquet(ACUTE_PARQUET)
    assert_columns(df21, base_spec)
    if race is not None:
        race21 = race.set_index("person_id").reindex(df21["person_id"].values).reset_index(drop=True)
        race21 = race21.set_index(df21.index)
    else:
        race21 = None
    spec21 = insert_race_rows(base_spec, race21)
    t21 = build_table1(df21, spec21)
    out21 = OUTDIR / "table1_w0_21.csv"
    t21.to_csv(out21, index=False)
    print(f"  Wrote {out21} ({len(t21)} rows)")
    png21 = OUTDIR / "table1_w0_21.png"
    render_table_png(t21, png21, "Table 1. Cohort characteristics (w0-21 acute window)")
    print(f"  Wrote {png21}")

    # Severity cross-check (hosp / ICU should be near-identical across windows).
    for row in ["Hospitalised within 21 d", "ICU / mechanical ventilation within 21 d"]:
        v90 = t90.loc[t90["characteristic"] == row, "overall"].iloc[0]
        v21 = t21.loc[t21["characteristic"] == row, "overall"].iloc[0]
        flag = "" if v90 == v21 else "  <-- DIFFERS"
        print(f"  [severity] {row}: w0_90={v90}  w0_21={v21}{flag}")

    # Lab missingness by window.
    lab_miss = lab_missingness_by_window()
    out_lab = OUTDIR / "lab_missingness_by_window.csv"
    lab_miss.to_csv(out_lab, index=False)
    print(f"  Wrote {out_lab}")

    # Prose scalars (race passed as per-person frame keyed by person_id).
    scalars = build_scalars(df90, race, lab_miss)
    out_sc = OUTDIR / "scalars.json"
    out_sc.write_text(json.dumps(scalars, indent=2))
    print(f"  Wrote {out_sc}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
