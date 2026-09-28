#!/usr/bin/env python
"""
Patient-flow (CONSORT-style) diagram of the modelled population (thesis Figure 4.1).

Replicates the full inclusion/exclusion chain:
  - Upstream cohort build (combined_cohort.build_combined_cohorts -> saved parquet)
  - Modeling-time gates applied inside run_enhanced_mechsig.py

All counts are computed directly from the saved cohort parquet; the figure is
drawn with matplotlib (no image generation). Run with:

    python make_consort_enhanced_mechsig.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- Config (mirrors run_enhanced_mechsig.py defaults) ----------------------
from pasc_paths import COHORT_DIR as _COHORT_DIR, RESULTS_DIR as _RESULTS_DIR
COHORT_DIR = str(_COHORT_DIR)
COHORT_MODE = "strict"
MIN_DAYS_POST_ACUTE = 90      # PASC label enforcement / acute-prevalent drop
MIN_FOLLOWUP_DAYS = 365       # confirmed-negative gate (PASC_MIN_FOLLOWUP_DAYS)
OUT_PATH = os.path.join(str(_RESULTS_DIR), "consort_enhanced_mechsig_strict.png")


def compute_counts():
    """Load the saved cohort and reproduce both modeling gates; return counts."""
    path = os.path.join(COHORT_DIR, f"combined_{COHORT_MODE}_all_patients.parquet")
    df = pd.read_parquet(path)
    for c in ["covid_index_date", "first_pacs_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    n_saved = len(df)
    saved_pos = int((df["label"] == 1).sum())
    saved_neg = int((df["label"] == 0).sum())
    retained_dead = int(df["death_bucket"].isin(
        ["post_acute_death", "d_post_pasc_true_positive"]).sum())
    idx_min, idx_max = df["covid_index_date"].min().date(), df["covid_index_date"].max().date()

    # Modeling gate 1: drop acute/prevalent PASC (first PACS dx < 90 d post-index)
    days_to_pasc = (df["first_pacs_date"] - df["covid_index_date"]).dt.days
    acute_mask = df["first_pacs_date"].notna() & (days_to_pasc < MIN_DAYS_POST_ACUTE)
    n_acute = int(acute_mask.sum())
    df2 = df[~acute_mask].copy()

    # Modeling gate 2: confirmed-negative gate (controls need >= MIN_FOLLOWUP_DAYS)
    short_mask = (df2["label"] == 0) & (df2["followup_days"] < MIN_FOLLOWUP_DAYS)
    n_short = int(short_mask.sum())
    df3 = df2[~short_mask].copy()

    pos = int((df3["label"] == 1).sum())
    neg = int((df3["label"] == 0).sum())
    pos_in = int(((df3["label"] == 1) & (df3["is_inpatient"] == 1)).sum())
    pos_out = pos - pos_in
    neg_in = int(((df3["label"] == 0) & (df3["is_inpatient"] == 1)).sum())
    neg_out = neg - neg_in

    return dict(
        n_saved=n_saved, saved_pos=saved_pos, saved_neg=saved_neg,
        retained_dead=retained_dead, idx_min=idx_min, idx_max=idx_max,
        n_acute=n_acute, n_after_gate1=len(df2),
        n_short=n_short, n_final=len(df3),
        pos=pos, neg=neg, pos_in=pos_in, pos_out=pos_out,
        neg_in=neg_in, neg_out=neg_out,
    )


def draw(c):
    """Render the CONSORT figure from the computed counts dict `c`."""
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 20)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#33526e", fs=10, bold=False, align="center"):
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.10,rounding_size=0.10",
                           fc=fc, ec=ec, lw=1.6)
        ax.add_patch(b)
        ha = "center" if align == "center" else "left"
        tx = x if align == "center" else x - w / 2 + 0.2
        ax.text(tx, y, text, ha=ha, va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.7, color="#33526e"))

    cx = 4.6
    # ---- main vertical flow ----
    box(cx, 19.1, 7.6, 1.2,
        "Source population — OMOP CDM (schema CDMPHI), Mount Sinai Health System\n"
        "All persons in condition_occurrence / measurement / visit_occurrence",
        fc="#dce8f7", bold=True, fs=10.5)
    arrow(cx, 18.5, cx, 17.95)

    box(cx, 17.1, 7.6, 1.7,
        "INCLUSION — COVID-19 positive base population\n"
        "(a) U07.1 diagnosis: condition_concept_id = 37311061   OR\n"
        "(b) positive SARS-CoV-2 test: measurement_concept_id \u2208\n"
        "    {706169, 586526, 706170, 706163, 723476}\n"
        "    AND value_as_concept_id \u2208 {45884084, 45877985}\n"
        "covid_index_date = earliest positive event",
        fc="#e3edf9", fs=9.3, align="left")
    arrow(cx, 16.25, cx, 15.55)

    box(cx, 14.85, 7.6, 1.4,
        "Effect-based death exclusion (apply_death_and_followup)\n"
        "EXCLUDE bucket a: death \u2264 index (ineligible)\n"
        "EXCLUDE bucket b: index < death \u2264 acute_end (incomplete features)\n"
        "Retain post-acute deaths (censored if control)",
        fc="#f6eee3", ec="#9e6b33", fs=9.3, align="left")
    arrow(cx, 14.15, cx, 13.5)

    box(cx, 12.85, 7.6, 1.0,
        f"Index-date window filter:  2020-01-01 \u2264 index \u2264 data cutoff\n"
        f"(observed index range in saved cohort: {c['idx_min']} \u2192 {c['idx_max']})",
        fc="#f6eee3", ec="#9e6b33", fs=9.3, align="left")
    arrow(cx, 12.35, cx, 11.7)

    box(cx, 11.0, 7.6, 1.4,
        "LABELING (add_long_covid_labels)\n"
        "label = 1 if PACS dx \u2265 90 d post-index\n"
        "PACS: condition_concept_id \u2208 {600588, 600589}\n"
        "(proxy for ICD-10 U09.9); else label = 0",
        fc="#e7f6ec", ec="#2e7d4f", fs=9.3, align="left")
    arrow(cx, 10.3, cx, 9.65)

    box(cx, 8.95, 7.6, 1.1,
        f"SAVED STRICT COHORT  (combined_strict_all_patients.parquet)\n"
        f"N = {c['n_saved']:,}   (label 1 = {c['saved_pos']:,};  label 0 = {c['saved_neg']:,})\n"
        f"retained post-acute deaths: {c['retained_dead']:,}",
        fc="#dce8f7", bold=True, fs=9.6)
    arrow(cx, 8.4, cx, 7.75)

    box(cx, 7.05, 7.6, 1.0,
        f"MODELING EXCLUSION 1 (run_enhanced_mechsig.py)\n"
        f"Drop acute/prevalent PASC: first PACS dx < 90 d post-index\n"
        f"\u2192 remaining N = {c['n_after_gate1']:,}",
        fs=9.3, align="left")
    arrow(cx, 6.55, cx, 5.9)

    box(cx, 5.2, 7.6, 1.0,
        f"MODELING EXCLUSION 2 — confirmed-negative gate\n"
        f"Drop controls with follow-up < 365 d (PASC_MIN_FOLLOWUP_DAYS)\n"
        f"\u2192 FINAL ANALYTIC N = {c['n_final']:,}",
        bold=True, fc="#dce8f7", fs=9.3, align="left")

    # ---- exclusion side boxes ----
    ex = 11.4
    box(ex, 14.85, 4.6, 1.1, "EXCLUDED (upstream)\ndeath \u2264 index  or\ndeath in acute window\n(count not stored)",
        fc="#fbeeee", ec="#9e4444", fs=8.8)
    arrow(cx + 3.8, 14.85, ex - 2.3, 14.85)
    box(ex, 12.85, 4.6, 0.95, "EXCLUDED (upstream)\nindex < 2020 or > cutoff\n(count not stored)",
        fc="#fbeeee", ec="#9e4444", fs=8.8)
    arrow(cx + 3.8, 12.85, ex - 2.3, 12.85)
    box(ex, 7.05, 4.6, 0.85, f"EXCLUDED\nacute/prevalent PASC < 90 d\nn = {c['n_acute']:,}",
        fc="#fbeeee", ec="#9e4444", fs=8.8)
    arrow(cx + 3.8, 7.05, ex - 2.3, 7.05)
    box(ex, 5.2, 4.6, 0.85, f"EXCLUDED controls\nfollow-up < 365 d\nn = {c['n_short']:,}",
        fc="#fbeeee", ec="#9e4444", fs=8.8)
    arrow(cx + 3.8, 5.2, ex - 2.3, 5.2)

    # ---- final split ----
    arrow(cx, 4.7, cx, 4.1)
    ax.text(cx, 4.32, "Final analytic cohort split by outcome label",
            ha="center", fontsize=9.5, style="italic", color="#33526e")
    lx, rx = 2.3, 7.0
    arrow(cx, 3.75, lx, 3.1)
    arrow(cx, 3.75, rx, 3.1)
    box(lx, 2.15, 4.0, 1.85,
        f"LABEL = 1  (PASC+ / cases)\n"
        f"n = {c['pos']:,}   ({c['pos'] / c['n_final'] * 100:.2f}%)\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"inpatient:  {c['pos_in']:,}\n"
        f"outpatient: {c['pos_out']:,}",
        fc="#e7f6ec", ec="#2e7d4f", bold=True, fs=9.5)
    box(rx, 2.15, 4.0, 1.85,
        f"LABEL = 0  (controls)\n"
        f"n = {c['neg']:,}   ({c['neg'] / c['n_final'] * 100:.2f}%)\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"inpatient:  {c['neg_in']:,}\n"
        f"outpatient: {c['neg_out']:,}",
        fc="#eef1f5", ec="#5a6b7d", bold=True, fs=9.5)

    ax.set_title("CONSORT — run_enhanced_mechsig.py cohort (strict mode)\n"
                 "inclusion/exclusion criteria with OMOP concept codes",
                 fontsize=13.5, fontweight="bold", pad=14)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"SAVED {OUT_PATH}")


def main():
    counts = compute_counts()
    print("COUNTS:", counts)
    draw(counts)


if __name__ == "__main__":
    main()
