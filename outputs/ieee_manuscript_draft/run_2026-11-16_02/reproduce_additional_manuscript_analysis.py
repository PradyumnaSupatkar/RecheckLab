"""Reproduce completion-conditioned token means and manuscript cost figures.

Reads only existing v0.2 prepared/raw response records and the already-verified
policy comparison CSV. It does not fit models, change labels, or call an LLM.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREP = ROOT / "outputs" / "v0.2" / "data" / "prepared"
RAW = ROOT / "data" / "raw"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
sys.path.insert(0, str(ROOT))
from rechecklab.data import question_hash


def load_cohorts() -> dict[str, pd.DataFrame]:
    cohorts = {
        "MATH source test": pd.read_parquet(PREP / "source_test.parquet"),
        "CommonsenseQA target test": pd.read_parquet(PREP / "target_test.parquet"),
    }
    for raw_name, display_name in (("math500", "MATH500"), ("gsm8k", "GSM8K")):
        frame = pd.read_parquet(RAW / f"{raw_name}.parquet")
        frame = frame.loc[frame["action"].eq("active_verify")].copy()
        frame.loc[frame["base_done_reason"].eq(""), "base_done_reason"] = "unknown"
        cohorts[display_name] = frame
    return cohorts


def completion_token_summary(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    ordered = ["MATH source test", "MATH500", "GSM8K", "CommonsenseQA target test"]
    for cohort in ordered:
        frame = cohorts[cohort].copy()
        frame["group_id"] = frame["query"].map(question_hash)
        frame["state"] = frame["base_done_reason"].astype(str).str.strip().str.casefold()
        frame["action_actual_tokens"] = pd.to_numeric(frame["action_actual_tokens"], errors="raise")
        tokens = frame["action_actual_tokens"].to_numpy(float)
        if not np.isfinite(tokens).all() or (tokens < 0).any():
            raise ValueError(f"Invalid action token count in {cohort}.")

        states_per_question = frame.groupby("group_id")["state"].nunique(dropna=False)
        if not states_per_question.eq(1).all():
            raise ValueError(f"Completion state differs across rollouts for {cohort}.")

        # First average repeated action calls within each canonical question;
        # then average these per-question values inside each completion state.
        qlevel = frame.groupby(["group_id", "state"], sort=True)["action_actual_tokens"].mean().reset_index()
        qstate = qlevel.groupby("state", sort=True)["action_actual_tokens"].agg(["mean", "sum", "count"])
        qcounts = frame.drop_duplicates("group_id").groupby("state").size().to_dict()
        callcounts = frame.groupby("state").size().to_dict()
        length_mean = float(qstate.loc["length", "mean"]) if "length" in qstate.index else np.nan
        stop_mean = float(qstate.loc["stop", "mean"]) if "stop" in qstate.index else np.nan
        length_q = int(qcounts.get("length", 0))
        stop_q = int(qcounts.get("stop", 0))
        length_calls = int(callcounts.get("length", 0))
        stop_calls = int(callcounts.get("stop", 0))
        known_calls = length_calls + stop_calls
        total_q_equal_revision_budget = float(qlevel["action_actual_tokens"].sum())
        length_q_equal_revision_budget = float(qstate.loc["length", "sum"]) if "length" in qstate.index else 0.0
        rows.append({
            "cohort": cohort,
            "questions_total": int(frame["group_id"].nunique()),
            "length_questions": length_q,
            "stop_questions": stop_q,
            "unknown_questions": int(qcounts.get("unknown", 0)),
            "length_action_calls": length_calls,
            "stop_action_calls": stop_calls,
            "length_call_rate_pct": 100.0 * length_calls / known_calls if known_calls else np.nan,
            "mean_length_action_tokens_per_question": length_mean,
            "mean_stop_action_tokens_per_question": stop_mean,
            "length_revision_token_share_question_equal_pct": (
                100.0 * length_q_equal_revision_budget / total_q_equal_revision_budget
                if total_q_equal_revision_budget else np.nan
            ),
        })
    return pd.DataFrame(rows)


def policy_cost_figures() -> None:
    policies = pd.read_csv(TABLES / "MAIN_POLICY_COMPARISON.csv")
    style = {
        "Never": ("#1f77b4", "o"),
        "Always": ("#ff7f0e", "s"),
        "Completion": ("#2ca02c", "^"),
        "Frozen ML": ("#d62728", "D"),
        "Oracle (hindsight)": ("#9467bd", "X"),
    }
    evaluated = [
        "MATH source test", "CommonsenseQA target test", "MATH500", "GSM8K"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=True)
    for ax, cohort in zip(axes.flat, evaluated):
        part = policies.loc[policies["cohort"].eq(cohort)]
        for policy, (color, marker) in style.items():
            row = part.loc[part["policy"].eq(policy)]
            if row.empty:
                continue
            ax.scatter(
                row["average_total_tokens"], row["accuracy_pct"],
                color=color, marker=marker, s=62, edgecolors="black", linewidths=0.35,
                label=policy, zorder=3,
            )
        ax.set_title(cohort)
        ax.set_xlabel("Recorded total tokens / question")
        ax.grid(True, alpha=0.28)
        ax.set_ylim(55, 96)
    axes[0, 0].set_ylabel("Final-answer accuracy (%)")
    axes[1, 0].set_ylabel("Final-answer accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Accuracy versus recorded token cost: four evaluated cohorts", y=0.99)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(FIGURES / "accuracy_vs_tokens.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Keep development on its own figure because it informed model/threshold selection.
    dev = policies.loc[policies["cohort"].eq("MATH source development")]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for policy, (color, marker) in style.items():
        row = dev.loc[dev["policy"].eq(policy)]
        if row.empty:
            continue
        ax.scatter(row["average_total_tokens"], row["accuracy_pct"], color=color,
                   marker=marker, s=70, edgecolors="black", linewidths=0.4,
                   label=policy, zorder=3)
    ax.set_title("MATH source development (selection data; descriptive only)")
    ax.set_xlabel("Recorded total tokens / question")
    ax.set_ylabel("Final-answer accuracy (%)")
    ax.set_ylim(78, 94)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "source_development_accuracy_vs_tokens.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = completion_token_summary(load_cohorts())
    summary.to_csv(TABLES / "CALL_VS_TOKEN_REDUCTION.csv", index=False, float_format="%.8f")
    policy_cost_figures()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
