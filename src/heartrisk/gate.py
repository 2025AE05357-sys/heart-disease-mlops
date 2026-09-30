"""CI quality gate (decision D9): fail the pipeline if the champion is not good enough.

Reads reports/metrics/champion.json, fails when the hold-out ROC-AUC is below the minimum, and
appends a metrics table to the GitHub Actions step summary.

    python -m heartrisk.gate --min-roc-auc 0.80 --summary "$GITHUB_STEP_SUMMARY"
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from heartrisk.columns import SITE_LABELS


def check(champion: dict[str, Any], min_roc_auc: float) -> tuple[bool, str]:
    """(passed, message) for the hold-out ROC-AUC against `min_roc_auc` (inclusive)."""
    auc = champion["holdout"]["roc_auc"]
    ok = auc >= min_roc_auc
    verdict = "PASS" if ok else "FAIL"
    return ok, f"quality gate {verdict}: hold-out ROC-AUC {auc:.3f} (minimum {min_roc_auc:.3f})"


def _fmt(value: float | None) -> str:
    return (
        "n/a"
        if value is None or (isinstance(value, float) and math.isnan(value))
        else f"{value:.3f}"
    )


def summary_markdown(champion: dict[str, Any], ok: bool, min_roc_auc: float) -> str:
    """Markdown block for the step summary: verdict, selection and one row per evaluation set."""
    cv = champion["cv"]
    rows = [("Cleveland hold-out", champion["holdout"])]
    rows += [(SITE_LABELS.get(site, site), m) for site, m in champion["external"].items()]
    lines = [
        f"### Quality gate: {'passed' if ok else 'FAILED'}",
        "",
        f"Champion **{champion['family']}**: {champion['selection_reason']}.  ",
        f"CV ROC-AUC {_fmt(cv['cv_roc_auc_mean'])} ± {_fmt(cv['cv_roc_auc_std'])}; "
        f"gate: hold-out ROC-AUC ≥ {min_roc_auc:.2f}.",
        "",
        "| Evaluation set | n | Disease | ROC-AUC | Recall |",
        "|---|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['n']} | {100 * m['prevalence']:.0f}% | {_fmt(m['roc_auc'])} | "
            f"{_fmt(m['recall'])} |"
        )
    return "\n".join(lines) + "\n\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit code 1 when the gate fails."""
    parser = argparse.ArgumentParser(description="Fail when the champion is below the bar (D9)")
    parser.add_argument("--champion", default="reports/metrics/champion.json")
    parser.add_argument("--min-roc-auc", type=float, default=0.80)
    parser.add_argument(
        "--summary", help="append a markdown table here (e.g. $GITHUB_STEP_SUMMARY)"
    )
    args = parser.parse_args(argv)
    champion = json.loads(Path(args.champion).read_text(encoding="utf-8"))
    ok, message = check(champion, args.min_roc_auc)
    print(message)
    if args.summary:
        with Path(args.summary).open("a", encoding="utf-8") as fh:
            fh.write(summary_markdown(champion, ok, args.min_roc_auc))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
