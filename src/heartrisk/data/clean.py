"""Validation gate and cleaning: raw UCI files -> data/processed/heart_clean.csv.

Per site: check the raw bytes against MANIFEST.json, parse ("?" -> missing), recode zeros that
mean "not measured" (D2), then apply the gate: a row that breaks any rule in configs/data.yaml
goes to rejected.csv with its reason instead of being dropped silently. The run fails if a
site's row count differs from the config or its rejection rate exceeds the threshold.

No imputation happens here. Imputation belongs in the model pipeline, fitted on training folds
only, so that no information leaks from validation data into training.

    python -m heartrisk.data.clean      # writes data/processed/{heart_clean.csv, rejected.csv,
                                        #   validation_report.json, DATA_CARD.md}
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandera.errors import SchemaErrors

from heartrisk.config import load_config
from heartrisk.data.download import sha256_bytes, verify
from heartrisk.data.schema import (
    CLEAN_COLUMNS,
    FEATURES,
    FLOAT_COLUMNS,
    INTEGER_FEATURES,
    RAW_COLUMNS,
    clean_schema,
    rules_schema,
)

log = logging.getLogger(__name__)

CLEAN_NAME = "heart_clean.csv"
REJECTED_NAME = "rejected.csv"
REPORT_NAME = "validation_report.json"
CARD_NAME = "DATA_CARD.md"


class GateError(RuntimeError):
    """The validation gate refused the data; details are in validation_report.json."""


def read_site(path: Path, missing_token: str) -> pd.DataFrame:
    """Parse one headerless UCI file; `missing_token` becomes NaN, all else must be numeric."""
    try:
        return pd.read_csv(
            path,
            header=None,
            names=RAW_COLUMNS,
            na_values=[missing_token],
            keep_default_na=False,
            dtype="float64",
        )
    except ValueError as exc:
        raise GateError(f"{path.name}: not parseable as numeric columns ({exc})") from exc


def recode_zeros(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace zeros in `columns` with NaN; returns the new frame and a count per column."""
    out = df.copy()
    counts = {}
    for column in columns:
        mask = out[column] == 0
        counts[column] = int(mask.sum())
        out.loc[mask, column] = np.nan
    return out, counts


def split_by_rules(df: pd.DataFrame, schema: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(accepted, rejected); each rejected row lists every rule it broke in `reason`."""
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        if cases["index"].isna().any():
            raise GateError(f"schema-level failure: {cases['check'].tolist()}") from exc
        # pandas 3 keeps NaN as missing under astype(str), so name missing values explicitly.
        values = cases["failure_case"].map(lambda v: "missing" if pd.isna(v) else str(v))
        reason = cases["column"].astype(str) + "=" + values + " fails " + cases["check"].astype(str)
        cases = cases.assign(index=cases["index"].astype(int), reason=reason).sort_values(
            ["index", "column", "check"]
        )
        reasons = cases.groupby("index")["reason"].agg("; ".join)
        return df.drop(index=reasons.index), df.loc[reasons.index].assign(reason=reasons)
    return df, df.iloc[0:0].assign(reason=pd.Series(dtype="str"))


def to_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename `num` to `num_raw`, derive `target = num > 0`, cast codes to nullable ints."""
    out = df.rename(columns={"num": "num_raw"})
    out["target"] = out["num_raw"] > 0
    for column in [*INTEGER_FEATURES, "num_raw", "target"]:
        out[column] = out[column].astype("Int64")
    return out[CLEAN_COLUMNS].reset_index(drop=True)


def load_clean(path: Path) -> pd.DataFrame:
    """Read heart_clean.csv with its intended dtypes (Int64 codes, float oldpeak)."""
    dtypes = {c: "float64" if c in FLOAT_COLUMNS else "Int64" for c in CLEAN_COLUMNS[2:]}
    return pd.read_csv(
        path,
        dtype={"record_id": "str", "site": "str", **dtypes},
        keep_default_na=False,
        na_values=[""],
    )


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


def _site_summary(site: str, spec: dict, path: Path, raw, accepted, rejected, zeros) -> dict:
    return {
        "label": spec.get("label", site),
        "input_sha256": sha256_bytes(path.read_bytes()),
        "rows_read": len(raw),
        "rows_expected": spec["expected_rows"],
        "rows_accepted": len(accepted),
        "rows_rejected": len(rejected),
        "rejection_rate": round(len(rejected) / len(raw), 4) if len(raw) else 1.0,
        "zeros_recoded": zeros,
        "exact_duplicates": int(raw.duplicated(subset=RAW_COLUMNS).sum()),
        "disease_pct": _pct(int((accepted["num"] > 0).sum()), len(accepted)),
        "missing_pct": {c: _pct(int(accepted[c].isna().sum()), len(accepted)) for c in FEATURES},
    }


def prepare(config: dict[str, Any], raw_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Run the gate over every site and write the four outputs; raises GateError on failure."""
    problems = verify(config, raw_dir)
    if problems:
        raise GateError("raw data does not match MANIFEST.json: " + "; ".join(problems))
    out_dir.mkdir(parents=True, exist_ok=True)
    rules = rules_schema(config)
    threshold = config["gate"]["max_rejection_rate"]
    accepted_parts, rejected_parts, sites, failures = [], [], {}, []

    for site, spec in config["sites"].items():
        path = raw_dir / f"{site}.data"
        raw = read_site(path, config["missing_token"])
        raw.insert(0, "site", site)
        raw.insert(0, "record_id", [f"{site}-{i:03d}" for i in range(1, len(raw) + 1)])
        recoded, zeros = recode_zeros(raw, config["zero_means_missing"])
        accepted, rejected = split_by_rules(recoded, rules)
        accepted_parts.append(accepted)
        rejected_parts.append(rejected)
        summary = _site_summary(site, spec, path, raw, accepted, rejected, zeros)
        sites[site] = summary
        if summary["rows_read"] != spec["expected_rows"]:
            failures.append(
                f"{site}: read {summary['rows_read']} rows, config expects {spec['expected_rows']}"
            )
        if summary["rejection_rate"] > threshold:
            failures.append(
                f"{site}: rejection rate {summary['rejection_rate']:.1%} exceeds {threshold:.0%}"
            )

    rejected_all = pd.concat(rejected_parts)[["record_id", "site", "reason", *RAW_COLUMNS]]
    rejected_all.to_csv(out_dir / REJECTED_NAME, index=False)
    totals = {
        key: sum(s[key] for s in sites.values())
        for key in ("rows_read", "rows_accepted", "rows_rejected")
    }
    report = {
        "status": "fail" if failures else "pass",
        "gate": {
            "max_rejection_rate": threshold,
            "zero_means_missing": config["zero_means_missing"],
            "rules": "ranges and codes in configs/data.yaml",
        },
        "failures": failures,
        "totals": totals,
        "sites": sites,
        "output": None,
    }
    _log_summary(sites, threshold)

    clean_path, card_path = out_dir / CLEAN_NAME, out_dir / CARD_NAME
    if failures:
        clean_path.unlink(missing_ok=True)  # never leave a stale clean file behind a failed gate
        card_path.unlink(missing_ok=True)
        _write_json(out_dir / REPORT_NAME, report)
        raise GateError("validation gate failed: " + "; ".join(failures))

    clean = to_clean_frame(pd.concat(accepted_parts))
    clean_schema(config).validate(clean)
    clean.to_csv(clean_path, index=False)
    report["output"] = {
        "file": CLEAN_NAME,
        "rows": len(clean),
        "sha256": sha256_bytes(clean_path.read_bytes()),
    }
    _write_json(out_dir / REPORT_NAME, report)
    card_path.write_text(render_data_card(report, config), encoding="utf-8")
    return report


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _log_summary(sites: dict[str, dict], threshold: float) -> None:
    log.info(
        "%-12s %5s %9s %9s %7s %16s %6s %9s",
        "site", "read", "expected", "rejected", "rate", "zeros->NaN(bp/ch)", "dupes", "disease%",
    )  # fmt: skip
    for site, s in sites.items():
        zeros = f"{s['zeros_recoded'].get('trestbps', 0)}/{s['zeros_recoded'].get('chol', 0)}"
        passed = s["rejection_rate"] <= threshold and s["rows_read"] == s["rows_expected"]
        flag = "" if passed else "  <-- FAIL"
        log.info(
            "%-12s %5d %9d %9d %6.1f%% %16s %6d %9.1f%s",
            site, s["rows_read"], s["rows_expected"], s["rows_rejected"],
            100 * s["rejection_rate"], zeros, s["exact_duplicates"], s["disease_pct"], flag,
        )  # fmt: skip


def render_data_card(report: dict[str, Any], config: dict[str, Any]) -> str:
    """Markdown data card for heart_clean.csv, generated from the validation report."""
    source, sites = config["source"], report["sites"]
    labels = [s["label"] for s in sites.values()]
    lines = [
        "# Data card: heart_clean.csv",
        "",
        "Generated by `python -m heartrisk.data.clean` (`make prepare`). Do not edit by hand.",
        "",
        "> Educational project: not a medical device and not medical advice.",
        "",
        "## Source",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Dataset | {source['name']} |",
        f"| DOI | https://doi.org/{source['doi']} |",
        f"| Licence | {source['licence']} |",
        f"| Citation | {' '.join(source['citation'].split())} |",
        "",
        "## Processing",
        "",
        "| Step | Rule |",
        "|---|---|",
        "| Integrity | Every raw file must match its SHA-256 in `data/raw/MANIFEST.json` |",
        f"| Parse | `{config['missing_token']}` becomes a missing value |",
        f"| Recode | Zeros in {', '.join(f'`{c}`' for c in config['zero_means_missing'])} "
        'mean "not measured" and become missing |',
        "| Gate | Rows outside the ranges or codes in `configs/data.yaml`, or without a "
        "diagnosis, go to `rejected.csv` with a reason |",
        f"| Fail | A site fails if its row count differs from the config or more than "
        f"{report['gate']['max_rejection_rate']:.0%} of its rows are rejected |",
        "| Target | `target = 1` if `num_raw > 0` (any disease severity), else `0` |",
        "| Not done | No imputation or scaling: both happen inside the model pipeline |",
        "",
        "## Rows per site",
        "",
        "| Site | Rows read | Rejected | Zeros recoded | Exact duplicates | Disease % | "
        "Input SHA-256 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sites.values():
        zeros = " · ".join(f"{c} {n}" for c, n in s["zeros_recoded"].items())
        lines.append(
            f"| {s['label']} | {s['rows_read']} | {s['rows_rejected']} | {zeros} | "
            f"{s['exact_duplicates']} | {s['disease_pct']:.1f} | `{s['input_sha256'][:16]}…` |"
        )
    lines += [
        "",
        "## Missing values (% of accepted rows, after recoding zeros)",
        "",
        "| Feature | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for feature in FEATURES:
        cells = " | ".join(f"{s['missing_pct'][feature]:.1f}" for s in sites.values())
        lines.append(f"| `{feature}` | {cells} |")
    output = report["output"]
    lines += [
        "",
        "## Output",
        "",
        "| File | Rows | SHA-256 |",
        "|---|---|---|",
        f"| `{output['file']}` | {output['rows']} | `{output['sha256']}` |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (1 when the gate fails)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml", help="data config (YAML)")
    parser.add_argument("--raw-dir", help="override paths.raw_dir from the config")
    parser.add_argument("--out-dir", help="override paths.processed_dir from the config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    config = load_config(args.config)
    raw_dir = Path(args.raw_dir or config["paths"]["raw_dir"])
    out_dir = Path(args.out_dir or config["paths"]["processed_dir"])
    try:
        report = prepare(config, raw_dir, out_dir)
    except GateError as exc:
        log.error("GATE FAIL: %s", exc)
        return 1
    totals, output = report["totals"], report["output"]
    log.info(
        "GATE PASS: %d rows read, %d accepted, %d rejected -> %s (sha256 %s…)",
        totals["rows_read"], totals["rows_accepted"], totals["rows_rejected"],
        out_dir / output["file"], output["sha256"][:16],
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
