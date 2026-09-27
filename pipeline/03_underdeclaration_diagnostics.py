#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_underdeclaration_diagnostics.py — under-declaration diagnostics
===================================================================
The ITBI 'valor de transação' (transaction price) is self-declared. In São
Paulo the tax base is max(declared price, VVR — the municipal reference
assessed value), so declaring BELOW the VVR yields no tax saving; the
incentive is to declare AT the VVR when the true price is higher, which shows
up as bunching of the declared/VVR ratio at exactly 1. Financed deals are
audited by the lender (bank appraisal, loan contract), so they are less
exposed to under-declaration; a financed amount larger than the declared
price would be a direct red flag (none exist in this base).

Since pipeline v3, cash deals pegged to the VVR (ratio in [0.99, 1.01], not
financed) are excluded upstream, so this report shows the RESIDUAL picture
after that exclusion; the bunching that motivated the rule is documented in
`data/filter_report.md`.

Reads the cleaned bases and writes data/underdeclaration_report.md.
Run from the project root:  python pipeline/03_underdeclaration_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "underdeclaration_report.md"

TYPES = ["apartment", "house", "commercial"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sql": str, "cep": str, "setor": str,
                                  "setor_quadra": str, "padrao_iptu": str})
    df["financiado"] = df["financiado"].astype(bool)
    return df


def pct(x) -> str:
    return f"{100 * x:.1f}%"


def ratio_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-type profile of the declared/VVR ratio."""
    rows = []
    for t in TYPES + ["TOTAL"]:
        g = df if t == "TOTAL" else df[df["tipo_imovel"] == t]
        r = g["razao_vvr"].dropna()
        rows.append({
            "type": t,
            "n": len(g),
            "VVR>0": pct(g["razao_vvr"].notna().mean()),
            "P25": f"{r.quantile(.25):.2f}",
            "median": f"{r.median():.2f}",
            "P75": f"{r.quantile(.75):.2f}",
            "ratio<0.7": pct((r < 0.7).mean()),
            "0.7<=ratio<0.95": pct(((r >= 0.7) & (r < 0.95)).mean()),
            "pegged to 1 (+/-5%)": pct(r.between(0.95, 1.05).mean()),
            "exactly 1 (+/-1%)": pct(r.between(0.99, 1.01).mean()),
            "ratio>1.05": pct((r > 1.05).mean()),
        })
    return pd.DataFrame(rows)


def bunching_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Fine histogram of the ratio around 1 (0.80–1.20, 2% bins)."""
    r = df["razao_vvr"].dropna()
    bins = np.round(np.arange(0.80, 1.22, 0.02), 2)
    cut = pd.cut(r[(r >= 0.80) & (r < 1.20)], bins=bins, right=False)
    out = cut.value_counts().sort_index()
    return pd.DataFrame({"bin": [f"[{i.left:.2f}, {i.right:.2f})" for i in out.index],
                         "n": out.values,
                         "% of base": [f"{100 * v / len(df):.2f}%" for v in out.values]})


def financing_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TYPES + ["TOTAL"]:
        g = df if t == "TOTAL" else df[df["tipo_imovel"] == t]
        fin, cash = g[g["financiado"]], g[~g["financiado"]]
        rf, rc = fin["razao_vvr"].dropna(), cash["razao_vvr"].dropna()
        rows.append({
            "type": t,
            "% financed": pct(g["financiado"].mean()),
            "median ratio (financed)": f"{rf.median():.2f}",
            "median ratio (cash)": f"{rc.median():.2f}",
            "pegged +/-5% (fin.)": pct(rf.between(0.95, 1.05).mean()),
            "pegged +/-5% (cash)": pct(rc.between(0.95, 1.05).mean()),
            "ratio<1 (fin.)": pct((rf < 1).mean()),
            "ratio<1 (cash)": pct((rc < 1).mean()),
            "median R$/m2 (fin.)": f"{np.exp(fin['ln_vu']).median():,.0f}",
            "median R$/m2 (cash)": f"{np.exp(cash['ln_vu']).median():,.0f}",
        })
    return pd.DataFrame(rows)


def ltv_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Financed amount vs declared price (LTV>1 = declared below the loan)."""
    fin = df[df["financiado"] & (df["valor_financiado"] > 0)].copy()
    fin["ltv"] = fin["valor_financiado"] / fin["valor_transacao"]
    rows = []
    for t in TYPES + ["TOTAL"]:
        g = fin if t == "TOTAL" else fin[fin["tipo_imovel"] == t]
        rows.append({
            "type": t,
            "n financed w/ amount": len(g),
            "median LTV": f"{g['ltv'].median():.2f}",
            "LTV>0.9": pct((g["ltv"] > 0.9).mean()),
            "LTV>1 (loan above declared price)":
                f"{(g['ltv'] > 1).sum():,} ({pct((g['ltv'] > 1).mean())})",
            "LTV>1.2": f"{(g['ltv'] > 1.2).sum():,} ({pct((g['ltv'] > 1.2).mean())})",
        })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                     for row in df.itertuples(index=False))
    return "\n".join([head, sep, body])


def main() -> None:
    d25 = load(ROOT / "data" / "itbi_sp_2025_train.csv")
    d26 = load(ROOT / "data" / "itbi_sp_2026_test.csv")

    sections = ["""# Under-declaration diagnostics — ITBI-SP base (v3)

The ITBI transaction price is **self-declared by the taxpayer**. In São Paulo
the tax base is max(declared price, VVR): declaring below the VVR saves no
tax, so the economic incentive is to declare **exactly the VVR** when the
true price is higher — visible as bunching of the declared/VVR ratio at 1.
Financed deals go through the lender's underwriting (bank appraisal, loan
contract) and are less exposed; a loan larger than the declared price
(LTV > 1) would be a direct red flag.

Since pipeline v3, cash deals pegged to the VVR (ratio in [0.99, 1.01],
not financed) are excluded upstream, so this report shows the RESIDUAL
picture after that exclusion — the bunching that motivated the rule is
documented in `data/filter_report.md`. Reminder: the gross-error ratio
filter [0.3, 5] is applied upstream as well.
"""]

    for label, d in [("2025 (training)", d25), ("2026 (test)", d26)]:
        sections.append(f"## {label} — declared/VVR ratio by property type\n\n"
                        + md_table(ratio_profile(d)))

    sections.append("## Ratio bunching around 1 — 2025 base (2% bins)\n\n"
                    + md_table(bunching_histogram(d25)))

    for label, d in [("2025 (training)", d25), ("2026 (test)", d26)]:
        sections.append(f"## {label} — financed vs cash\n\n"
                        + md_table(financing_profile(d)))

    sections.append("## Loan-to-declared-price (LTV) — 2025 base\n\n"
                    + md_table(ltv_flags(d25)))
    sections.append("## Loan-to-declared-price (LTV) — 2026 base\n\n"
                    + md_table(ltv_flags(d26)))

    # combined residual signals (2025)
    r = d25["razao_vvr"]
    pegged = r.between(0.95, 1.05)
    fin = d25["financiado"]
    ltv = np.where(d25["valor_financiado"] > 0,
                   d25["valor_financiado"] / d25["valor_transacao"], np.nan)
    flag_ltv = pd.Series(ltv, index=d25.index) > 1
    cross = pd.DataFrame({
        "signal": ["ratio pegged to 1 (+/-5%)", "ratio < 0.95",
                   "LTV > 1 (financed only)",
                   "pegged AND cash", "pegged AND financed"],
        "n (2025)": [int(pegged.sum()), int((r < 0.95).sum()), int(flag_ltv.sum()),
                     int((pegged & ~fin).sum()), int((pegged & fin).sum())],
        "% of base": [pct(pegged.mean()), pct((r < 0.95).mean()),
                      pct(flag_ltv.mean()),
                      pct((pegged & ~fin).mean()), pct((pegged & fin).mean())],
    })
    sections.append("## Combined residual signals — 2025 base\n\n" + md_table(cross))

    OUT.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
