"""Build Perspektywy ranking datasets (2022-2026) from raw PDF-extracted rows.

Raw data: official Perspektywy "Ranking Uczelni Akademickich" PDF tables.
Each indicator score is relative to the best university (100 = best in indicator).

Pipeline per year:
1. Parse rows -> university x indicator matrix.
2. Fit per-column weights via NNLS against the published WSK (ranking score),
   constrained validation of the header-derived weight hypothesis.
3. Aggregate indicators into the 7 criterion groups (weighted mean, then
   renormalized so the best university in the group = 100 -- the same
   convention as the paper's Table 2, verified for 2025).
4. Emit CSVs: indicator-level (all universities) and criterion-level
   (all + technical-only subset).
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "raw"
OUT = HERE

# ----------------------------------------------------------------------------
# Year configuration: number of prev-rank columns, indicator count,
# group boundaries (contiguous column ranges) and within-group weights
# in printed column order. Group order in the PDF (2022-2026):
#   Prestiz | Absolwenci | Potencjal naukowy | Innowacyjnosc |
#   Efektywnosc naukowa | Warunki ksztalcenia | Umiedzynarodowienie
# Weights verified for 2025 against the paper's Table 2 (exact match);
# 2026 uses the same methodology. For 2022-2024 within-group weights are
# header-derived hypotheses validated below via NNLS fit against WSK.
# ----------------------------------------------------------------------------
GROUPS = ["Prestiz", "Absolwenci", "PotencjalNaukowy", "Innowacyjnosc",
          "EfektywnoscNaukowa", "WarunkiKsztalcenia", "Umiedzynarodowienie"]

CONFIG = {
    2026: dict(nprev=2, nind=30,
               weights=[[10, 2], [6, 6], [8, 2, 1, 2], [3, 3, 2],
                        [5, 4, 3, 3, 3, 5, 3, 4], [5, 5],
                        [3, 3, 2, 1, 1, 1, 1.5, 1.5, 1]]),
    2025: dict(nprev=3, nind=30,
               weights=[[10, 2], [6, 6], [8, 2, 1, 2], [3, 3, 2],
                        [5, 4, 3, 3, 3, 5, 3, 4], [5, 5],
                        [3, 3, 2, 1, 1, 1, 1.5, 1.5, 1]]),
    2024: dict(nprev=3, nind=31,
               weights=[[10, 2], [6, 6], [10, 1, 1, 1], [3, 3, 2],
                        [6, 3, 3, 3, 3, 3, 3, 3, 3], [5, 5],
                        [3, 3, 2, 1, 1, 1, 1.5, 1.5, 1]]),
    2023: dict(nprev=3, nind=30,
               weights=[[10, 2], [12], [12, 1, 1, 1], [3, 3, 2],
                        [6, 4, 3, 3, 3, 3, 3, 3], [5, 5],
                        [3, 3, 2, 1, 1, 1, 1, 1, 1, 1]]),
    2022: dict(nprev=3, nind=29,
               weights=[[10, 2], [12], [10, 3, 1, 1], [3, 3, 2],
                        [6, 4, 3, 3, 3, 3, 3, 3], [5, 5],
                        [3, 2, 3, 2, 1, 1, 1, 1, 1]]),
}

# Canonical names for technical universities across editions
TECHNICAL = {
    "Politechnika Warszawska": ["Politechnika Warszawska"],
    "Politechnika Gdanska": ["Politechnika Gdańska"],
    "AGH": ["Akademia Górniczo-Hutnicza"],
    "Politechnika Wroclawska": ["Politechnika Wrocławska"],
    "Politechnika Poznanska": ["Politechnika Poznańska"],
    "Politechnika Slaska": ["Politechnika Śląska"],
    "Politechnika Lodzka": ["Politechnika Łódzka"],
    "PJATK": ["Polsko-Japońska Akademia"],
    "Politechnika Krakowska": ["Politechnika Krakowska"],
    "Politechnika Lubelska": ["Politechnika Lubelska"],
    "ZUT": ["Zachodniopomorski Uniwersytet Technologiczny"],
    "Politechnika Opolska": ["Politechnika Opolska"],
    "WAT": ["Wojskowa Akademia Techniczna"],
    "Politechnika Bialostocka": ["Politechnika Białostocka"],
    "Politechnika Bydgoska": ["Politechnika Bydgoska"],
    "Politechnika Czestochowska": ["Politechnika Częstochowska"],
    "Politechnika Rzeszowska": ["Politechnika Rzeszowska"],
    "Uniwersytet Morski w Gdyni": ["Uniwersytet Morski w Gdyni"],
    "Politechnika Swietokrzyska": ["Politechnika Świętokrzyska"],
    "Politechnika Koszalinska": ["Politechnika Koszalińska"],
    "Politechnika Morska w Szczecinie": ["Politechnika Morska w Szczecinie",
                                          "Akademia Morska w Szczecinie"],
    "Uniwersytet Bielsko-Bialski": ["Uniwersytet Bielsko-Bialski",
                                     "Akademia Techniczno-Humanistyczna"],
    "Uniwersytet Radomski": ["Uniwersytet Radomski",
                              "Uniwersytet Technologiczno-Human",
                              "Uniwersytet Technologiczno-Humanistyczny"],
}

NUM = re.compile(r"^\d{1,3},\d{1,2}$")


def is_val(tok):
    return bool(NUM.match(tok)) or tok == "*"


def to_f(tok):
    return np.nan if tok == "*" else float(tok.replace(",", "."))


def parse_year(year):
    cfg = CONFIG[year]
    rows = []
    for line in (RAW / f"rows_{year}.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        vals = toks[-cfg["nind"]:]
        assert all(is_val(t) for t in vals), f"{year}: bad values in: {line[:80]}"
        rest = toks[:-cfg["nind"]]
        wsk_tok = rest[-1]
        wsk = to_f(wsk_tok) if NUM.match(wsk_tok) else np.nan
        rest = rest[:-1]
        prev = rest[-cfg["nprev"]:]
        head = rest[:-cfg["nprev"]]
        rank = head[0]
        name = " ".join(head[1:])
        # 2026 has a TYP column (P/N) after the name
        typ = None
        if name.endswith(" P") or name.endswith(" N"):
            typ = name[-1]
            name = name[:-2]
        rows.append(dict(rank=rank, name=name, typ=typ, wsk=wsk,
                         **{f"ind_{i+1:02d}": to_f(v) for i, v in enumerate(vals)}))
    df = pd.DataFrame(rows)
    df["year"] = year
    return df


def fit_weights(df, nind):
    """NNLS fit of per-column weights against published WSK (validation)."""
    from scipy.optimize import nnls
    sub = df.dropna(subset=["wsk"])
    X = sub[[f"ind_{i+1:02d}" for i in range(nind)]].fillna(0).to_numpy()
    y = sub["wsk"].to_numpy()
    w, _ = nnls(X, y)
    pred = X @ w
    ss = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return w, ss


def aggregate(df, cfg):
    weights = cfg["weights"]
    col = 0
    out = pd.DataFrame({"name": df["name"], "rank": df["rank"],
                        "wsk": df["wsk"], "year": df["year"]})
    for gname, gw in zip(GROUPS, weights):
        gw = np.array(gw, dtype=float)
        cols = [f"ind_{col+i+1:02d}" for i in range(len(gw))]
        vals = df[cols].to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        score = vals @ gw / gw.sum()
        out[gname] = score
        out[gname + "_rel"] = 100 * score / score.max()
        col += len(gw)
    return out


def canonical(name):
    for canon, pats in TECHNICAL.items():
        for p in pats:
            if name.startswith(p) or p in name:
                return canon
    return None


def main():
    all_ind, all_crit = [], []
    for year in sorted(CONFIG):
        cfg = CONFIG[year]
        df = parse_year(year)
        try:
            w, r2 = fit_weights(df, cfg["nind"])
            hyp = np.concatenate([np.array(g, dtype=float) for g in cfg["weights"]])
            hyp_n = hyp / hyp.sum()
            w_n = w / w.sum() if w.sum() else w
            print(f"{year}: parsed {len(df)} rows | NNLS R2 vs WSK = {r2:.5f}")
            print(f"      fitted vs hypothesised weights (normalised, first 12):")
            print("      fit:", np.round(w_n[:12] * 100, 2))
            print("      hyp:", np.round(hyp_n[:12] * 100, 2))
        except Exception as e:  # scipy may be missing
            print(f"{year}: parsed {len(df)} rows | weight fit skipped ({e})")
        crit = aggregate(df, cfg)
        crit["canonical"] = crit["name"].map(canonical)
        df["canonical"] = df["name"].map(canonical)
        all_ind.append(df)
        all_crit.append(crit)

    ind = pd.concat(all_ind, ignore_index=True)
    crit = pd.concat(all_crit, ignore_index=True)
    ind.to_csv(OUT / "perspektywy_indicators_2022_2026.csv", index=False)
    crit.to_csv(OUT / "perspektywy_criteria_2022_2026.csv", index=False)
    tech = crit[crit["canonical"].notna()]
    tech.to_csv(OUT / "perspektywy_criteria_technical_2022_2026.csv", index=False)
    print("\nTechnical universities per year:")
    print(tech.groupby("year")["canonical"].count())
    missing = set(TECHNICAL) - set(tech["canonical"].unique())
    if missing:
        print("MISSING canonical:", missing)
    for year in sorted(CONFIG):
        got = set(tech[tech.year == year]["canonical"])
        miss = set(TECHNICAL) - got
        if miss:
            print(f"  {year} missing: {sorted(miss)}")


if __name__ == "__main__":
    main()
