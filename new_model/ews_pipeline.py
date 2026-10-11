"""
Early-Warning-System preprocessing for tokamak disruption prediction (toksym data).

Pipeline order matters:
    load -> drop leaky cols -> locate quench (full shot) -> trim ends (Ip) ->
    causal rolling features -> label expansion + drop "too late" rows -> shot-level CV

Usage:
    python ews_pipeline.py --input timeseries_clean.csv --output ews_dataset.csv \
        --horizon 0.5 --min-warning 0.02 --ip-min 0.1 --feature-mode raw_rolling [--evaluate]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# Column groups
# ----------------------------------------------------------------------------------------
LEAKY_COLS = ["disruption_risk", "status", "seed", "time_to_end_s"]   # simulator output / label proxies
DIAG_ONLY = ["diag_d_alpha", "diag_soft_xray", "diag_locked_mode",    # diagnostics with no clean twin
             "diag_neutron_rate", "diag_v_loop_V"]
PHYS_TREND = ["f_greenwald", "beta_n", "q95", "li", "te0_keV",        # slow "margin" signals: we want trends
              "w_th_MJ", "p_rad_MW"]
META_COLS = ["shot_id", "time_s", "y", "t_to_quench_s", "shot_disrupts"]  # NEVER features


@dataclass
class Config:
    ip_min_MA: float = 0.1          # trim shot ends where Ip < threshold (ramp-up / ramp-down)
    horizon_s: float = 0.5          # positive window: [quench - horizon, quench - min_warning]
    min_warning_s: float = 0.02     # rows closer to the quench than this are "too late to act" -> dropped
    collapse_ratio: float = 0.5     # thermal quench = one-step drop of te0 below this fraction
    quench_col: str = "te0_keV"
    windows_s: tuple = (0.1, 0.5)   # causal rolling windows for diagnostics
    feature_mode: str = "raw_rolling"   # raw | raw_rolling | rolling_only


# ----------------------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------------------
def load_data(path: str) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(path).sort_values(["shot_id", "time_s"]).reset_index(drop=True)
    dt = float(df.groupby("shot_id").time_s.diff().median())   # sampling step (0.02 s here)
    return df, dt


# ----------------------------------------------------------------------------------------
# 2. Remove leakage
# ----------------------------------------------------------------------------------------
def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """disruption_risk is the simulator's own risk output (AUC 0.92 alone, freezes after the
    label); status/seed/time_to_end_s are label proxies. `disrupted` is kept for now ONLY to
    identify disrupted shots + the precursor onset; it is dropped at the end."""
    return df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])


# ----------------------------------------------------------------------------------------
# 3. Locate the thermal quench per shot (replaces the `diag_locked_mode >= 5.0` rule)
# ----------------------------------------------------------------------------------------
def locate_quench(df: pd.DataFrame, cfg: Config, dt: float) -> pd.Series:
    """Quench time per disrupted shot (NaN for healthy shots).

    diag_locked_mode >= 5 is NOT a usable marker: it appears in only 20/85 disrupted shots.
    The reliable signature is a one-step collapse of te0 after the precursor onset (79/85).
    Fallback (6/85): the simulator ended the shot on the quench row, which was already trimmed
    away -> quench time = last available time + dt.
    Searching only AFTER the onset prevents ELM-related dips from being mistaken for a quench."""
    tq, n_fallback = {}, 0
    for sid, g in df.groupby("shot_id", sort=False):
        flag = g["disrupted"].to_numpy().astype(bool)
        if not flag.any():
            continue                                    # healthy shot
        te, t = g[cfg.quench_col].to_numpy(), g["time_s"].to_numpy()
        onset = int(flag.argmax())
        ratio = te[1:] / np.maximum(te[:-1], 1e-9)
        hit = np.flatnonzero(ratio[onset:] < cfg.collapse_ratio)
        if hit.size:
            tq[sid] = t[onset + hit[0] + 1]             # first row already in collapsed state
        else:
            tq[sid] = t[-1] + dt
            n_fallback += 1
    print(f"[quench] {len(tq)} disrupted shots, {n_fallback} via end-of-shot fallback")
    return pd.Series(tq, name="t_quench")


# ----------------------------------------------------------------------------------------
# 4. Trim shot ends (configurable Ip threshold)
# ----------------------------------------------------------------------------------------
def trim_ends(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep the contiguous span between the first and last row with Ip >= threshold, per shot.
    (A plain row filter would also cut mid-shot Ip dips and leave holes in the time axis.)"""
    pos = df.groupby("shot_id").cumcount()
    ok_pos = pos.where(df["ip_MA"] >= cfg.ip_min_MA)
    first = ok_pos.groupby(df["shot_id"]).transform("min")
    last = ok_pos.groupby(df["shot_id"]).transform("max")
    keep = (pos >= first) & (pos <= last)
    print(f"[trim] Ip>={cfg.ip_min_MA} MA: dropped {int((~keep).sum())} rows")
    return df.loc[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# 5. Feature strategy: causal rolling statistics, per shot
# ----------------------------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame, cfg: Config, dt: float) -> pd.DataFrame:
    """All windows look BACKWARD only and never cross shot boundaries (groupby shot_id).
    - spiky diagnostics: rolling mean / std / max  -> "has activity been building up?"
    - slow margins:      x(t) - x(t - w)           -> trend toward a limit (density, beta, q95)
    A single instantaneous spike no longer decides the prediction, and trends give the model
    something to learn from well before the final precursor rows."""
    g = df.groupby("shot_id")
    new = {}
    for w in cfg.windows_s:
        n = max(int(round(w / dt)), 2)
        tag = f"{int(round(w * 1000))}ms"
        for c in [c for c in DIAG_ONLY if c in df.columns]:
            r = g[c].rolling(n, min_periods=1)
            new[f"{c}__mean{tag}"] = r.mean().reset_index(level=0, drop=True)
            new[f"{c}__max{tag}"] = r.max().reset_index(level=0, drop=True)
            new[f"{c}__std{tag}"] = (g[c].rolling(n, min_periods=2).std()
                                     .reset_index(level=0, drop=True).fillna(0.0))
        for c in [c for c in PHYS_TREND if c in df.columns]:
            new[f"{c}__slope{tag}"] = (df[c] - g[c].shift(n)).fillna(0.0)
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    if cfg.feature_mode == "rolling_only":              # forbid the model from seeing raw spikes
        out = out.drop(columns=[c for c in DIAG_ONLY if c in out.columns])
    return out


# ----------------------------------------------------------------------------------------
# 6. Target engineering: label expansion + removal of trivial / too-late rows
# ----------------------------------------------------------------------------------------
def make_labels(df: pd.DataFrame, tq: pd.Series, cfg: Config) -> pd.DataFrame:
    """y = 1  if  min_warning <= (t_quench - t) <= horizon
       dropped if (t_quench - t) < min_warning   (includes ALL post-quench rows: trivial positives,
                                                   and the last instants where no action is possible)
       y = 0  otherwise (healthy shots + early part of disrupted shots)"""
    eps = 1e-6
    d = df["shot_id"].map(tq) - df["time_s"]            # time to quench; NaN for healthy shots
    df = df.assign(t_to_quench_s=d, shot_disrupts=d.notna())
    too_late = d < cfg.min_warning_s - eps              # NaN compares False -> healthy rows kept
    df["y"] = ((d >= cfg.min_warning_s - eps) & (d <= cfg.horizon_s + eps)).astype(int)
    print(f"[labels] horizon={cfg.horizon_s}s, min_warning={cfg.min_warning_s}s: "
          f"dropped {int(too_late.sum())} too-late rows")
    return df.loc[~too_late].drop(columns=["disrupted"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# 7. Orchestration
# ----------------------------------------------------------------------------------------
def build_dataset(path: str, cfg: Config) -> pd.DataFrame:
    df, dt = load_data(path)
    df = drop_leakage(df)
    tq = locate_quench(df, cfg, dt)          # on the FULL shot, before any trimming
    df = trim_ends(df, cfg)
    if cfg.feature_mode != "raw":
        df = add_rolling_features(df, cfg, dt)
    df = make_labels(df, tq, cfg)
    feats = feature_columns(df)
    df = df[META_COLS + feats]
    print(f"[done] {df.shape[0]} rows, {len(feats)} features, positives {int(df.y.sum())} "
          f"({df.y.mean():.2%}), disrupted shots {df.loc[df.shot_disrupts, 'shot_id'].nunique()}")
    return df


def feature_columns(df: pd.DataFrame, include_time: bool = False) -> list[str]:
    drop = set(META_COLS) - ({"time_s"} if include_time else set())
    return [c for c in df.columns if c not in drop]


# ----------------------------------------------------------------------------------------
# 8. Validation: group-aware CV + shot-level early-warning metrics
# ----------------------------------------------------------------------------------------
def evaluate(df: pd.DataFrame, n_splits: int = 5, far_target: float = 0.10, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    X_cols = feature_columns(df)
    oof = np.zeros(len(df))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(df, df["shot_disrupts"], df["shot_id"]):   # split by SHOT, never by row
        m = HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=seed)
        m.fit(df.iloc[tr][X_cols], df["y"].iloc[tr])
        oof[te] = m.predict_proba(df.iloc[te][X_cols])[:, 1]
    p = pd.Series(oof, index=df.index)

    # Shot-level: pick the threshold that lets <= far_target of healthy shots alarm at all
    healthy_max = p[~df.shot_disrupts].groupby(df.shot_id).max()
    thr = float(np.quantile(healthy_max, 1 - far_target))
    warn = []
    for sid, g in df[df.shot_disrupts].groupby("shot_id"):
        a = g.index[p[g.index] > thr]
        warn.append(g.loc[a[0], "t_to_quench_s"] if len(a) else np.nan)
    warn = pd.Series(warn)
    return dict(auc=roc_auc_score(df.y, p), ap=average_precision_score(df.y, p),
                pos_rate=df.y.mean(), thr=thr, detect_rate=warn.notna().mean(),
                warn_median_s=warn.median(), warn_p25_s=warn.quantile(0.25))


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output")
    ap.add_argument("--horizon", type=float, default=0.5)
    ap.add_argument("--min-warning", type=float, default=0.02)
    ap.add_argument("--ip-min", type=float, default=0.1)
    ap.add_argument("--feature-mode", default="raw_rolling", choices=["raw", "raw_rolling", "rolling_only"])
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    cfg = Config(ip_min_MA=a.ip_min, horizon_s=a.horizon, min_warning_s=a.min_warning,
                 feature_mode=a.feature_mode)
    data = build_dataset(a.input, cfg)
    if a.output:
        data.to_csv(a.output, index=False, float_format="%.6g")
    if a.evaluate:
        print(evaluate(data))
