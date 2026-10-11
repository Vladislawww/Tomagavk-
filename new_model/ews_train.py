"""
Train the disruption early-warning model on ews_dataset.csv.

    python ews_train.py --data ews_dataset.csv --out ews_model.joblib

Split is BY SHOT (never by row): ~64% fit / ~16% calibration / ~20% test shots.
  fit  -> gradient boosting (unweighted, so probabilities keep the true base rate)
  cal  -> Platt scaling + alarm threshold (healthy-shot false-alarm target)
  test -> final report only
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import ews_pipeline as E


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def predict_proba(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Calibrated P(quench within horizon) for rows of a dataset built by ews_pipeline."""
    raw = bundle["model"].predict_proba(df[bundle["features"]].to_numpy(np.float32))[:, 1]
    return bundle["platt"].predict_proba(logit(raw))[:, 1]


def shot_report(df: pd.DataFrame, p: np.ndarray, thr: float, horizon: float) -> dict:
    """Shot-level metrics at a fixed alarm threshold."""
    s = pd.Series(p, index=df.index)
    healthy = s[~df.shot_disrupts].groupby(df.shot_id).max()
    leads = []
    for _, g in df[df.shot_disrupts].groupby("shot_id"):
        a = g[(s.loc[g.index] > thr).to_numpy()]
        w = a[a.t_to_quench_s <= horizon + 1e-6]
        leads.append(w.t_to_quench_s.max() if len(w) else np.nan)   # earliest alarm inside the window
    leads = pd.Series(leads)
    return dict(healthy_shots=len(healthy), false_alarm_rate=float((healthy > thr).mean()),
                disrupted_shots=len(leads), detect_rate=float(leads.notna().mean()),
                median_lead_s=float(leads.median()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ews_dataset.csv")
    ap.add_argument("--out", default="ews_model.joblib")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--far", type=float, default=0.10, help="target false-alarm rate on healthy shots")
    ap.add_argument("--horizon", type=float, default=0.5, help="must match the horizon used to build the dataset")
    a = ap.parse_args()

    df = pd.read_csv(a.data)

    # Якщо сирий файл не містить мітки y — запускаємо пайплайн обробки автоматично
    if "y" not in df.columns:
        print(f"[pipeline] Колонка 'y' відсутня в {a.data}. Обробка сирих даних на льоту...")
        cfg = E.Config(horizon_s=a.horizon)
        df = E.build_dataset(a.data, cfg)

    # Відбираємо виключно числові ознаки (відсікаємо текст на зразок 'device', 'jet')
    feats = [c for c in E.feature_columns(df) if pd.api.types.is_numeric_dtype(df[c])]
    X, y = df[feats].to_numpy(np.float32), df.y.to_numpy()

    # --- shot-level split: test shots first, then calibration shots out of the rest
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=a.seed).split(df, df.shot_disrupts, df.shot_id))
    sub = df.iloc[tr]
    fi, ci = next(StratifiedGroupKFold(4, shuffle=True, random_state=a.seed).split(sub, sub.shot_disrupts, sub.shot_id))
    fit, cal = tr[fi], tr[ci]
    assert not (set(df.shot_id.iloc[fit]) & set(df.shot_id.iloc[te])), "shot leakage between fit and test"
    assert not (set(df.shot_id.iloc[cal]) & set(df.shot_id.iloc[te])), "shot leakage between cal and test"

    # --- train + calibrate
    model = HGB(max_iter=150, random_state=a.seed).fit(X[fit], y[fit])
    platt = LogisticRegression(C=1e6).fit(logit(model.predict_proba(X[cal])[:, 1]), y[cal])
    bundle = dict(model=model, platt=platt, features=feats, horizon_s=a.horizon)

    # --- alarm threshold from CALIBRATION shots only (so the test report is honest)
    p_cal = predict_proba(bundle, df.iloc[cal])
    healthy_max = pd.Series(p_cal, index=df.index[cal])[~df.shot_disrupts.iloc[cal]].groupby(df.shot_id.iloc[cal]).max()
    bundle["threshold"] = float(np.quantile(healthy_max, 1 - a.far))

    # --- test report
    dte, p = df.iloc[te], predict_proba(bundle, df.iloc[te])
    yt = dte.y.to_numpy()
    print(f"shots: fit {df.shot_id.iloc[fit].nunique()} | cal {df.shot_id.iloc[cal].nunique()} | test {dte.shot_id.nunique()}"
          f" (disrupted in test: {dte[dte.shot_disrupts].shot_id.nunique()})")
    print(f"row-level  AUC={roc_auc_score(yt, p):.3f}  AP={average_precision_score(yt, p):.3f}  "
          f"Brier={brier_score_loss(yt, p):.4f}  (base rate {yt.mean():.4f})")
    print("shot-level", {k: round(v, 3) for k, v in shot_report(dte, p, bundle["threshold"], a.horizon).items()},
          f"threshold={bundle['threshold']:.3f}")
    joblib.dump(bundle, a.out)
    print("saved", a.out)

if __name__ == "__main__":
    main()
