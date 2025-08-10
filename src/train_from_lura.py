import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, classification_report
import joblib

THREAT = {"usv", "military_vessel"}  # tanker+ghost handled via feature

def heuristic_label(row):
    c = row["true_cls"]
    if c in THREAT: return 1
    if c == "merchant_tanker" and row.get("true_ais_anomaly",0)==1: return 1
    return 0

def main():
    Path("model").mkdir(parents=True, exist_ok=True)
    ev = pd.read_json("data/events_simulated.json")
    rp = pd.read_json("data/lura_reports.json")

    # join reports with truth at same timestamp+track (approx: use track_id only)
    # Note: for stricter join use nearest-in-time within small window.
    tr = ev[["track_id","timestamp","cls","ais_anomaly","lat","lon"]].rename(
        columns={"cls":"true_cls","ais_anomaly":"true_ais_anomaly"}
    )
    df = rp.merge(tr[["track_id","true_cls","true_ais_anomaly"]], left_on="target_track_id", right_on="track_id", how="left")
    df["true_cls"] = df["true_cls"].fillna("unknown")
    df["true_ais_anomaly"] = df["true_ais_anomaly"].fillna(0).astype(int)
    df["label"] = df.apply(heuristic_label, axis=1).astype(int)

    # features from LURA report only
    df["bearing_rad"] = np.deg2rad(df["bearing_deg"].astype(float))
    feat_cat = ["est_cls"]
    feat_num = ["est_confidence","range_m","ghost_hint", "bearing_rad"]
    groups = df["lura_id"]  # group by sensor

    X = df[feat_cat + feat_num]; y = df["label"].values

    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer([("cat", cat, feat_cat), ("num", num, feat_num)])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    grid = GridSearchCV(pipe, {"clf__C":[0.25,0.5,1.0,2.0], "clf__penalty":["l1","l2"]},
                        scoring="average_precision",
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        n_jobs=-1, refit=True)
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y[tr_idx], y[te_idx]
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    p = best.predict_proba(Xte)[:,1]
    ap = average_precision_score(yte, p); auc = roc_auc_score(yte, p)
    thr = float(np.quantile(p[yte==1], 0.15)) if (yte==1).any() else 0.5  # conservative

    yhat = (p >= thr).astype(int); f1 = f1_score(yte, yhat)
    print(grid.best_params_)
    print(f"AP={ap:.3f} AUC={auc:.3f} F1@thr={thr:.3f}={f1:.3f}")
    print(classification_report(yte, yhat, digits=3))

    joblib.dump(best, "model/lura_relevance.pkl")
    meta = {
        "features": feat_cat + feat_num,
        "threshold": thr,
        "notes": "trained on LURA reports",
    }
    with open("model/lura_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
