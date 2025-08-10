import json, numpy as np, pandas as pd, joblib
from pathlib import Path
from datetime import datetime

def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    df = pd.read_json("data/lura_reports.json")

    pipe = joblib.load("model/lura_relevance.pkl")
    meta = json.load(open("model/lura_metadata.json","r",encoding="utf-8"))
    feats = meta["features"]; thr = float(meta["threshold"])

    # feature hygiene
    df["bearing_rad"] = np.deg2rad(df["bearing_deg"].astype(float))
    for c in ["est_confidence","range_m","ghost_hint","bearing_rad"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "est_cls" not in df: df["est_cls"] = "unknown"

    X = df[[*feats]]
    prob = pipe.predict_proba(X)[:,1]
    df["relevant_prob"] = prob
    df["relevant"] = (prob >= thr).astype(int)

    # ROE (placeholder, à ajuster plus tard)
    adj = np.full(len(df), thr)
    adj[(df["est_cls"]=="usv")] *= 0.85
    adj[(df["est_cls"]=="military_vessel")] *= 0.90
    adj[(df["est_cls"]=="merchant_tanker") & (df["ghost_hint"]==1)] *= 0.95

    df["engage"] = (df["relevant_prob"] >= adj).astype(int)

    # outputs
    df.to_csv("data/lura_scored_reports.csv", index=False)
    summary = {
        "generated_at": datetime.utcnow().isoformat()+"Z",
        "reports": int(len(df)),
        "relevant": int(df["relevant"].sum()),
        "engage": int(df["engage"].sum()),
        "ratio_relevant": float(df["relevant"].mean()),
        "ratio_engage": float(df["engage"].mean()),
    }
    pd.Series(summary).to_json("data/lura_summary.json", indent=2)

if __name__ == "__main__":
    main()