# ISR to Strike Replay

> Research prototype **inspired by Helsing products** (SG-1 subsea drones and HX-1/2 strike systems). Synthetic data only.

## Project Description
End-to-end pipeline: simulate maritime traffic in the Gulf of Finland, run **SG1-like** autonomous subsea patrols to collect observations, evaluate coverage, adapt patrols, and (next) sketch HX-style strike recommendations.

## Features
- Oriented corridor (parallelogram) with merchant cargo/tankers, small boats, USVs, military vessels, ghost tankers (AIS anomaly).
- ~100 **SG1-inspired** drones: lawnmower patrols, reactive investigate, probabilistic sensing.
- Coverage metrics (track/point recall), miss heatmap, **cell_weights.json** to bias next patrol.
- Training/scoring from **LURA reports** (not raw simulated truth).
- OSM basemap previews (Cartopy).

## Installation

git clone https://github.com/Romain-Jaffuel/ISR2Strike-Replay.git cd <PROJECT_FOLDER>
python -m venv venv312
# Windows: venv312\Scripts\activate   # macOS/Linux: source venv312/bin/activate
pip install -r requirements.txt

# 1) Simulate traffic
python src/simulate_events.py

# 2) Patrol & collect (uses out/cell_weights.json if present)
python src/lura_patrol.py

# 3) Visualize patrols
python src/viz_lura.py

# 4) Evaluate & generate adaptive weights
python src/lura_eval.py

# 5) Train/Score from LURA data
python src/train_from_lura.py
python src/score_from_lura.py

## Notes
All data is synthetic and does not represent any real-world scenario.