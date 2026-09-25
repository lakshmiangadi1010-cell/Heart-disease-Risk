# PulseCheck

Run the local app from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

The supplied CSV contains 70,000 rows of 10 input features but no `Heart_Risk` target column. The app therefore trains a logistic-regression demo model using a transparent proxy label based on weighted indicators. Training uses a reproducible stratified 80/20 split and exposes accuracy, precision, recall, F1, and row counts at `/api/health`; these metrics describe the proxy labels only. Replace the fallback in `train_model()` with real labeled outcomes before using this for clinical or production decisions.