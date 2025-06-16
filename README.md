# Flask CCPM Project

This is a simple Flask-based project that predicts Carbon Credit Prices using a pre-trained `.pkl` machine learning model.

## Files

- `app.py` – Flask API to serve predictions
- `cc_price_model.pkl` – Trained ML model
- `env/` – Virtual environment (ignored from Git)
- `.gitignore` – Excludes unnecessary files
- `requirements.txt` – (Generate this using `pip freeze`)

## How to Run

```bash
# Activate virtual environment
source env/Scripts/activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
