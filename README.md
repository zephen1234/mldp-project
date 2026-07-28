# HDB Price Compass

HDB Price Compass is a Streamlit web application that estimates resale prices
for flats in Ang Mo Kio, Bishan and Toa Payoh. It uses the tuned Random Forest
pipeline developed and evaluated in the accompanying MLDP notebook.

## Project files

- `MLDP Program Codes Submission .ipynb` - model development, evaluation and
  written justification.
- `streamlit_app.py` - interactive web application.
- `model_utils.py` - shared cleaning, feature engineering and model pipeline.
- `train_model.py` - reproducibly trains and exports the final model.
- `hdb_price_pipeline.joblib` - trained model loaded by Streamlit.
- `model_metadata.json` - model metrics and valid interface options.
- `hdb.csv` - project dataset.
- `requirements.txt` - deployment dependencies.

## Run locally

Open a terminal in this folder and run:

```powershell
conda activate mldp
python train_model.py
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Commit and push every project file to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Select **Create app** and choose the `zephen1234/mldp-project` repository.
4. Set the branch to `main` and the main file path to `streamlit_app.py`.
5. Deploy the app and copy its public URL into the required Word document.

No secret keys are required.

## Model scope

The model is trained on HDB resale transactions from 2017 to May 2025 for
Ang Mo Kio, Bishan and Toa Payoh. The estimate is intended for early research
and does not replace an official valuation.
