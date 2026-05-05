# Sales Prediction AI - Streamlit Edition

A Streamlit Community Cloud-ready sales analytics app for CSV upload, preprocessing, revenue prediction, India 2026 live forecasting, review insights, anomaly detection, AI chatbot answers, and downloadable reports.

## Features

- Upload CSV sales data or use the included sample dataset
- Validate required columns: `date`, `region`, `product_category`, `units_sold`, `revenue`, `discount_pct`, `customer_reviews`
- Clean missing values, remove duplicates, parse dates, and create time-series features
- Train a browser/server-side Streamlit RandomForest revenue prediction model
- Show KPI cards, revenue trend, region sales, category performance, forecast charts, and anomaly alerts
- Predict 2026 India sales with today estimate, YTD actuals, remaining-year forecast, and full-year projection
- Analyze review sentiment, keywords, and recurring issues
- Chatbot uses Groq or Gemini from Streamlit secrets, with local fallback answers
- Download cleaned CSV, prediction CSV, and PDF report

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud Deployment

The official Streamlit docs say Community Cloud deploys from a GitHub repository by selecting the repository, branch, and entrypoint file. This app is organized with `streamlit_app.py` and `requirements.txt` in the repo root, which is the recommended layout.

1. Push this folder to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose **Create app**.
4. Select the GitHub repository and branch.
5. Set the main file path to:

```text
streamlit_app.py
```

6. Open advanced settings and paste secrets if you want AI chatbot answers:

```toml
GROQ_API_KEY = "your_groq_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

7. Deploy.

## Secrets

Do not commit real API keys. For local development, create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`. For Streamlit Cloud, paste the same TOML values in the app's secrets settings.

If no API key is configured, the chatbot still works with local rule-based answers.

## Notes

- The sample data is India-focused and includes 2026 rows so the India forecast can show YTD actuals.
- Forecasts are business estimates, not financial guarantees.
- For heavier models such as XGBoost or Prophet, add them later only if your Streamlit Cloud resource limits are comfortable.
