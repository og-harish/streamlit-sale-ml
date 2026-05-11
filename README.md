# Sales Forecast AI - Streamlit Edition

A Streamlit Community Cloud-ready professional sales forecasting workspace for flexible sales data upload, quota tracking, CRM-style forecast categories, territory rollups, Google sign-in, revenue prediction, Groq-powered insight answers, and downloadable executive reports.

## Features

- Full dashboard loads with the included sample dataset, then drag-and-drop CSV upload can replace it through an explicit **Process Dataset** workflow
- Automatic post-upload processing pipeline with field mapping, cleaning, duplicate handling, and engineered forecasting features
- Included messy CSV test file at `sample_data/messy_sales_test.csv` for validating auto-detection, missing-value cleanup, duplicate removal, forecasts, reports, and assistant answers
- Auto-detect flexible sales columns such as `order_date`, `city`, `product`, `qty`, `sales`, `amount`, `price`, `feedback`, and `comment`
- Manually map columns with dropdowns if the app cannot detect them automatically
- Work with messy CSV files by filling missing region/category/review values and calculating revenue from `quantity x price` when possible
- Clean missing values, remove duplicates, parse dates, and create time-series features
- Train a server-side predictive model leaderboard and automatically select the strongest validation model
- CRM-style left navigation for Home, Reports, Analytics, Sales modules, and Activity modules
- CRM-style Forecast Center inspired by professional sales forecasting workflows
- Quota, weighted forecast, closed revenue, quota gap, pipeline coverage, and confidence metrics
- Live Forecast Pulse for data quality, model health, forecast direction, anomaly risk, and next best actions
- Forecast categories for Closed, Committed, Best Case, Pipeline, and Omitted revenue
- Territory forecast rollups with manager ownership and risk status
- Pipeline stage probability chart for weighted revenue planning
- Native Streamlit Google sign-in using `st.login("google")`
- Download a generated Google Colab notebook for data loading, preprocessing, EDA, model training, evaluation, and future-sales export
- Google Colab preprocessing notebook linked from the website
- Editable PowerPoint project deck linked and downloadable from the website
- Show KPI cards, revenue trend, region sales, category performance, forecast charts, and anomaly alerts
- Show **Live Tamil Nadu E-Commerce Sales Prediction - 2026** with every-second simulated counter
- Predict May 2026 Tamil Nadu monthly, daily, hourly, minute, and second-level estimated sales
- Show statewide Tamil Nadu district/market predicted sales with live heatmap coverage across major commerce regions
- Predict 2026 India sales with today estimate, YTD actuals, remaining-year forecast, and full-year projection
- Live sales prediction cards for today, tomorrow, next 7 days, and next 30 days
- Analyze review sentiment, keywords, recurring issues, and NLP-based insight extraction for themes, intent signals, urgent feedback, and region/category sentiment
- Chatbot uses Groq or Gemini from backend Streamlit secrets only, with richer dataset-aware local fallback answers
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

6. Open advanced settings and paste secrets for Groq and Google authentication:

```toml
GROQ_API_KEY = "your_groq_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"

[auth]
redirect_uri = "https://your-app-name.streamlit.app/oauth2callback"
cookie_secret = "replace_with_a_long_random_secret"

[auth.google]
client_id = "your_google_client_id"
client_secret = "your_google_client_secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

7. Deploy.

## Project Asset Links

After this repository is pushed to GitHub, the website exposes these links under the header and on the Project Assets page:

- Google Colab preprocessing notebook: [Open in Colab](https://colab.research.google.com/github/og-harish/streamlit-sale-ml/blob/main/notebooks/sales_forecast_preprocessing_colab.ipynb)
- PowerPoint project deck: [Download PPTX](https://github.com/og-harish/streamlit-sale-ml/raw/main/project_assets/Sales_Forecast_AI_Project_Deck.pptx)
- GitHub repository: [og-harish/streamlit-sale-ml](https://github.com/og-harish/streamlit-sale-ml)

## Secrets

Do not commit real API keys. For local development, create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`. For Streamlit Cloud, paste the same TOML values in the app's secrets settings.

Recommended Streamlit Cloud secrets:

```toml
GROQ_API_KEY = "your_groq_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Optional fallback provider
GEMINI_API_KEY = "your_gemini_key_here"

[auth]
redirect_uri = "https://your-app-name.streamlit.app/oauth2callback"
cookie_secret = "replace_with_a_long_random_secret"

[auth.google]
client_id = "your_google_client_id"
client_secret = "your_google_client_secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

API keys are not shown in the app UI and are not committed to GitHub. The app reads them only from Streamlit Cloud secrets, local `.streamlit/secrets.toml`, or server environment variables.

To use your Groq key on the live Streamlit site, paste it in **Streamlit Cloud -> App settings -> Secrets** as `GROQ_API_KEY`. For Google sign-in, create a Google OAuth client, add your Streamlit callback URL, and paste the `auth` and `auth.google` blocks into the same Streamlit secrets panel. Do not paste real keys into `streamlit_app.py`, README, or GitHub files. Local development can use `.streamlit/secrets.toml`, which is already ignored by `.gitignore`.

If no API key is configured, the chatbot still works with local dataset-aware answers for questions about best regions, weak categories, forecasts, anomalies, model metrics, customer complaints, and recommendations.

## Next-Level Upgrades Included

- Rich AI context builder so chatbot answers vary by question and use real uploaded dataset metrics
- Quick-question handling that avoids repeating the same answer on every Streamlit rerun
- AI provider fallback: if Groq/Gemini fails, the app returns local business analysis instead of breaking
- Futuristic drag-and-drop uploader with multi-format dataset parsing
- Multi-model prediction leaderboard using RandomForest, ExtraTrees, and GradientBoosting with time-aware validation
- Expanded feature engineering with lag, rolling, volatility, calendar, weekend, city, and product signals
- Dedicated statewide Tamil Nadu live prediction page with district heatmap, live counter, and every-second velocity estimates
- Higher-contrast launch screen, cards, upload panel, sidebar, and floating chatbot button for better visibility
- Live forecast chart and executive insight text for professional sales review storytelling
- PDF report now includes live sales prediction values
- PDF report includes Tamil Nadu May 2026 live prediction, district/category forecasts, confidence score, and predictive-estimate disclaimer
- Flexible dataset engine supports different sales CSV schemas instead of forcing one exact column format

## Notes

- The sample data is India-focused and includes 2026 rows so the India forecast can show YTD actuals.
- Tamil Nadu live values are predictive simulations based on uploaded or demo trend patterns, not official live government data.
- Forecasts are business estimates, not financial guarantees.
- XGBoost and Prophet are included for the professional CSV processing workflow; keep an eye on Streamlit Cloud resource limits for very large datasets.
