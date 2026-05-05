# Sales Prediction AI - Streamlit Edition

A Streamlit Community Cloud-ready sales analytics app for flexible CSV upload, preprocessing, revenue prediction, Tamil Nadu 2026 live e-commerce simulation, India 2026 forecasting, review insights, anomaly detection, AI chatbot answers, and downloadable reports.

## Features

- Upload CSV sales data or use the included sample dataset
- Auto-detect flexible sales columns such as `order_date`, `city`, `product`, `qty`, `sales`, `amount`, `price`, `feedback`, and `comment`
- Manually map columns with dropdowns if the app cannot detect them automatically
- Work with partial data by filling missing region/category/review values and calculating revenue from `quantity x price` when possible
- Clean missing values, remove duplicates, parse dates, and create time-series features
- Train a browser/server-side Streamlit RandomForest revenue prediction model
- Show KPI cards, revenue trend, region sales, category performance, forecast charts, and anomaly alerts
- Show **Live Tamil Nadu E-Commerce Sales Prediction - 2026** with every-second simulated counter
- Predict May 2026 Tamil Nadu monthly, daily, hourly, minute, and second-level estimated sales
- Show Tamil Nadu city-wise and category-wise predicted sales for Chennai, Coimbatore, Madurai, Trichy, Salem, Tirunelveli, Erode, Vellore, and Thanjavur
- Predict 2026 India sales with today estimate, YTD actuals, remaining-year forecast, and full-year projection
- Live sales prediction cards for today, tomorrow, next 7 days, and next 30 days
- Analyze review sentiment, keywords, and recurring issues
- Chatbot uses Groq or Gemini from Streamlit secrets, with richer dataset-aware local fallback answers
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

Recommended Streamlit Cloud secrets:

```toml
GROQ_API_KEY = "your_groq_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Optional fallback provider
GEMINI_API_KEY = "your_gemini_key_here"
```

For quick testing, the sidebar also supports temporary Groq/Gemini keys. Temporary keys stay in the current Streamlit session and are not written to GitHub.

If no API key is configured, the chatbot still works with local dataset-aware answers for questions about best regions, weak categories, forecasts, anomalies, model metrics, customer complaints, and recommendations.

## Next-Level Upgrades Included

- Rich AI context builder so chatbot answers vary by question and use real uploaded dataset metrics
- Quick-question handling that avoids repeating the same answer on every Streamlit rerun
- AI provider fallback: if Groq/Gemini fails, the app returns local business analysis instead of breaking
- Live forecast chart and executive insight text for portfolio-ready dashboard storytelling
- PDF report now includes live sales prediction values
- PDF report includes Tamil Nadu May 2026 live prediction, city/category forecasts, confidence score, and predictive-estimate disclaimer
- Flexible dataset engine supports different sales CSV schemas instead of forcing one exact column format

## Notes

- The sample data is India-focused and includes 2026 rows so the India forecast can show YTD actuals.
- Tamil Nadu live values are predictive simulations based on uploaded or demo trend patterns, not official live government data.
- Forecasts are business estimates, not financial guarantees.
- For heavier models such as XGBoost or Prophet, add them later only if your Streamlit Cloud resource limits are comfortable.
