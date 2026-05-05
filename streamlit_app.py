import io
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from fpdf import FPDF, XPos, YPos
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REQUIRED_COLUMNS = [
    "date",
    "region",
    "product_category",
    "units_sold",
    "revenue",
    "discount_pct",
    "customer_reviews",
]

NUMERIC_FEATURES = [
    "units_sold",
    "discount_pct",
    "month",
    "year",
    "day_of_week",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
]

CATEGORICAL_FEATURES = ["region", "product_category"]


st.set_page_config(
    page_title="Sales Prediction AI",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.3rem; }
    .hero {
        padding: 2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #103B37 0%, #1F6D5B 55%, #D8894C 100%);
        color: #FFF9EF;
        box-shadow: 0 24px 80px rgba(16, 59, 55, .28);
    }
    .hero h1 { font-size: 2.6rem; margin-bottom: .4rem; }
    .hero p { color: rgba(255, 249, 239, .82); font-size: 1.05rem; }
    .metric-card {
        border: 1px solid rgba(23, 48, 46, .08);
        background: rgba(255,255,255,.86);
        padding: 1rem 1.15rem;
        border-radius: 22px;
        box-shadow: 0 16px 45px rgba(23, 48, 46, .08);
    }
    .metric-label { color: #4E6F67; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 800; }
    .metric-value { color: #17302E; font-size: 1.7rem; font-weight: 900; margin-top: .35rem; }
    .metric-detail { color: rgba(23, 48, 46, .62); font-size: .86rem; margin-top: .2rem; }
    .alert-box {
        border-radius: 18px;
        padding: .9rem 1rem;
        background: rgba(216, 137, 76, .13);
        border: 1px solid rgba(216, 137, 76, .25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def inr(value):
    return f"INR {float(value or 0):,.0f}"


def get_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


@st.cache_data
def load_sample_data():
    return pd.read_csv("sample_data/sample_sales.csv")


def validate_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


def preprocess_data(df):
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    clean = df[REQUIRED_COLUMNS].copy()
    clean = clean.drop_duplicates()

    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean = clean.dropna(subset=["date"])
    clean["region"] = clean["region"].fillna("Unknown").astype(str).str.strip()
    clean["product_category"] = clean["product_category"].fillna("Unknown").astype(str).str.strip()
    clean["customer_reviews"] = clean["customer_reviews"].fillna("").astype(str)

    for col in ["units_sold", "revenue", "discount_pct"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean[col] = clean[col].fillna(clean[col].median() if clean[col].notna().any() else 0)

    clean = clean.sort_values("date")
    clean["month"] = clean["date"].dt.month
    clean["year"] = clean["date"].dt.year
    clean["day_of_week"] = clean["date"].dt.dayofweek
    clean["lag_7"] = clean["revenue"].shift(7)
    clean["lag_30"] = clean["revenue"].shift(30)
    clean["rolling_mean_7"] = clean["revenue"].rolling(7, min_periods=1).mean().shift(1)

    for col in ["lag_7", "lag_30", "rolling_mean_7"]:
        clean[col] = clean[col].fillna(clean["revenue"].expanding().mean())
        clean[col] = clean[col].fillna(clean["revenue"].mean())

    return clean.reset_index(drop=True)


def build_model(df):
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = df[features]
    y = df["revenue"]

    if len(df) < 8:
        raise ValueError("Need at least 8 cleaned rows to train a prediction model.")

    test_size = 0.2 if len(df) >= 20 else 0.3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=180,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions)) if len(y_test) > 1 else 0.0

    validation = pd.DataFrame(
        {
            "date": df.loc[y_test.index, "date"],
            "actual": y_test.values,
            "predicted": predictions,
        }
    )

    return model, {"RMSE": rmse, "MAE": mae, "R2": r2}, validation


def daily_revenue(df):
    return df.groupby("date", as_index=False).agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))


def forecast_daily_sales(df, periods=30):
    daily = daily_revenue(df).sort_values("date")
    values = daily["revenue"].to_numpy(dtype=float)
    if len(values) == 0:
        return pd.DataFrame()

    avg7 = np.mean(values[-7:]) if len(values) >= 7 else np.mean(values)
    avg14 = np.mean(values[-14:]) if len(values) >= 14 else avg7
    avg30 = np.mean(values[-30:]) if len(values) >= 30 else avg14
    moving_avg = avg7 * 0.55 + avg14 * 0.25 + avg30 * 0.2

    recent = values[-30:] if len(values) >= 2 else values
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0] if len(recent) > 1 else 0
    trend_pct = float(np.clip(slope / max(moving_avg, 1), -0.04, 0.04))
    global_avg = np.mean(values)

    weekday_factor = {}
    for weekday in range(7):
        weekday_values = daily[daily["date"].dt.dayofweek == weekday]["revenue"]
        weekday_factor[weekday] = float(np.clip(weekday_values.mean() / global_avg, 0.8, 1.25)) if len(weekday_values) else 1

    last_date = daily["date"].max()
    forecast_rows = []
    for day in range(1, periods + 1):
        next_date = last_date + timedelta(days=day)
        factor = weekday_factor.get(next_date.dayofweek, 1)
        trend_factor = float(np.clip(1 + trend_pct * day, 0.7, 1.45))
        forecast_rows.append({"date": next_date, "forecast": max(0, moving_avg * factor * trend_factor)})

    actual_chart = daily.tail(45).copy()
    actual_chart["forecast"] = np.nan
    forecast_chart = pd.DataFrame(forecast_rows)
    forecast_chart["revenue"] = np.nan
    return pd.concat([actual_chart, forecast_chart], ignore_index=True)


def india_2026_forecast(df):
    india_rows = df[df["region"].str.contains("india|bharat", case=False, na=False)]
    market_df = india_rows if len(india_rows) else df
    scope = "Filtered to India/Bharat rows" if len(india_rows) else "All uploaded regions treated as India market"

    daily = daily_revenue(market_df).sort_values("date")
    values = daily["revenue"].to_numpy(dtype=float)
    avg7 = np.mean(values[-7:]) if len(values) >= 7 else np.mean(values)
    avg14 = np.mean(values[-14:]) if len(values) >= 14 else avg7
    avg30 = np.mean(values[-30:]) if len(values) >= 30 else avg14
    moving_avg = avg7 * 0.55 + avg14 * 0.25 + avg30 * 0.2
    recent = values[-30:] if len(values) >= 2 else values
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0] if len(recent) > 1 else 0
    trend_pct = float(np.clip(slope / max(moving_avg, 1), -0.04, 0.04))
    global_avg = np.mean(values)
    last_date = daily["date"].max()
    today = pd.Timestamp(date.today())
    as_of = min(max(today, pd.Timestamp("2026-01-01")), pd.Timestamp("2026-12-31"))
    actual_map = dict(zip(daily["date"].dt.strftime("%Y-%m-%d"), daily["revenue"]))

    def forecast_for(day):
        days_ahead = max(1, (day - last_date).days)
        weekday_values = daily[daily["date"].dt.dayofweek == day.dayofweek]["revenue"]
        weekday = float(np.clip(weekday_values.mean() / global_avg, 0.8, 1.25)) if len(weekday_values) else 1
        damped_trend = trend_pct * np.sqrt(days_ahead) * 2.5
        trend_factor = float(np.clip(1 + damped_trend, 0.58, 1.85))
        return max(0, moving_avg * weekday * trend_factor)

    rows = []
    for day in pd.date_range("2026-01-01", "2026-12-31", freq="D"):
        key = day.strftime("%Y-%m-%d")
        actual = actual_map.get(key) if day <= as_of else None
        forecast = None if actual is not None else forecast_for(day)
        rows.append({"date": day, "actual": actual, "forecast": forecast, "value": actual if actual is not None else forecast})

    year_df = pd.DataFrame(rows)
    ytd_actual = float(year_df[(year_df["date"] <= as_of) & year_df["actual"].notna()]["actual"].sum())
    remaining_forecast = float(year_df[year_df["date"] > as_of]["value"].sum())
    full_year = float(year_df["value"].sum()) if ytd_actual == 0 else ytd_actual + remaining_forecast
    today_estimate = float(year_df[year_df["date"] == as_of]["value"].iloc[0])
    next_90 = float(year_df[year_df["date"] > as_of].head(90)["value"].sum())
    run_rate = max(moving_avg * 365, 1)
    growth = ((full_year - run_rate) / run_rate) * 100
    volatility = float(np.std(values[-30:]) / avg30) if avg30 else 1
    confidence = "High" if len(values) >= 60 and volatility < 0.3 else "Medium" if len(values) >= 24 and volatility < 0.55 else "Low"

    monthly = (
        year_df.assign(month=year_df["date"].dt.strftime("%b"))
        .groupby("month", sort=False)
        .agg(actual=("actual", "sum"), forecast=("forecast", "sum"))
        .reset_index()
    )

    return {
        "scope": scope,
        "as_of": as_of.date().isoformat(),
        "today_estimate": today_estimate,
        "ytd_actual": ytd_actual,
        "remaining_forecast": remaining_forecast,
        "next_90": next_90,
        "full_year": full_year,
        "growth": growth,
        "direction": "increase" if growth >= 0 else "decrease",
        "confidence": confidence,
        "monthly": monthly,
    }


def analyze_reviews(df):
    positive = {"excellent", "great", "good", "loved", "love", "fresh", "quick", "fast", "helpful", "smooth", "friendly", "premium", "value"}
    negative = {"delay", "late", "damaged", "bad", "slow", "poor", "issue", "stock", "return", "broken", "missing"}
    issues = {
        "Delivery delay": ["delay", "late", "delivery", "shipping"],
        "Damaged packaging": ["damaged", "packaging", "broken"],
        "Stock availability": ["out of stock", "stock", "missing"],
        "Returns or exchange": ["return", "exchange", "refund"],
        "Pricing or discount": ["price", "discount", "cost"],
    }

    sentiment = {"positive": 0, "neutral": 0, "negative": 0}
    issue_counts = {name: 0 for name in issues}
    keywords = {}

    for text in df["customer_reviews"].fillna("").astype(str).str.lower():
        words = [word.strip(".,!?;:\"'()[]") for word in text.split()]
        score = sum(word in positive for word in words) - sum(word in negative for word in words)
        if score > 0:
            sentiment["positive"] += 1
        elif score < 0:
            sentiment["negative"] += 1
        else:
            sentiment["neutral"] += 1
        for word in words:
            if len(word) > 3 and word not in {"this", "that", "with", "product", "sales"}:
                keywords[word] = keywords.get(word, 0) + 1
        for issue, patterns in issues.items():
            if any(pattern in text for pattern in patterns):
                issue_counts[issue] += 1

    total = max(len(df), 1)
    sentiment_pct = {key: round(value / total * 100, 1) for key, value in sentiment.items()}
    top_keywords = sorted(keywords.items(), key=lambda item: item[1], reverse=True)[:8]
    top_issues = sorted(issue_counts.items(), key=lambda item: item[1], reverse=True)
    return sentiment_pct, top_keywords, top_issues


def detect_anomalies(df):
    daily = daily_revenue(df).sort_values("date")
    if len(daily) < 10:
        return pd.DataFrame()

    model = IsolationForest(contamination=0.08, random_state=42)
    daily["rolling_mean_7"] = daily["revenue"].rolling(7, min_periods=1).mean()
    daily["pct_change"] = daily["revenue"].pct_change().fillna(0)
    daily["anomaly"] = model.fit_predict(daily[["revenue", "rolling_mean_7", "pct_change"]])
    anomalies = daily[daily["anomaly"] == -1].copy()
    anomalies["type"] = np.where(anomalies["pct_change"] >= 0, "Revenue Spike", "Revenue Drop")
    return anomalies.tail(10).sort_values("date", ascending=False)


def business_summary(df, kpis, sentiment_pct, issues, anomalies, india_forecast):
    weak_region = df.groupby("region")["revenue"].sum().idxmin()
    top_issue = next((issue for issue, count in issues if count > 0), "no repeated issue detected")
    anomaly_text = f"{len(anomalies)} anomaly alert(s) detected" if len(anomalies) else "No major anomaly detected"
    summary = (
        f"The dataset contains {len(df):,} cleaned rows with total revenue of {inr(kpis['total_revenue'])} "
        f"and {kpis['total_units']:,.0f} units sold. {kpis['best_region']} is the strongest region, "
        f"while {weak_region} needs attention. Customer sentiment is {sentiment_pct['positive']}% positive "
        f"and {sentiment_pct['negative']}% negative. {anomaly_text}. "
        f"India 2026 full-year sales are projected at {inr(india_forecast['full_year'])}."
    )
    recommendations = [
        f"Protect momentum in {kpis['best_region']} by repeating the campaigns, inventory planning, and service practices working there.",
        f"Investigate {weak_region} before increasing spend; review discounts, delivery experience, and local product mix.",
        f"Fix {top_issue.lower()} first because repeated review themes usually explain hidden sales friction.",
        "Compare discount percentage against revenue lift so promotions grow demand without eroding margin.",
        "Review anomaly alerts weekly and assign an owner for every sharp revenue drop.",
    ]
    return summary, recommendations


def ask_ai(question, context):
    groq_key = get_secret("GROQ_API_KEY")
    gemini_key = get_secret("GEMINI_API_KEY")
    prompt = f"You are a practical business analyst. Answer from this sales context only.\n\nContext:\n{context}\n\nQuestion:\n{question}"

    if groq_key:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": get_secret("GROQ_MODEL") or "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "Answer concisely using only the provided dataset context."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.25,
                "max_tokens": 500,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip(), "Groq"

    if gemini_key:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip(), "Gemini"

    lower = question.lower()
    if "india" in lower or "2026" in lower:
        return f"India 2026 sales are projected at {context['india_full_year']} with {context['india_confidence']} confidence.", "Local fallback"
    if "highest" in lower or "best region" in lower:
        return f"The highest-sales region is {context['best_region']} with revenue of {context['best_region_revenue']}.", "Local fallback"
    if "recommend" in lower:
        return "Focus on the best region, fix repeated customer issues, monitor anomalies, and validate discount ROI.", "Local fallback"
    return context["summary"], "Local fallback"


def pdf_report(summary, recommendations, kpis, india_forecast, anomalies):
    pdf = FPDF()
    pdf.add_page()
    text_width = 180
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(text_width, 10, "Sales Prediction AI Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(text_width, 6, f"Total revenue: {inr(kpis['total_revenue'])}\nTotal units: {kpis['total_units']:,.0f}\nBest region: {kpis['best_region']}\nBest category: {kpis['best_category']}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "India 2026 Forecast", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(text_width, 6, f"As of: {india_forecast['as_of']}\nToday estimate: {inr(india_forecast['today_estimate'])}\nRemaining forecast: {inr(india_forecast['remaining_forecast'])}\nFull-year projection: {inr(india_forecast['full_year'])}\nConfidence: {india_forecast['confidence']}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "Business Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(text_width, 6, summary)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "Recommendations", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    for rec in recommendations:
        pdf.multi_cell(text_width, 6, f"- {rec}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "Anomalies", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    if len(anomalies):
        for _, row in anomalies.head(6).iterrows():
            pdf.multi_cell(text_width, 6, f"- {row['date'].date()} {row['type']}: {inr(row['revenue'])}")
    else:
        pdf.multi_cell(text_width, 6, "No major anomalies detected.")
    return bytes(pdf.output())


def metric_card(label, value, detail):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>Sales Prediction System with NLP Insights</h1>
        <p>Upload CSV sales data, forecast revenue, analyze reviews, detect anomalies, chat with AI, and export reports. Built for Streamlit Community Cloud.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Sales AI")
    uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"])
    use_sample = st.button("Load sample dataset", width="stretch")
    st.caption("Required columns: date, region, product_category, units_sold, revenue, discount_pct, customer_reviews")
    st.divider()
    st.caption("Secrets for Streamlit Cloud")
    st.code("GROQ_API_KEY = \"your_key\"\nGROQ_MODEL = \"llama-3.3-70b-versatile\"", language="toml")


try:
    raw_df = pd.read_csv(uploaded_file) if uploaded_file is not None and not use_sample else load_sample_data()
    clean_df = preprocess_data(raw_df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

model, metrics, validation = build_model(clean_df)
forecast_chart = forecast_daily_sales(clean_df)
india_forecast = india_2026_forecast(clean_df)
sentiment_pct, keywords, issues = analyze_reviews(clean_df)
anomalies = detect_anomalies(clean_df)

region_sales = clean_df.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
category_sales = clean_df.groupby("product_category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
kpis = {
    "total_revenue": clean_df["revenue"].sum(),
    "total_units": clean_df["units_sold"].sum(),
    "best_region": region_sales.iloc[0]["region"],
    "best_region_revenue": region_sales.iloc[0]["revenue"],
    "best_category": category_sales.iloc[0]["product_category"],
}
summary, recommendations = business_summary(clean_df, kpis, sentiment_pct, issues, anomalies, india_forecast)

tab_dashboard, tab_prediction, tab_chatbot, tab_report = st.tabs(["Dashboard", "Prediction", "AI Chatbot", "Reports"])

with tab_dashboard:
    st.subheader("Live Dashboard")
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Revenue", inr(kpis["total_revenue"]), "Cleaned dataset revenue")
    with cols[1]:
        metric_card("Total Units", f"{kpis['total_units']:,.0f}", "Units sold")
    with cols[2]:
        metric_card("Best Region", kpis["best_region"], inr(kpis["best_region_revenue"]))
    with cols[3]:
        metric_card("Best Category", kpis["best_category"], "Top product category")

    st.markdown("### India 2026 Live Sales Forecast")
    india_cols = st.columns(4)
    with india_cols[0]:
        metric_card("Today's India Estimate", inr(india_forecast["today_estimate"]), f"As of {india_forecast['as_of']}")
    with india_cols[1]:
        metric_card("2026 YTD Actual", inr(india_forecast["ytd_actual"]), india_forecast["scope"])
    with india_cols[2]:
        metric_card("Remaining 2026", inr(india_forecast["remaining_forecast"]), f"Next 90 days: {inr(india_forecast['next_90'])}")
    with india_cols[3]:
        metric_card("2026 Projection", inr(india_forecast["full_year"]), f"{abs(india_forecast['growth']):.1f}% {india_forecast['direction']}")
    st.info(f"Confidence: {india_forecast['confidence']}. This forecast uses uploaded data, moving averages, damped trend, and weekday seasonality.")
    st.plotly_chart(
        go.Figure()
        .add_bar(x=india_forecast["monthly"]["month"], y=india_forecast["monthly"]["actual"], name="Actual")
        .add_bar(x=india_forecast["monthly"]["month"], y=india_forecast["monthly"]["forecast"], name="Forecast")
        .update_layout(barmode="stack", height=360, margin=dict(l=10, r=10, t=40, b=10)),
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(daily_revenue(clean_df), x="date", y="revenue", title="Revenue Trend"), width="stretch")
    with c2:
        st.plotly_chart(px.bar(region_sales, x="region", y="revenue", title="Region-wise Sales"), width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.bar(category_sales, x="product_category", y="revenue", title="Product Category Performance"), width="stretch")
    with c4:
        st.plotly_chart(px.line(forecast_chart, x="date", y=["revenue", "forecast"], title="30-Day Forecast"), width="stretch")

    st.markdown("### Anomaly Alerts")
    if len(anomalies):
        for _, row in anomalies.head(6).iterrows():
            st.markdown(f"<div class='alert-box'><b>{row['type']}</b> on {row['date'].date()}: {inr(row['revenue'])}</div>", unsafe_allow_html=True)
    else:
        st.success("No major anomaly detected.")

with tab_prediction:
    st.subheader("Manual Revenue Prediction")
    st.caption("Select a scenario and the trained Streamlit model will predict revenue.")
    c1, c2, c3 = st.columns(3)
    with c1:
        pred_date = st.date_input("Prediction date", value=date(2026, 5, 5))
        region = st.selectbox("Region", sorted(clean_df["region"].unique()))
    with c2:
        product = st.selectbox("Product category", sorted(clean_df["product_category"].unique()))
        units = st.number_input("Units sold", min_value=0.0, value=float(clean_df["units_sold"].median()))
    with c3:
        discount = st.slider("Discount %", 0.0, 90.0, float(clean_df["discount_pct"].median()))

    lag_7 = clean_df["revenue"].tail(7).mean()
    lag_30 = clean_df["revenue"].tail(30).mean()
    feature_row = pd.DataFrame(
        [
            {
                "units_sold": units,
                "discount_pct": discount,
                "month": pred_date.month,
                "year": pred_date.year,
                "day_of_week": pred_date.weekday(),
                "lag_7": lag_7,
                "lag_30": lag_30,
                "rolling_mean_7": lag_7,
                "region": region,
                "product_category": product,
            }
        ]
    )
    predicted_revenue = float(model.predict(feature_row)[0])
    st.metric("Predicted Revenue", inr(predicted_revenue))

    st.markdown("### Model Leaderboard")
    st.dataframe(pd.DataFrame([metrics]).assign(Model="RandomForest Regressor")[["Model", "RMSE", "MAE", "R2"]], width="stretch")
    st.plotly_chart(px.line(validation, x="date", y=["actual", "predicted"], title="Actual vs Predicted Validation"), width="stretch")

with tab_chatbot:
    st.subheader("AI Chatbot")
    st.caption("Uses Groq or Gemini from Streamlit secrets. If no key is configured, local business rules answer.")
    context = {
        "summary": summary,
        "best_region": kpis["best_region"],
        "best_region_revenue": inr(kpis["best_region_revenue"]),
        "india_full_year": inr(india_forecast["full_year"]),
        "india_confidence": india_forecast["confidence"],
    }
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    quick_questions = ["Which region has highest sales?", "Predict 2026 India sales", "Why did sales drop?", "Give business recommendations"]
    selected_question = st.selectbox("Try a quick question", [""] + quick_questions)
    question = st.chat_input("Ask about the uploaded dataset")
    question = question or selected_question

    if question:
        try:
            answer, source = ask_ai(question, context)
        except Exception as exc:
            answer, source = f"AI provider failed: {exc}. Local summary: {summary}", "Local fallback"
        st.session_state.chat_history.append(("user", question, ""))
        st.session_state.chat_history.append(("assistant", answer, source))

    for role, text, source in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)
            if source:
                st.caption(source)

with tab_report:
    st.subheader("Business Report Summary")
    st.write(summary)
    st.markdown("### Recommendations")
    for rec in recommendations:
        st.write(f"- {rec}")

    st.markdown("### Customer Sentiment")
    st.write(sentiment_pct)
    if keywords:
        st.write("Top keywords:", ", ".join([word for word, _count in keywords]))

    cleaned_csv = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned dataset CSV", cleaned_csv, "cleaned_sales_data.csv", "text/csv")

    prediction_output = pd.DataFrame(
        [
            {
                "date": pred_date,
                "region": region,
                "product_category": product,
                "units_sold": units,
                "discount_pct": discount,
                "predicted_revenue": predicted_revenue,
            }
        ]
    ).to_csv(index=False).encode("utf-8")
    st.download_button("Download latest prediction CSV", prediction_output, "prediction_results.csv", "text/csv")

    report_bytes = pdf_report(summary, recommendations, kpis, india_forecast, anomalies)
    st.download_button("Download PDF report", report_bytes, "sales_prediction_ai_report.pdf", "application/pdf")
