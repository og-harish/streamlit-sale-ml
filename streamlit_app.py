import io
import os
from datetime import date, datetime, timedelta

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

COLUMN_ALIASES = {
    "date": ["date", "order_date", "sale_date", "created_at", "timestamp"],
    "region": ["region", "state", "location", "city", "area"],
    "product_category": ["product_category", "category", "product", "item", "product_type"],
    "units_sold": ["units_sold", "quantity", "qty", "units", "sales_count"],
    "revenue": ["revenue", "sales", "amount", "total_price", "price"],
    "unit_price": ["unit_price", "selling_price", "price", "rate", "mrp"],
    "discount_pct": ["discount_pct", "discount", "discount_percent", "offer"],
    "customer_reviews": ["customer_reviews", "review", "feedback", "comment", "customer_text"],
}

CORE_FIELDS = ["date", "units_sold", "revenue"]
OPTIONAL_FIELDS = ["region", "product_category", "discount_pct", "customer_reviews", "unit_price"]

TAMIL_NADU_CITIES = [
    "Chennai",
    "Coimbatore",
    "Madurai",
    "Trichy",
    "Salem",
    "Tirunelveli",
    "Erode",
    "Vellore",
    "Thanjavur",
]

TAMIL_NADU_CATEGORIES = [
    "Electronics",
    "Fashion",
    "Grocery",
    "Home Appliances",
    "Beauty",
    "Mobiles",
    "Books",
]

TN_CITY_MULTIPLIERS = {
    "Chennai": 1.38,
    "Coimbatore": 1.14,
    "Madurai": 0.98,
    "Trichy": 0.88,
    "Salem": 0.82,
    "Tirunelveli": 0.72,
    "Erode": 0.78,
    "Vellore": 0.75,
    "Thanjavur": 0.68,
}

TN_CATEGORY_MULTIPLIERS = {
    "Electronics": 1.2,
    "Fashion": 1.08,
    "Grocery": 0.94,
    "Home Appliances": 1.02,
    "Beauty": 0.84,
    "Mobiles": 1.28,
    "Books": 0.64,
}

TN_DISCLAIMER = "This is a predictive simulation based on dataset trends, not official live government data."

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
    session_key = f"runtime_{name.lower()}"
    session_value = st.session_state.get(session_key)
    if session_value:
        return str(session_value).strip()

    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


def ai_provider_status():
    if get_secret("GROQ_API_KEY"):
        return "Groq", "Connected"
    if get_secret("GEMINI_API_KEY"):
        return "Gemini", "Connected"
    return "Local insight engine", "No API key configured"


@st.cache_data
def load_sample_data():
    return pd.read_csv("sample_data/sample_sales.csv")


def normalize_column_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def auto_detect_column_mapping(df):
    normalized = {normalize_column_name(col): col for col in df.columns}
    mapping = {}
    for field, aliases in COLUMN_ALIASES.items():
        match = next((normalized[normalize_column_name(alias)] for alias in aliases if normalize_column_name(alias) in normalized), None)
        mapping[field] = match
    return mapping


def mapping_completeness(mapping):
    has_revenue = bool(mapping.get("revenue"))
    has_revenue_formula = bool(mapping.get("units_sold") and mapping.get("unit_price"))
    warnings = []
    if not mapping.get("date"):
        warnings.append("No date column mapped; the app will create synthetic sequential dates.")
    if not mapping.get("region"):
        warnings.append("No region/city column mapped; the app will use 'Overall'.")
    if not mapping.get("product_category"):
        warnings.append("No product/category column mapped; the app will use 'General'.")
    if not mapping.get("units_sold"):
        warnings.append("No units/quantity column mapped; units will be estimated from available sales rows.")
    if not has_revenue and not has_revenue_formula:
        warnings.append("No revenue/sales amount or quantity x price mapping found; revenue will fall back to units/proxy values.")
    return warnings


def render_column_mapping(df, detected_mapping):
    st.markdown("#### Flexible Column Mapping")
    st.caption("Auto-detected columns are preselected. Change any dropdown if your dataset uses a different name.")
    options = ["Not available"] + list(df.columns)
    mapping = {}
    fields = CORE_FIELDS + OPTIONAL_FIELDS
    field_labels = {
        "date": "Date / order date",
        "units_sold": "Units / quantity",
        "revenue": "Revenue / sales amount",
        "region": "Region / city / state",
        "product_category": "Product / category",
        "discount_pct": "Discount %",
        "customer_reviews": "Review / feedback",
        "unit_price": "Unit price (used if revenue is missing)",
    }
    cols = st.columns(2)
    for index, field in enumerate(fields):
        default_col = detected_mapping.get(field)
        default_index = options.index(default_col) if default_col in options else 0
        with cols[index % 2]:
            selected = st.selectbox(field_labels[field], options, index=default_index, key=f"map_{field}")
            mapping[field] = None if selected == "Not available" else selected
    return mapping


def preprocess_data(df, mapping):
    source = df.drop_duplicates().copy()
    clean = pd.DataFrame(index=source.index)

    if mapping.get("date"):
        clean["date"] = pd.to_datetime(source[mapping["date"]], errors="coerce")
        if clean["date"].notna().any():
            first_valid = clean["date"].dropna().min()
            clean["date"] = clean["date"].fillna(first_valid)
        else:
            clean["date"] = pd.date_range("2026-01-01", periods=len(clean), freq="D")
    else:
        clean["date"] = pd.date_range("2026-01-01", periods=len(clean), freq="D")

    clean["region"] = (
        source[mapping["region"]].fillna("Overall").astype(str).str.strip()
        if mapping.get("region")
        else "Overall"
    )
    clean["product_category"] = (
        source[mapping["product_category"]].fillna("General").astype(str).str.strip()
        if mapping.get("product_category")
        else "General"
    )
    clean["customer_reviews"] = (
        source[mapping["customer_reviews"]].fillna("").astype(str)
        if mapping.get("customer_reviews")
        else "No review text available"
    )

    if mapping.get("units_sold"):
        clean["units_sold"] = pd.to_numeric(source[mapping["units_sold"]], errors="coerce")
    else:
        clean["units_sold"] = np.nan

    if mapping.get("revenue"):
        clean["revenue"] = pd.to_numeric(source[mapping["revenue"]], errors="coerce")
    else:
        clean["revenue"] = np.nan

    if clean["revenue"].isna().all() and mapping.get("unit_price"):
        unit_price = pd.to_numeric(source[mapping["unit_price"]], errors="coerce")
        units_for_formula = clean["units_sold"].fillna(clean["units_sold"].median() if clean["units_sold"].notna().any() else 1)
        clean["revenue"] = units_for_formula * unit_price

    if clean["units_sold"].isna().all():
        clean["units_sold"] = np.where(clean["revenue"].notna(), np.maximum(clean["revenue"] / 100, 1), 1)

    if clean["revenue"].isna().all():
        clean["revenue"] = np.maximum(clean["units_sold"], 1) * 100

    if mapping.get("discount_pct"):
        clean["discount_pct"] = pd.to_numeric(source[mapping["discount_pct"]], errors="coerce")
    else:
        clean["discount_pct"] = 0

    for col in ["units_sold", "revenue", "discount_pct"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        fallback = clean[col].median() if clean[col].notna().any() else 0
        clean[col] = clean[col].fillna(fallback)

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

    if len(df) < 4:
        model.fit(x, y)
        y_test = y
        predictions = model.predict(x)
        validation_dates = df["date"]
    else:
        test_size = 0.2 if len(df) >= 20 else 0.3
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        validation_dates = df.loc[y_test.index, "date"]

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions)) if len(y_test) > 1 else 0.0

    validation = pd.DataFrame(
        {
            "date": validation_dates,
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


def live_sales_prediction(df):
    daily = daily_revenue(df).sort_values("date")
    values = daily["revenue"].to_numpy(dtype=float)
    if len(values) == 0:
        return {}

    avg7 = np.mean(values[-7:]) if len(values) >= 7 else np.mean(values)
    avg14 = np.mean(values[-14:]) if len(values) >= 14 else avg7
    avg30 = np.mean(values[-30:]) if len(values) >= 30 else avg14
    moving_avg = avg7 * 0.6 + avg14 * 0.25 + avg30 * 0.15
    recent = values[-30:] if len(values) >= 2 else values
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0] if len(recent) > 1 else 0
    trend_pct = float(np.clip(slope / max(moving_avg, 1), -0.05, 0.05))
    volatility = float(np.std(recent) / max(avg30, 1))
    confidence = "High" if len(values) >= 60 and volatility < 0.25 else "Medium" if len(values) >= 20 and volatility < 0.55 else "Low"
    direction = "increase" if trend_pct >= 0 else "decrease"
    weekly_growth = trend_pct * 7 * 100
    global_avg = np.mean(values)

    weekday_factor = {}
    for weekday in range(7):
        weekday_values = daily[daily["date"].dt.dayofweek == weekday]["revenue"]
        weekday_factor[weekday] = float(np.clip(weekday_values.mean() / global_avg, 0.8, 1.25)) if len(weekday_values) else 1

    last_date = daily["date"].max()
    rows = []
    for day in range(0, 31):
        forecast_date = last_date + timedelta(days=day)
        factor = weekday_factor.get(forecast_date.dayofweek, 1)
        trend_factor = float(np.clip(1 + trend_pct * max(day, 1), 0.65, 1.55))
        rows.append(
            {
                "date": forecast_date,
                "forecast": max(0, moving_avg * factor * trend_factor),
                "period": "Estimate" if day == 0 else "Forecast",
            }
        )

    forecast_df = pd.DataFrame(rows)
    recent_actuals = daily.tail(30).copy()
    recent_actuals["forecast"] = np.nan
    forecast_df["revenue"] = np.nan
    chart_df = pd.concat([recent_actuals, forecast_df], ignore_index=True)
    today_estimate = float(forecast_df.iloc[0]["forecast"])
    tomorrow_forecast = float(forecast_df.iloc[1]["forecast"]) if len(forecast_df) > 1 else today_estimate
    next_7 = float(forecast_df.iloc[1:8]["forecast"].sum())
    next_30 = float(forecast_df.iloc[1:31]["forecast"].sum())

    return {
        "today_estimate": today_estimate,
        "tomorrow_forecast": tomorrow_forecast,
        "next_7": next_7,
        "next_30": next_30,
        "confidence": confidence,
        "direction": direction,
        "weekly_growth": weekly_growth,
        "insight": f"Sales are expected to {direction} by {abs(weekly_growth):.1f}% next week based on recent trend, moving average, and weekday seasonality.",
        "chart": chart_df,
    }


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


def demo_tamil_nadu_rows():
    rows = []
    for city_index, city in enumerate(TAMIL_NADU_CITIES):
        for category_index, category in enumerate(TAMIL_NADU_CATEGORIES):
            day = ((city_index + category_index) % 28) + 1
            city_factor = TN_CITY_MULTIPLIERS.get(city, 1)
            category_factor = TN_CATEGORY_MULTIPLIERS.get(category, 1)
            rows.append(
                {
                    "date": pd.Timestamp(2026, 5, day),
                    "region": city,
                    "product_category": category,
                    "units_sold": round(340 * city_factor * category_factor),
                    "revenue": 420000 * city_factor * category_factor * (0.92 + ((city_index + category_index) % 5) * 0.04),
                    "discount_pct": 8 + (category_index % 4) * 3,
                    "customer_reviews": "Tamil Nadu e-commerce demo trend",
                }
            )
    return pd.DataFrame(rows)


def tamil_nadu_live_prediction(df=None, now=None):
    now = now or datetime.now()
    simulated_now = datetime(2026, 5, min(now.day, 31), now.hour, now.minute, now.second)
    source_df = df.copy() if df is not None and len(df) else demo_tamil_nadu_rows()

    tn_pattern = "|".join(TAMIL_NADU_CITIES + ["Tamil Nadu", "TN"])
    tn_rows = source_df[source_df["region"].astype(str).str.contains(tn_pattern, case=False, na=False)]
    market_df = tn_rows if len(tn_rows) else source_df
    source_label = "Uploaded Tamil Nadu rows" if len(tn_rows) else "Uploaded dataset trend" if df is not None and len(df) else "Demo Tamil Nadu e-commerce baseline"

    daily = daily_revenue(market_df).sort_values("date")
    values = daily["revenue"].to_numpy(dtype=float)
    if len(values) == 0:
        market_df = demo_tamil_nadu_rows()
        daily = daily_revenue(market_df).sort_values("date")
        values = daily["revenue"].to_numpy(dtype=float)
        source_label = "Demo Tamil Nadu e-commerce baseline"

    avg7 = np.mean(values[-7:]) if len(values) >= 7 else np.mean(values)
    avg30 = np.mean(values[-30:]) if len(values) >= 30 else avg7
    moving_avg = avg7 * 0.68 + avg30 * 0.32
    recent = values[-30:] if len(values) >= 2 else values
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0] if len(recent) > 1 else 0
    trend_pct = float(np.clip(slope / max(moving_avg, 1), -0.035, 0.045))
    may_seasonality = 1.12
    weekday_values = daily[daily["date"].dt.dayofweek == simulated_now.weekday()]["revenue"]
    weekday_factor = float(np.clip(weekday_values.mean() / max(np.mean(values), 1), 0.82, 1.22)) if len(weekday_values) else 1
    hour_factor = 0.72 + 0.48 * np.sin(((simulated_now.hour - 8) / 24) * 2 * np.pi) ** 2
    category_strength = np.mean(list(TN_CATEGORY_MULTIPLIERS.values()))
    base_daily = moving_avg * may_seasonality * weekday_factor * category_strength
    elapsed_days = simulated_now.day - 1 + simulated_now.hour / 24 + simulated_now.minute / 1440 + simulated_now.second / 86400
    month_days = 31
    month_trend_factor = float(np.clip(1 + trend_pct * month_days, 0.72, 1.75))

    monthly_prediction = max(0, base_daily * month_days * month_trend_factor)
    today_prediction = max(0, base_daily * float(np.clip(1 + trend_pct * simulated_now.day, 0.75, 1.55)))
    hourly_prediction = today_prediction / 24 * hour_factor
    minute_prediction = hourly_prediction / 60
    second_prediction = minute_prediction / 60
    month_counter = monthly_prediction * min(elapsed_days / month_days, 1)
    growth_rate = ((monthly_prediction - max(avg30 * month_days, 1)) / max(avg30 * month_days, 1)) * 100
    volatility = float(np.std(recent) / max(avg30, 1))
    confidence_score = int(np.clip(88 - volatility * 45 + min(len(values), 90) * 0.18, 42, 94))

    city_total = sum(TN_CITY_MULTIPLIERS.values())
    city_sales = pd.DataFrame(
        [
            {"city": city, "predicted_sales": monthly_prediction * TN_CITY_MULTIPLIERS[city] / city_total}
            for city in TAMIL_NADU_CITIES
        ]
    ).sort_values("predicted_sales", ascending=False)
    category_total = sum(TN_CATEGORY_MULTIPLIERS.values())
    category_sales = pd.DataFrame(
        [
            {"category": category, "predicted_sales": monthly_prediction * TN_CATEGORY_MULTIPLIERS[category] / category_total}
            for category in TAMIL_NADU_CATEGORIES
        ]
    ).sort_values("predicted_sales", ascending=False)
    insight = (
        f"Tamil Nadu e-commerce sales are expected to {'grow' if growth_rate >= 0 else 'decline'} by "
        f"{abs(growth_rate):.1f}% in May 2026 based on uploaded dataset trend, moving average, seasonality, "
        f"and city/category demand patterns."
    )

    return {
        "source": source_label,
        "timestamp": simulated_now.strftime("%H:%M:%S"),
        "monthly_prediction": monthly_prediction,
        "today_prediction": today_prediction,
        "hourly_prediction": hourly_prediction,
        "minute_prediction": minute_prediction,
        "second_prediction": second_prediction,
        "month_counter": month_counter,
        "growth_rate": growth_rate,
        "confidence_score": confidence_score,
        "city_sales": city_sales,
        "category_sales": category_sales,
        "insight": insight,
        "disclaimer": TN_DISCLAIMER,
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


def business_summary(df, kpis, sentiment_pct, issues, anomalies, india_forecast, tn_forecast):
    weak_region = df.groupby("region")["revenue"].sum().idxmin()
    top_issue = next((issue for issue, count in issues if count > 0), "no repeated issue detected")
    anomaly_text = f"{len(anomalies)} anomaly alert(s) detected" if len(anomalies) else "No major anomaly detected"
    summary = (
        f"The dataset contains {len(df):,} cleaned rows with total revenue of {inr(kpis['total_revenue'])} "
        f"and {kpis['total_units']:,.0f} units sold. {kpis['best_region']} is the strongest region, "
        f"while {weak_region} needs attention. Customer sentiment is {sentiment_pct['positive']}% positive "
        f"and {sentiment_pct['negative']}% negative. {anomaly_text}. "
        f"India 2026 full-year sales are projected at {inr(india_forecast['full_year'])}. "
        f"Tamil Nadu May 2026 e-commerce sales are estimated at {inr(tn_forecast['monthly_prediction'])} "
        f"with {tn_forecast['confidence_score']}% confidence."
    )
    recommendations = [
        f"Protect momentum in {kpis['best_region']} by repeating the campaigns, inventory planning, and service practices working there.",
        f"Investigate {weak_region} before increasing spend; review discounts, delivery experience, and local product mix.",
        f"Fix {top_issue.lower()} first because repeated review themes usually explain hidden sales friction.",
        f"Use {tn_forecast['city_sales'].iloc[0]['city']} as the Tamil Nadu growth benchmark and compare weaker cities against its product mix.",
        "Compare discount percentage against revenue lift so promotions grow demand without eroding margin.",
        "Review anomaly alerts weekly and assign an owner for every sharp revenue drop.",
    ]
    return summary, recommendations


def compact_table(df, max_rows=10):
    if df is None or len(df) == 0:
        return "None"
    return df.head(max_rows).to_csv(index=False).strip()


def build_ai_context(df, kpis, sentiment_pct, keywords, issues, anomalies, india_forecast, live_forecast, tn_forecast, recommendations, metrics):
    daily = daily_revenue(df).sort_values("date")
    recent_daily = daily.tail(12)
    region_breakdown = (
        df.groupby("region", as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"), avg_discount=("discount_pct", "mean"))
        .sort_values("revenue", ascending=False)
    )
    category_breakdown = (
        df.groupby("product_category", as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"), avg_discount=("discount_pct", "mean"))
        .sort_values("revenue", ascending=False)
    )
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        .tail(12)
    )
    discount_corr = df["discount_pct"].corr(df["revenue"]) if df["discount_pct"].nunique() > 1 and df["revenue"].nunique() > 1 else 0
    top_keywords = ", ".join([f"{word} ({count})" for word, count in keywords]) or "None"
    issue_text = ", ".join([f"{issue}: {count}" for issue, count in issues if count > 0]) or "No repeated issues"
    anomaly_text = compact_table(anomalies[["date", "revenue", "pct_change", "type"]], 8) if len(anomalies) else "No major anomalies detected"
    tn_city_text = compact_table(tn_forecast["city_sales"], 9)
    tn_category_text = compact_table(tn_forecast["category_sales"], 7)

    return f"""
Dataset period: {df['date'].min().date()} to {df['date'].max().date()}
Rows after cleaning: {len(df):,}
Total revenue: {inr(kpis['total_revenue'])}
Total units sold: {kpis['total_units']:,.0f}
Best region: {kpis['best_region']} ({inr(kpis['best_region_revenue'])})
Best product category: {kpis['best_category']}
Model performance: RMSE {metrics['RMSE']:.2f}, MAE {metrics['MAE']:.2f}, R2 {metrics['R2']:.3f}
Live sales forecast: Today {inr(live_forecast['today_estimate'])}, Tomorrow {inr(live_forecast['tomorrow_forecast'])}, Next 7 days {inr(live_forecast['next_7'])}, Next 30 days {inr(live_forecast['next_30'])}, confidence {live_forecast['confidence']}, direction {live_forecast['direction']}, weekly growth {live_forecast['weekly_growth']:.1f}%.
India 2026 forecast: As of {india_forecast['as_of']}, full-year projection {inr(india_forecast['full_year'])}, remaining forecast {inr(india_forecast['remaining_forecast'])}, confidence {india_forecast['confidence']}.
Tamil Nadu live May 2026 prediction: Month {inr(tn_forecast['monthly_prediction'])}, today {inr(tn_forecast['today_prediction'])}, current hour {inr(tn_forecast['hourly_prediction'])}, current minute {inr(tn_forecast['minute_prediction'])}, current second {inr(tn_forecast['second_prediction'])}, live counter {inr(tn_forecast['month_counter'])}, growth {tn_forecast['growth_rate']:.1f}%, confidence score {tn_forecast['confidence_score']}%, source {tn_forecast['source']}.
Tamil Nadu disclaimer: {tn_forecast['disclaimer']}
Customer sentiment: positive {sentiment_pct['positive']}%, neutral {sentiment_pct['neutral']}%, negative {sentiment_pct['negative']}%.
Top review keywords: {top_keywords}
Detected customer issues: {issue_text}
Discount-to-revenue correlation: {discount_corr:.3f}
Anomalies:
{anomaly_text}

Region breakdown:
{compact_table(region_breakdown, 12)}

Product category breakdown:
{compact_table(category_breakdown, 12)}

Tamil Nadu city-wise predicted sales:
{tn_city_text}

Tamil Nadu category-wise predicted sales:
{tn_category_text}

Recent daily sales:
{compact_table(recent_daily, 12)}

Recent monthly sales:
{compact_table(monthly, 12)}

Current recommendations:
{" | ".join(recommendations)}
""".strip()


def local_chatbot_answer(question, df, kpis, sentiment_pct, issues, anomalies, india_forecast, live_forecast, tn_forecast, recommendations, metrics):
    lower = question.lower()
    region_sales = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    category_sales = df.groupby("product_category")["revenue"].sum().sort_values(ascending=False)
    daily = daily_revenue(df).sort_values("date")
    weak_region = region_sales.idxmin()
    weak_category = category_sales.idxmin()
    latest_revenue = float(daily.iloc[-1]["revenue"])
    previous_revenue = float(daily.iloc[-2]["revenue"]) if len(daily) > 1 else latest_revenue
    latest_change = ((latest_revenue - previous_revenue) / max(previous_revenue, 1)) * 100
    top_issue = next((issue for issue, count in issues if count > 0), "no repeated customer issue")
    top_tn_city = tn_forecast["city_sales"].iloc[0]

    if any(phrase in lower for phrase in ["live sales now", "sales now", "current second", "this hour", "current hour"]):
        return (
            f"Live Tamil Nadu simulated sales for May 2026 are currently {inr(tn_forecast['month_counter'])}. "
            f"This hour is estimated at {inr(tn_forecast['hourly_prediction'])}, this minute at {inr(tn_forecast['minute_prediction'])}, "
            f"and this second at {inr(tn_forecast['second_prediction'])}. {tn_forecast['disclaimer']}"
        )
    if any(phrase in lower for phrase in ["tamil nadu", "tamilnadu", "tn sales", "may 2026", "this month", "current month"]):
        if "city" in lower or "highest" in lower:
            return (
                f"{top_tn_city['city']} has the highest Tamil Nadu predicted sales at {inr(top_tn_city['predicted_sales'])}. "
                f"The top three cities are "
                f"{', '.join([f'{row.city} ({inr(row.predicted_sales)})' for row in tn_forecast['city_sales'].head(3).itertuples()])}."
            )
        return (
            f"May 2026 Tamil Nadu e-commerce sales are forecast at {inr(tn_forecast['monthly_prediction'])}. "
            f"Today is {inr(tn_forecast['today_prediction'])}, growth is {tn_forecast['growth_rate']:.1f}%, "
            f"and confidence score is {tn_forecast['confidence_score']}%. {tn_forecast['insight']}"
        )
    if "download" in lower or "pdf" in lower:
        return "Go to the Reports tab and click Download PDF report. It includes dataset summary, predictions, Tamil Nadu live forecast, city/category sales, recommendations, anomalies, and the predictive-estimate disclaimer."
    if any(word in lower for word in ["highest", "best", "top region", "region has"]):
        return f"{region_sales.index[0]} has the highest sales with {inr(region_sales.iloc[0])}. The weakest region is {weak_region} with {inr(region_sales.iloc[-1])}."
    if any(word in lower for word in ["product", "category", "item"]):
        return f"{category_sales.index[0]} is the top category with {inr(category_sales.iloc[0])}. {weak_category} is the weakest category with {inr(category_sales.iloc[-1])}."
    if any(word in lower for word in ["drop", "decrease", "down", "why"]):
        if len(anomalies):
            latest_anomaly = anomalies.iloc[0]
            return f"The clearest drop/spike signal is {latest_anomaly['type']} on {latest_anomaly['date'].date()} at {inr(latest_anomaly['revenue'])}. Check {weak_region}, {weak_category}, discount changes, and review issue '{top_issue}' first."
        return f"No major anomaly is currently flagged. The latest day changed by {latest_change:.1f}% versus the previous day; monitor {weak_region}, {weak_category}, and review issue '{top_issue}'."
    if any(word in lower for word in ["sentiment", "review", "customer", "complaint", "issue"]):
        return f"Customer sentiment is {sentiment_pct['positive']}% positive, {sentiment_pct['neutral']}% neutral, and {sentiment_pct['negative']}% negative. The most important repeated issue is {top_issue}."
    if any(word in lower for word in ["forecast", "predict", "prediction", "tomorrow", "next 7", "next 30", "future"]):
        return f"Today is estimated at {inr(live_forecast['today_estimate'])}, tomorrow at {inr(live_forecast['tomorrow_forecast'])}, next 7 days at {inr(live_forecast['next_7'])}, and next 30 days at {inr(live_forecast['next_30'])}. Confidence is {live_forecast['confidence']} and the trend points to a {live_forecast['direction']}."
    if any(word in lower for word in ["india", "2026"]):
        return f"India 2026 full-year sales are projected at {inr(india_forecast['full_year'])}, with {inr(india_forecast['remaining_forecast'])} remaining forecast and {india_forecast['confidence']} confidence."
    if any(word in lower for word in ["model", "accuracy", "rmse", "mae", "r2"]):
        return f"The RandomForest model reports RMSE {metrics['RMSE']:.2f}, MAE {metrics['MAE']:.2f}, and R2 {metrics['R2']:.3f}. Use this as a directional business estimate, not a guaranteed financial forecast."
    if any(word in lower for word in ["recommend", "improve", "action", "strategy"]):
        return " ".join(recommendations)
    if any(word in lower for word in ["summary", "report", "overall"]):
        return f"Total revenue is {inr(kpis['total_revenue'])} from {kpis['total_units']:,.0f} units. {kpis['best_region']} and {kpis['best_category']} lead performance. Live forecast confidence is {live_forecast['confidence']}."

    return f"I can answer from the uploaded dataset. Key snapshot: revenue {inr(kpis['total_revenue'])}, best region {kpis['best_region']}, best category {kpis['best_category']}, forecast direction {live_forecast['direction']}, and top issue {top_issue}."


def ask_ai(question, context_text):
    groq_key = get_secret("GROQ_API_KEY")
    gemini_key = get_secret("GEMINI_API_KEY")
    prompt = f"""
You are a practical senior business analyst inside a sales prediction dashboard.
Answer the user's question using only the dataset context below.
Be specific, numeric when possible, and business-friendly.
If the question asks for a cause, explain the most likely drivers from region/category trends, anomalies, discounts, and reviews.
If the context is insufficient, say what is missing and give the best available answer.

Dataset context:
{context_text}

User question:
{question}
""".strip()

    if groq_key:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": get_secret("GROQ_MODEL") or "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You answer sales analytics questions from the provided dashboard context. Never invent rows or secret keys."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.35,
                "max_tokens": 750,
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

    raise RuntimeError("No AI API key configured")


def pdf_report(summary, recommendations, kpis, india_forecast, live_forecast, tn_forecast, anomalies):
    pdf = FPDF()
    pdf.add_page()
    text_width = 180
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(text_width, 10, "Sales Prediction AI Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(text_width, 6, f"Total revenue: {inr(kpis['total_revenue'])}\nTotal units: {kpis['total_units']:,.0f}\nBest region: {kpis['best_region']}\nBest category: {kpis['best_category']}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "Live Sales Prediction", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(text_width, 6, f"Today estimate: {inr(live_forecast['today_estimate'])}\nTomorrow forecast: {inr(live_forecast['tomorrow_forecast'])}\nNext 7 days: {inr(live_forecast['next_7'])}\nNext 30 days: {inr(live_forecast['next_30'])}\nConfidence: {live_forecast['confidence']}\nInsight: {live_forecast['insight']}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "India 2026 Forecast", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(text_width, 6, f"As of: {india_forecast['as_of']}\nToday estimate: {inr(india_forecast['today_estimate'])}\nRemaining forecast: {inr(india_forecast['remaining_forecast'])}\nFull-year projection: {inr(india_forecast['full_year'])}\nConfidence: {india_forecast['confidence']}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "Tamil Nadu Live E-Commerce Sales Prediction - 2026", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    city_lines = "\n".join([f"{row.city}: {inr(row.predicted_sales)}" for row in tn_forecast["city_sales"].head(6).itertuples()])
    category_lines = "\n".join([f"{row.category}: {inr(row.predicted_sales)}" for row in tn_forecast["category_sales"].head(6).itertuples()])
    pdf.multi_cell(
        text_width,
        6,
        (
            f"Current month forecast: {inr(tn_forecast['monthly_prediction'])}\n"
            f"Today: {inr(tn_forecast['today_prediction'])}\n"
            f"Current hour: {inr(tn_forecast['hourly_prediction'])}\n"
            f"Current minute: {inr(tn_forecast['minute_prediction'])}\n"
            f"Current second estimate: {inr(tn_forecast['second_prediction'])}\n"
            f"Growth for May 2026: {tn_forecast['growth_rate']:.1f}%\n"
            f"Confidence score: {tn_forecast['confidence_score']}%\n"
            f"Insight: {tn_forecast['insight']}\n"
            f"City-wise predicted sales:\n{city_lines}\n"
            f"Category-wise predicted sales:\n{category_lines}\n"
            f"Disclaimer: {tn_forecast['disclaimer']}"
        ),
    )
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


@st.fragment(run_every="1s")
def render_tamil_nadu_live_dashboard(df):
    forecast = tamil_nadu_live_prediction(df)
    signature = f"{len(df)}-{round(float(df['revenue'].sum()), 2)}-{str(df['date'].max())}"
    if st.session_state.get("tn_live_signature") != signature:
        st.session_state["tn_live_signature"] = signature
        st.session_state["tn_live_points"] = []

    point = {
        "time": forecast["timestamp"],
        "live_counter": forecast["month_counter"],
        "second_estimate": forecast["second_prediction"],
    }
    st.session_state.setdefault("tn_live_points", []).append(point)
    st.session_state["tn_live_points"] = st.session_state["tn_live_points"][-45:]

    st.markdown("### Live Tamil Nadu E-Commerce Sales Prediction - 2026")
    st.warning(forecast["disclaimer"])
    top_line = st.columns([1.4, 1, 1])
    with top_line[0]:
        metric_card("Live Simulated May 2026 Counter", inr(forecast["month_counter"]), f"Updates every second at {forecast['timestamp']}")
    with top_line[1]:
        metric_card("May 2026 Growth", f"{forecast['growth_rate']:.1f}%", forecast["source"])
    with top_line[2]:
        metric_card("Confidence Score", f"{forecast['confidence_score']}%", "Moving average + trend + seasonality")

    live_cols = st.columns(5)
    live_metrics = [
        ("Current Month 2026 Sales", forecast["monthly_prediction"], "May full-month forecast"),
        ("Today's Sales", forecast["today_prediction"], "Clock-aware daily demand"),
        ("Current Hour Sales", forecast["hourly_prediction"], "Hourly run rate"),
        ("Current Minute Sales", forecast["minute_prediction"], "Minute-level estimate"),
        ("Current Second Estimated Sales", forecast["second_prediction"], "Second-level simulation"),
    ]
    for col, (label, value, detail) in zip(live_cols, live_metrics):
        with col:
            metric_card(label, inr(value), detail)

    st.info(forecast["insight"])
    chart_df = pd.DataFrame(st.session_state["tn_live_points"])
    st.plotly_chart(
        px.line(chart_df, x="time", y=["live_counter", "second_estimate"], title="Live Tamil Nadu Forecast Updating Every Second"),
        width="stretch",
    )
    city_col, category_col = st.columns(2)
    with city_col:
        st.plotly_chart(px.bar(forecast["city_sales"], x="city", y="predicted_sales", title="Tamil Nadu City-wise Predicted Sales"), width="stretch")
    with category_col:
        st.plotly_chart(px.bar(forecast["category_sales"], x="category", y="predicted_sales", title="Category-wise Predicted Sales"), width="stretch")


st.markdown(
    """
    <div class="hero">
        <h1>Sales Prediction System with NLP Insights</h1>
        <p>Upload almost any sales CSV, map columns flexibly, simulate Tamil Nadu 2026 e-commerce sales live, chat with AI, and export portfolio-ready reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Sales AI")
    uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"])
    use_sample = st.button("Load sample dataset", width="stretch")
    st.caption("Flexible upload: the app auto-detects date, region/city, category/product, quantity, revenue, discount, and review columns.")
    st.divider()
    st.caption("AI chatbot configuration")
    runtime_groq_key = st.text_input("Temporary Groq API key", type="password", help="Stored only in this Streamlit session. For production, use Streamlit Cloud secrets.")
    runtime_gemini_key = st.text_input("Temporary Gemini API key", type="password", help="Optional fallback provider for this session only.")
    if runtime_groq_key:
        st.session_state["runtime_groq_api_key"] = runtime_groq_key.strip()
    if runtime_gemini_key:
        st.session_state["runtime_gemini_api_key"] = runtime_gemini_key.strip()
    provider_name, provider_state = ai_provider_status()
    st.success(f"{provider_name}: {provider_state}" if provider_state == "Connected" else "Local fallback active")
    st.caption("Production secrets for Streamlit Cloud")
    st.code("GROQ_API_KEY = \"your_key\"\nGROQ_MODEL = \"llama-3.3-70b-versatile\"\nGEMINI_API_KEY = \"your_key\"", language="toml")


try:
    raw_df = pd.read_csv(uploaded_file) if uploaded_file is not None and not use_sample else load_sample_data()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

detected_mapping = auto_detect_column_mapping(raw_df)
mapping_warnings = mapping_completeness(detected_mapping)
with st.expander("Dataset preview and column mapping", expanded=bool(mapping_warnings)):
    st.markdown("#### Uploaded Dataset Preview")
    st.dataframe(raw_df.head(12), width="stretch")
    column_mapping = render_column_mapping(raw_df, detected_mapping)
    final_warnings = mapping_completeness(column_mapping)
    if final_warnings:
        for warning in final_warnings:
            st.warning(warning)
    else:
        st.success("Column mapping looks strong. The app can build full dashboard, prediction, chatbot, and report outputs.")

try:
    clean_df = preprocess_data(raw_df, column_mapping)
except Exception as exc:
    st.error(f"Could not preprocess the dataset: {exc}")
    st.stop()

if len(clean_df) < 2:
    st.error("The dataset needs at least 2 usable rows after cleaning.")
    st.stop()

with st.expander("Cleaned dataset preview", expanded=False):
    st.dataframe(clean_df[REQUIRED_COLUMNS + ["month", "year", "day_of_week", "lag_7", "lag_30", "rolling_mean_7"]].head(12), width="stretch")

model, metrics, validation = build_model(clean_df)
forecast_chart = forecast_daily_sales(clean_df)
live_forecast = live_sales_prediction(clean_df)
india_forecast = india_2026_forecast(clean_df)
tn_forecast = tamil_nadu_live_prediction(clean_df)
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
summary, recommendations = business_summary(clean_df, kpis, sentiment_pct, issues, anomalies, india_forecast, tn_forecast)
ai_context_text = build_ai_context(clean_df, kpis, sentiment_pct, keywords, issues, anomalies, india_forecast, live_forecast, tn_forecast, recommendations, metrics)

tab_dashboard, tab_prediction, tab_chatbot, tab_report = st.tabs(["Dashboard", "Prediction", "AI Chatbot", "Reports"])

with tab_dashboard:
    st.subheader("Live Dashboard")
    render_tamil_nadu_live_dashboard(clean_df)
    st.divider()

    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Revenue", inr(kpis["total_revenue"]), "Cleaned dataset revenue")
    with cols[1]:
        metric_card("Total Units", f"{kpis['total_units']:,.0f}", "Units sold")
    with cols[2]:
        metric_card("Best Region", kpis["best_region"], inr(kpis["best_region_revenue"]))
    with cols[3]:
        metric_card("Best Category", kpis["best_category"], "Top product category")

    st.markdown("### Live Sales Prediction")
    live_cols = st.columns(4)
    with live_cols[0]:
        metric_card("Today's Estimated Sales", inr(live_forecast["today_estimate"]), f"Confidence: {live_forecast['confidence']}")
    with live_cols[1]:
        metric_card("Tomorrow's Prediction", inr(live_forecast["tomorrow_forecast"]), f"Trend: {live_forecast['direction']}")
    with live_cols[2]:
        metric_card("Next 7 Days Forecast", inr(live_forecast["next_7"]), f"{abs(live_forecast['weekly_growth']):.1f}% {live_forecast['direction']}")
    with live_cols[3]:
        metric_card("Next 30 Days Forecast", inr(live_forecast["next_30"]), "Moving average + seasonality")
    st.info(live_forecast["insight"])
    st.plotly_chart(
        px.line(live_forecast["chart"], x="date", y=["revenue", "forecast"], title="Live Actual vs Forecast Sales"),
        width="stretch",
    )

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
    provider_name, provider_state = ai_provider_status()
    st.caption(f"Provider: {provider_name}. If the provider is unavailable, the local insight engine answers from the uploaded data.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_quick_question" not in st.session_state:
        st.session_state.last_quick_question = ""

    quick_questions = [
        "What are live sales now?",
        "Which Tamil Nadu city has highest sales?",
        "What is the sales prediction for this month?",
        "Summarize May 2026 Tamil Nadu sales",
        "Download the report",
        "Which region has highest sales?",
        "Predict next 30 days sales",
        "Why did sales drop?",
        "Summarize customer complaints",
        "Give business recommendations",
        "How accurate is the model?",
    ]
    selected_question = st.selectbox("Try a quick question", [""] + quick_questions, key="quick_question_select")
    typed_question = st.chat_input("Ask anything about the uploaded dataset")
    question = typed_question
    if selected_question and selected_question != st.session_state.last_quick_question:
        question = selected_question
        st.session_state.last_quick_question = selected_question

    if question:
        local_answer = local_chatbot_answer(
            question,
            clean_df,
            kpis,
            sentiment_pct,
            issues,
            anomalies,
            india_forecast,
            live_forecast,
            tn_forecast,
            recommendations,
            metrics,
        )
        try:
            answer, source = ask_ai(question, ai_context_text)
        except Exception as exc:
            answer, source = local_answer, f"Local insight engine ({exc})"
        st.session_state.chat_history.append(("user", question, ""))
        st.session_state.chat_history.append(("assistant", answer, source))

    for role, text, source in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)
            if source:
                st.caption(source)

    with st.expander("What the chatbot knows about this dataset"):
        st.code(ai_context_text[:6000], language="text")

with tab_report:
    st.subheader("Business Report Summary")
    st.write(summary)
    st.markdown("### Tamil Nadu Live Prediction Summary")
    st.write(tn_forecast["insight"])
    tn_report_cols = st.columns(3)
    with tn_report_cols[0]:
        metric_card("May 2026 Forecast", inr(tn_forecast["monthly_prediction"]), f"Growth: {tn_forecast['growth_rate']:.1f}%")
    with tn_report_cols[1]:
        metric_card("Top Tamil Nadu City", tn_forecast["city_sales"].iloc[0]["city"], inr(tn_forecast["city_sales"].iloc[0]["predicted_sales"]))
    with tn_report_cols[2]:
        metric_card("Top Category", tn_forecast["category_sales"].iloc[0]["category"], inr(tn_forecast["category_sales"].iloc[0]["predicted_sales"]))
    st.warning(tn_forecast["disclaimer"])
    st.markdown("### Live Forecast Summary")
    st.write(live_forecast["insight"])
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

    report_bytes = pdf_report(summary, recommendations, kpis, india_forecast, live_forecast, tn_forecast, anomalies)
    st.download_button("Download PDF report", report_bytes, "sales_prediction_ai_report.pdf", "application/pdf")
