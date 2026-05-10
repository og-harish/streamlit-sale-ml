import io
import json
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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None


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
    "date": ["date", "order_date", "sale_date", "created_at", "timestamp", "transaction_date", "invoice_date", "purchase_date"],
    "region": ["region", "state", "location", "city", "area", "district", "country", "market", "zone"],
    "product_category": ["product_category", "category", "product", "item", "product_type", "product_name", "sku", "department", "sub_category"],
    "units_sold": ["units_sold", "quantity", "qty", "units", "sales_count", "items_sold", "order_qty", "count"],
    "revenue": ["revenue", "sales", "amount", "total_price", "net_sales", "gross_sales", "order_value", "gmv", "subtotal"],
    "unit_price": ["unit_price", "selling_price", "price", "rate", "mrp", "item_price", "list_price"],
    "discount_pct": ["discount_pct", "discount", "discount_percent", "offer", "promo_discount", "coupon_discount"],
    "customer_reviews": ["customer_reviews", "review", "feedback", "comment", "customer_text", "customer_review", "remarks", "rating_text"],
}

CORE_FIELDS = ["date", "units_sold", "revenue"]
OPTIONAL_FIELDS = ["region", "product_category", "discount_pct", "customer_reviews", "unit_price"]

PAGES = [
    "Home",
    "Reports",
    "Analytics",
    "Assistant",
    "Leads",
    "Contacts",
    "Accounts",
    "Deals",
    "Forecasts",
    "Documents",
    "Campaigns",
    "Tasks",
    "Meetings",
    "Calls",
]

PAGE_GROUPS = {
    "General": ["Home", "Reports", "Analytics", "Assistant"],
    "Sales": ["Leads", "Contacts", "Accounts", "Deals", "Forecasts", "Documents", "Campaigns"],
    "Activities": ["Tasks", "Meetings", "Calls"],
}

NAV_LABELS = {
    "Home": "Home",
    "Reports": "Reports",
    "Analytics": "Analytics",
    "Assistant": "Assistant",
    "Leads": "Leads",
    "Contacts": "Contacts",
    "Accounts": "Accounts",
    "Deals": "Deals",
    "Forecasts": "Forecasts",
    "Documents": "Documents",
    "Campaigns": "Campaigns",
    "Tasks": "Tasks",
    "Meetings": "Meetings",
    "Calls": "Calls",
}
FORECAST_MANAGERS = [
    "Aarav Mehta",
    "Diya Raman",
    "Kabir Shah",
    "Meera Iyer",
    "Nisha Kapoor",
    "Rohan Nair",
]

REPO_URL = "https://github.com/og-harish/streamlit-sale-ml"
COLAB_NOTEBOOK_PATH = "notebooks/sales_forecast_preprocessing_colab.ipynb"
PPT_ASSET_PATH = "project_assets/Sales_Forecast_AI_Project_Deck.pptx"
COLAB_URL = f"https://colab.research.google.com/github/og-harish/streamlit-sale-ml/blob/main/{COLAB_NOTEBOOK_PATH}"
PPT_URL = f"{REPO_URL}/raw/main/{PPT_ASSET_PATH}"

TAMIL_NADU_CITIES = [
    "Chennai",
    "Coimbatore",
    "Madurai",
    "Tiruchirappalli",
    "Salem",
    "Tirunelveli",
    "Erode",
    "Vellore",
    "Thanjavur",
    "Tiruppur",
    "Kanchipuram",
    "Chengalpattu",
    "Tiruvallur",
    "Tiruvannamalai",
    "Cuddalore",
    "Villupuram",
    "Kallakurichi",
    "Dharmapuri",
    "Krishnagiri",
    "Namakkal",
    "Karur",
    "Dindigul",
    "Theni",
    "Sivaganga",
    "Ramanathapuram",
    "Virudhunagar",
    "Thoothukudi",
    "Tenkasi",
    "Kanyakumari",
    "Nagapattinam",
    "Mayiladuthurai",
    "Tiruvarur",
    "Pudukkottai",
    "Ariyalur",
    "Perambalur",
    "Ranipet",
    "Tirupathur",
    "Nilgiris",
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
    "Tiruchirappalli": 0.88,
    "Salem": 0.82,
    "Tirunelveli": 0.72,
    "Erode": 0.78,
    "Vellore": 0.75,
    "Thanjavur": 0.68,
    "Tiruppur": 0.9,
    "Kanchipuram": 0.84,
    "Chengalpattu": 0.91,
    "Tiruvallur": 0.86,
    "Tiruvannamalai": 0.62,
    "Cuddalore": 0.65,
    "Villupuram": 0.61,
    "Kallakurichi": 0.5,
    "Dharmapuri": 0.52,
    "Krishnagiri": 0.66,
    "Namakkal": 0.58,
    "Karur": 0.56,
    "Dindigul": 0.64,
    "Theni": 0.53,
    "Sivaganga": 0.49,
    "Ramanathapuram": 0.48,
    "Virudhunagar": 0.59,
    "Thoothukudi": 0.67,
    "Tenkasi": 0.51,
    "Kanyakumari": 0.63,
    "Nagapattinam": 0.45,
    "Mayiladuthurai": 0.43,
    "Tiruvarur": 0.44,
    "Pudukkottai": 0.5,
    "Ariyalur": 0.38,
    "Perambalur": 0.36,
    "Ranipet": 0.57,
    "Tirupathur": 0.46,
    "Nilgiris": 0.42,
}

TN_CITY_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Tirunelveli": (8.7139, 77.7567),
    "Erode": (11.3410, 77.7172),
    "Vellore": (12.9165, 79.1325),
    "Thanjavur": (10.7867, 79.1378),
    "Tiruppur": (11.1085, 77.3411),
    "Kanchipuram": (12.8342, 79.7036),
    "Chengalpattu": (12.6819, 79.9888),
    "Tiruvallur": (13.1439, 79.9089),
    "Tiruvannamalai": (12.2253, 79.0747),
    "Cuddalore": (11.7447, 79.7680),
    "Villupuram": (11.9401, 79.4861),
    "Kallakurichi": (11.7384, 78.9639),
    "Dharmapuri": (12.1277, 78.1579),
    "Krishnagiri": (12.5186, 78.2137),
    "Namakkal": (11.2194, 78.1678),
    "Karur": (10.9601, 78.0766),
    "Dindigul": (10.3673, 77.9803),
    "Theni": (10.0104, 77.4768),
    "Sivaganga": (9.8433, 78.4809),
    "Ramanathapuram": (9.3639, 78.8395),
    "Virudhunagar": (9.5851, 77.9579),
    "Thoothukudi": (8.7642, 78.1348),
    "Tenkasi": (8.9590, 77.3152),
    "Kanyakumari": (8.0883, 77.5385),
    "Nagapattinam": (10.7672, 79.8449),
    "Mayiladuthurai": (11.1018, 79.6529),
    "Tiruvarur": (10.7661, 79.6344),
    "Pudukkottai": (10.3797, 78.8208),
    "Ariyalur": (11.1401, 79.0786),
    "Perambalur": (11.2333, 78.8833),
    "Ranipet": (12.9249, 79.3330),
    "Tirupathur": (12.4950, 78.5678),
    "Nilgiris": (11.4102, 76.6950),
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
    "day_of_month",
    "quarter",
    "is_weekend",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_7",
]

CATEGORICAL_FEATURES = ["region", "product_category"]


st.set_page_config(
    page_title="Sales Forecast AI",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.3rem; }
    html, body, [class*="css"] {
        color: #0B1F1A;
    }
    p, span, label, small, strong, em, li,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stCaptionContainer"] {
        color: #0B1F1A;
        opacity: 1;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 4%, rgba(255, 180, 80, .28), transparent 30%),
            radial-gradient(circle at 100% 0%, rgba(31, 109, 91, .22), transparent 34%),
            linear-gradient(135deg, #FFFDF7 0%, #F7EFE2 55%, #EAF8F1 100%);
    }
    [data-testid="stMain"] p,
    [data-testid="stMain"] span,
    [data-testid="stMain"] label,
    [data-testid="stMain"] small,
    [data-testid="stMain"] li,
    [data-testid="stMain"] div[data-testid="stMarkdownContainer"],
    [data-testid="stMain"] div[data-testid="stCaptionContainer"] {
        color: #071B17 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09231F 0%, #103B37 58%, #184D43 100%);
    }
    [data-testid="stSidebar"] * {
        color: #FFFDF7 !important;
    }
    [data-testid="stSidebar"] code {
        color: #0B1F1A !important;
        background: #FFF6E8 !important;
        border: 1px solid rgba(255, 180, 80, .5);
    }
    [data-testid="stSidebar"] .stButton button {
        background: #FFB454;
        color: #0B1F1A !important;
        border: 0;
        font-weight: 900;
        box-shadow: 0 10px 28px rgba(255, 180, 80, .26);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
        opacity: 1 !important;
        font-weight: 760;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] * {
        color: #071B17 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: #103B37 !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, .6);
        font-weight: 900;
    }
    input, textarea,
    div[data-baseweb="input"] *,
    div[data-baseweb="textarea"] *,
    div[data-baseweb="select"] *,
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"] * {
        color: #071B17 !important;
        -webkit-text-fill-color: #071B17 !important;
        opacity: 1 !important;
    }
    input::placeholder,
    textarea::placeholder {
        color: #3D5B55 !important;
        opacity: 1 !important;
    }
    .stButton button,
    .stDownloadButton button,
    button[kind] {
        background: #103B37 !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(7, 27, 23, .22) !important;
        font-weight: 900 !important;
    }
    .stButton button *,
    .stDownloadButton button *,
    button[kind] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stDownloadButton button {
        background: #FFB454 !important;
        color: #071B17 !important;
    }
    [data-testid="stSidebar"] .stButton button *,
    [data-testid="stSidebar"] .stDownloadButton button * {
        color: #071B17 !important;
        -webkit-text-fill-color: #071B17 !important;
    }
    .hero {
        padding: 2.35rem;
        border-radius: 34px;
        background:
            linear-gradient(135deg, rgba(9, 35, 31, .98) 0%, rgba(16, 59, 55, .96) 55%, rgba(216, 96, 32, .96) 100%),
            radial-gradient(circle at 15% 0%, rgba(255, 180, 80, .42), transparent 34%);
        color: #FFFFFF;
        box-shadow: 0 28px 90px rgba(9, 35, 31, .36);
        border: 1px solid rgba(255, 255, 255, .22);
    }
    .hero h1 {
        color: #FFFFFF;
        font-size: clamp(2.4rem, 5vw, 4.4rem);
        line-height: .96;
        margin-bottom: .75rem;
        text-shadow: 0 4px 26px rgba(0, 0, 0, .32);
    }
    .hero p {
        color: #FFFDF7;
        font-size: 1.18rem;
        line-height: 1.72;
        font-weight: 700;
        max-width: 980px;
    }
    .hero, .hero * {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }
    .metric-card {
        border: 1px solid rgba(9, 35, 31, .16);
        background: rgba(255,255,255,.98);
        padding: 1.1rem 1.2rem;
        border-radius: 24px;
        box-shadow: 0 18px 48px rgba(9, 35, 31, .14);
    }
    .metric-label { color: #0F5A4F; font-size: .76rem; letter-spacing: .13em; text-transform: uppercase; font-weight: 900; }
    .metric-value { color: #071B17; font-size: 1.95rem; font-weight: 950; margin-top: .35rem; }
    .metric-detail { color: #2A4842; font-size: .93rem; font-weight: 700; margin-top: .25rem; }
    .metric-card, .metric-card * {
        opacity: 1 !important;
    }
    .metric-card .metric-label { color: #0F5A4F !important; }
    .metric-card .metric-value { color: #071B17 !important; }
    .metric-card .metric-detail { color: #2A4842 !important; }
    .alert-box {
        border-radius: 18px;
        padding: .9rem 1rem;
        background: #FFF1DD;
        color: #301405;
        border: 1px solid rgba(216, 96, 32, .45);
        font-weight: 750;
    }
    .upload-dropzone {
        border: 2px dashed rgba(255, 180, 80, .85);
        border-radius: 26px;
        padding: 1.15rem 1.25rem;
        background:
            radial-gradient(circle at top left, rgba(255, 180, 80, .26), transparent 32%),
            linear-gradient(135deg, rgba(255, 255, 255, .18), rgba(255, 255, 255, .08));
        color: #FFFDF7;
        font-weight: 800;
    }
    .upload-dropzone b { color: #FFFFFF; font-size: 1.05rem; }
    .stFileUploader section {
        border-radius: 22px;
        border: 2px dashed rgba(255, 180, 80, .72);
        background: rgba(255, 253, 247, .96);
        color: #0B1F1A;
    }
    .stFileUploader section * {
        color: #071B17 !important;
        opacity: 1 !important;
        font-weight: 750;
    }
    div[data-testid="stAlert"] {
        border-radius: 18px;
        border: 1px solid rgba(9, 35, 31, .18);
        box-shadow: 0 12px 32px rgba(9, 35, 31, .09);
    }
    div[data-testid="stAlert"] * {
        color: #071B17 !important;
        opacity: 1 !important;
        font-weight: 750;
    }
    .stDataFrame, .stPlotlyChart {
        background: rgba(255, 255, 255, .78);
        border-radius: 20px;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] div {
        color: #071B17 !important;
        opacity: 1 !important;
    }
    [data-testid="stMetric"] *,
    [data-testid="stChatMessage"] *,
    [data-testid="stChatInput"] *,
    [data-testid="stRadio"] *,
    [data-testid="stSelectbox"] *,
    [data-testid="stNumberInput"] *,
    [data-testid="stDateInput"] *,
    [data-testid="stSlider"] * {
        opacity: 1 !important;
    }
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, .92);
        border: 1px solid rgba(9, 35, 31, .14);
        border-radius: 18px;
    }
    [data-testid="stChatMessage"] * {
        color: #071B17 !important;
    }
    h1, h2, h3, h4 {
        color: #071B17;
    }
    .floating-chat-button {
        position: fixed;
        right: 26px;
        bottom: 26px;
        z-index: 999999;
        padding: 1rem 1.3rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #FFB454 0%, #F97316 45%, #103B37 100%);
        color: #FFFFFF !important;
        text-decoration: none !important;
        font-weight: 900;
        letter-spacing: .03em;
        box-shadow: 0 18px 50px rgba(249, 115, 22, .38);
        border: 2px solid rgba(255, 255, 255, .82);
    }
    .floating-chat-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 22px 58px rgba(16, 59, 55, .42);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    :root {
        --crm-sidebar: #243A63;
        --crm-sidebar-deep: #1F3359;
        --crm-active: #3C5180;
        --crm-border: #D7DFEC;
        --crm-canvas: #F4F7FC;
        --crm-text: #091B33;
        --crm-muted: #5F6F86;
        --crm-blue: #315BE8;
    }
    .main .block-container {
        padding-top: 0;
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--crm-canvas);
    }
    [data-testid="stSidebar"] {
        background: var(--crm-sidebar);
        border-right: 1px solid rgba(15, 23, 42, .18);
        min-width: 300px;
    }
    [data-testid="stSidebar"] * {
        color: #E8EEF9 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 8px;
        padding: .42rem .55rem;
        margin-bottom: .12rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] [aria-checked="true"] {
        background: var(--crm-active) !important;
        color: #FFFFFF !important;
    }
    .hero {
        padding: 1.25rem 1.35rem;
        border-radius: 0;
        background: #FFFFFF;
        border: 1px solid var(--crm-border);
        box-shadow: 0 6px 18px rgba(15, 23, 42, .08);
    }
    .hero h1 {
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        line-height: 1.08;
        margin-bottom: .4rem;
        text-shadow: none;
        letter-spacing: 0;
        color: var(--crm-text) !important;
    }
    .hero p {
        font-size: 1rem;
        line-height: 1.55;
        max-width: 1040px;
        font-weight: 600;
        color: var(--crm-muted) !important;
    }
    .metric-card, .forecast-card, .forecast-panel {
        border: 1px solid var(--crm-border);
        background: #FFFFFF;
        border-radius: 6px;
        box-shadow: none;
    }
    .metric-card {
        padding: 1rem 1.05rem;
    }
    .metric-label {
        color: var(--crm-muted) !important;
        font-size: .75rem;
        letter-spacing: 0;
        text-transform: uppercase;
        font-weight: 800;
    }
    .metric-value {
        color: var(--crm-text) !important;
        font-size: 1.72rem;
        font-weight: 850;
        margin-top: .25rem;
    }
    .metric-detail {
        color: var(--crm-muted) !important;
        font-size: .9rem;
        font-weight: 600;
        margin-top: .22rem;
    }
    .forecast-panel {
        padding: 1.15rem;
        margin: .6rem 0 1rem;
    }
    .forecast-card {
        padding: 1rem;
        height: 100%;
    }
    .forecast-label {
        color: #64748B !important;
        font-size: .76rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .forecast-value {
        color: var(--crm-text) !important;
        font-size: 1.45rem;
        font-weight: 850;
        margin-top: .25rem;
    }
    .forecast-detail {
        color: #475569 !important;
        font-size: .88rem;
        margin-top: .25rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: .35rem;
        padding: .28rem .6rem;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 800;
        background: #EFF6FF;
        color: #1D4ED8 !important;
        border: 1px solid rgba(37, 99, 235, .18);
    }
    .crm-alert {
        background: #E8F8EF;
        color: #0B1F1A;
        border-bottom: 1px solid #C7E9D5;
        text-align: center;
        padding: .55rem 1rem;
        font-size: .92rem;
    }
    .crm-toolbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        background: #FFFFFF;
        border: 1px solid var(--crm-border);
        box-shadow: 0 6px 18px rgba(15, 23, 42, .08);
        padding: .8rem .95rem;
        margin-bottom: 1rem;
    }
    .crm-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--crm-border);
        border-radius: 8px;
        padding: .48rem .8rem;
        margin-left: .35rem;
        color: var(--crm-text) !important;
        background: #F8FAFC;
        font-weight: 700;
        text-decoration: none !important;
    }
    .crm-button.primary {
        background: var(--crm-blue);
        border-color: var(--crm-blue);
        color: #FFFFFF !important;
    }
    .crm-panel {
        background: #FFFFFF;
        border: 1px solid var(--crm-border);
        border-radius: 6px;
        padding: 1rem;
        height: 100%;
    }
    .crm-section-title {
        color: var(--crm-text);
        font-size: 1rem;
        font-weight: 850;
        margin: 0 0 .8rem;
        text-transform: uppercase;
    }
    .teamspace {
        background: var(--crm-sidebar-deep);
        padding: .85rem;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,.14);
        margin: .8rem 0;
    }
    .teamspace-badge {
        display: inline-flex;
        width: 28px;
        height: 28px;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        background: #00C9A7;
        color: white !important;
        font-weight: 900;
        margin-right: .45rem;
    }
    .sidebar-section {
        color: #AFC0DD !important;
        font-size: .78rem;
        font-weight: 900;
        text-transform: uppercase;
        margin: .8rem 0 .35rem;
    }
    .upload-dropzone {
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, .35);
        background: rgba(255, 255, 255, .08);
        color: #E2E8F0;
        padding: 1rem;
    }
    .floating-chat-button {
        background: #2563EB;
        border-radius: 999px;
        box-shadow: 0 14px 34px rgba(37, 99, 235, .28);
        border: 1px solid rgba(255, 255, 255, .8);
        letter-spacing: 0;
    }
    h1, h2, h3, h4 {
        letter-spacing: 0;
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


def google_auth_configured():
    try:
        auth = st.secrets.get("auth") or {}
        provider = auth.get("google") or {}
        has_shared = bool(auth.get("redirect_uri") and auth.get("cookie_secret"))
        has_provider = bool(
            (auth.get("client_id") and auth.get("client_secret") and auth.get("server_metadata_url"))
            or (provider.get("client_id") and provider.get("client_secret") and provider.get("server_metadata_url"))
        )
        return has_shared and has_provider
    except Exception:
        return False


def render_google_auth_gate():
    configured = google_auth_configured()
    user = getattr(st, "user", None)
    logged_in = bool(getattr(user, "is_logged_in", False))

    if logged_in:
        return {
            "name": getattr(user, "name", None) or getattr(user, "email", None) or "Google user",
            "email": getattr(user, "email", None) or "",
            "picture": getattr(user, "picture", None) or "",
            "source": "Google",
        }

    if configured:
        st.markdown(
            """
            <div class="hero">
                <h1>Sales Forecast AI</h1>
                <p>Sign in with Google to open the CRM forecasting workspace.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_a, col_b, col_c = st.columns([1, 1.2, 1])
        with col_b:
            st.info("Google sign-in is required for the launched website.")
            if st.button("Sign in with Google", type="primary", width="stretch"):
                st.login("google")
        st.stop()

    if os.getenv("STREAMLIT_ENV") == "production":
        st.error("Google authentication is not configured. Add Streamlit auth secrets before launching.")
        st.stop()

    return {
        "name": "Demo Analyst",
        "email": "local-demo@salesforecast.ai",
        "picture": "",
        "source": "Local demo",
    }


def render_top_shell(user_info, current_page):
    st.markdown(
        f"""
        <div class="crm-toolbar">
            <div>
                <strong style="color:#091B33;">{current_page}</strong>
                <span style="color:#5F6F86;margin-left:.75rem;font-weight:700;">Sales Forecast Workspace</span>
            </div>
            <div>
                <span style="color:#5F6F86;font-weight:700;">{user_info['name']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ai_provider_status():
    if get_secret("GROQ_API_KEY"):
        return "Groq", "Connected"
    if get_secret("GEMINI_API_KEY"):
        return "Gemini", "Connected"
    return "Local insight engine", "No API key configured"


@st.cache_data
def load_sample_data():
    return pd.read_csv("sample_data/sample_sales.csv")


def read_uploaded_dataset(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Upload a CSV file before processing.")

    file_name = uploaded_file.name.lower()
    extension = file_name.rsplit(".", 1)[-1] if "." in file_name else "csv"
    if extension != "csv":
        raise ValueError("Only CSV files are supported in this upload workflow.")

    errors = []
    for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        uploaded_file.seek(0)
        try:
            frame = pd.read_csv(uploaded_file, encoding=encoding, engine="python", on_bad_lines="skip")
            frame.columns = [str(col).strip() for col in frame.columns]
            frame = frame.dropna(axis=1, how="all")
            if frame.empty:
                raise ValueError("The CSV file has no usable rows.")
            return frame, uploaded_file.name
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise ValueError(f"Could not parse {uploaded_file.name}. Tried common CSV encodings. Details: {' | '.join(errors[:2])}")


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
    clean["day_of_month"] = clean["date"].dt.day
    clean["quarter"] = clean["date"].dt.quarter
    clean["is_weekend"] = clean["day_of_week"].isin([5, 6]).astype(int)
    clean["lag_7"] = clean["revenue"].shift(7)
    clean["lag_30"] = clean["revenue"].shift(30)
    clean["rolling_mean_7"] = clean["revenue"].rolling(7, min_periods=1).mean().shift(1)
    clean["rolling_mean_30"] = clean["revenue"].rolling(30, min_periods=1).mean().shift(1)
    clean["rolling_std_7"] = clean["revenue"].rolling(7, min_periods=2).std().shift(1)

    for col in ["lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30"]:
        clean[col] = clean[col].fillna(clean["revenue"].expanding().mean())
        clean[col] = clean[col].fillna(clean["revenue"].mean())
    clean["rolling_std_7"] = clean["rolling_std_7"].fillna(clean["rolling_std_7"].median() if clean["rolling_std_7"].notna().any() else 0)

    return clean.reset_index(drop=True)


def build_processing_summary(raw_df, clean_df, column_mapping):
    date_column = column_mapping.get("date")
    raw_dates = pd.to_datetime(raw_df[date_column], errors="coerce") if date_column in raw_df.columns else pd.Series(dtype="object")
    summary = {
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(clean_df)),
        "duplicates_removed": max(0, int(len(raw_df) - len(raw_df.drop_duplicates()))),
        "missing_before": int(raw_df.isna().sum().sum()),
        "missing_after": int(clean_df.isna().sum().sum()),
        "mapped_fields": int(sum(bool(value) for value in column_mapping.values())),
        "engineered_features": int(len([col for col in clean_df.columns if col not in raw_df.columns])),
        "date_start": clean_df["date"].min().date().isoformat() if len(clean_df) else "-",
        "date_end": clean_df["date"].max().date().isoformat() if len(clean_df) else "-",
        "raw_date_coverage": int(raw_dates.notna().sum()) if len(raw_dates) else 0,
    }
    return summary


def regression_pipeline(regressor):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", regressor)])


def regression_candidates():
    candidates = {
        "RandomForest Quantum": RandomForestRegressor(
            n_estimators=260,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting Trend Learner": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.055,
            max_depth=3,
            random_state=42,
        ),
    }
    if XGBRegressor is not None:
        candidates["XGBoost Revenue Forecaster"] = XGBRegressor(
            n_estimators=260,
            learning_rate=0.055,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    return candidates


def score_predictions(y_true, predictions):
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    mae = float(mean_absolute_error(y_true, predictions))
    r2 = float(r2_score(y_true, predictions)) if len(y_true) > 1 else 0.0
    wape = float(np.sum(np.abs(y_true - predictions)) / max(np.sum(np.abs(y_true)), 1) * 100)
    accuracy = float(np.clip(100 - wape, 0, 100))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "WAPE": wape, "Accuracy": accuracy}


def predict_revenue_scenario(model, clean_df, pred_date, region, product, units, discount):
    lag_7 = clean_df["revenue"].tail(7).mean()
    lag_30 = clean_df["revenue"].tail(30).mean()
    rolling_std_7 = clean_df["revenue"].tail(7).std() if len(clean_df) > 1 else 0
    feature_row = pd.DataFrame(
        [
            {
                "units_sold": units,
                "discount_pct": discount,
                "month": pred_date.month,
                "year": pred_date.year,
                "day_of_week": pred_date.weekday(),
                "day_of_month": pred_date.day,
                "quarter": (pred_date.month - 1) // 3 + 1,
                "is_weekend": int(pred_date.weekday() in [5, 6]),
                "lag_7": lag_7,
                "lag_30": lag_30,
                "rolling_mean_7": lag_7,
                "rolling_mean_30": lag_30,
                "rolling_std_7": rolling_std_7 if np.isfinite(rolling_std_7) else 0,
                "region": region,
                "product_category": product,
            }
        ]
    )
    return float(model.predict(feature_row)[0]), feature_row


def build_model(df):
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = df[features]
    y = df["revenue"]

    if len(df) < 4:
        x_train, y_train = x, y
        x_eval, y_eval = x, y
        validation_dates = df["date"]
    else:
        test_size = 0.2 if len(df) >= 20 else 0.3
        x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=test_size, shuffle=False)
        validation_dates = df.loc[y_eval.index, "date"]

    leaderboard_rows = []
    trained_models = {}
    predictions_by_model = {}

    for name, estimator in regression_candidates().items():
        model = regression_pipeline(estimator)
        model.fit(x_train, y_train)
        predictions = model.predict(x_eval)
        scores = score_predictions(y_eval.values, predictions)
        leaderboard_rows.append({"Model": name, **scores})
        trained_models[name] = model
        predictions_by_model[name] = predictions

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(["RMSE", "MAE"], ascending=True).reset_index(drop=True)
    best_name = leaderboard.iloc[0]["Model"]
    best_predictions = predictions_by_model[best_name]

    final_model = regression_pipeline(regression_candidates()[best_name])
    final_model.fit(x, y)
    metrics = leaderboard.iloc[0].to_dict()
    metrics["Best Model"] = best_name
    metrics["Validation Rows"] = int(len(y_eval))

    validation = pd.DataFrame(
        {
            "date": validation_dates,
            "actual": y_eval.values,
            "predicted": best_predictions,
        }
    )

    return final_model, metrics, validation, leaderboard


def daily_revenue(df):
    return df.groupby("date", as_index=False).agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))


def forecast_daily_sales(df, periods=30):
    daily = daily_revenue(df).sort_values("date")
    values = daily["revenue"].to_numpy(dtype=float)
    if len(values) == 0:
        return pd.DataFrame()

    if Prophet is not None and len(daily) >= 10:
        try:
            prophet_df = daily[["date", "revenue"]].rename(columns={"date": "ds", "revenue": "y"})
            prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            prophet_model.fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=periods, freq="D")
            prophet_forecast = prophet_model.predict(future).tail(periods)
            forecast_chart = prophet_forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "forecast"})
            forecast_chart["forecast"] = forecast_chart["forecast"].clip(lower=0)
            forecast_chart["revenue"] = np.nan
            actual_chart = daily.tail(45).copy()
            actual_chart["forecast"] = np.nan
            return pd.concat([actual_chart, forecast_chart], ignore_index=True)
        except Exception:
            pass

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

    tn_pattern = "|".join(TAMIL_NADU_CITIES + ["Trichy", "Tamil Nadu", "TN"])
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
            {
                "city": city,
                "predicted_sales": monthly_prediction * TN_CITY_MULTIPLIERS[city] / city_total,
                "lat": TN_CITY_COORDS.get(city, (11.1271, 78.6569))[0],
                "lon": TN_CITY_COORDS.get(city, (11.1271, 78.6569))[1],
                "share_pct": TN_CITY_MULTIPLIERS[city] / city_total * 100,
                "second_velocity": second_prediction * TN_CITY_MULTIPLIERS[city] / city_total,
            }
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
        f"Use {tn_forecast['city_sales'].iloc[0]['city']} as the Tamil Nadu district benchmark and compare weaker markets against its product mix.",
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
Model performance: Best model {metrics['Best Model']}, RMSE {metrics['RMSE']:.2f}, MAE {metrics['MAE']:.2f}, R2 {metrics['R2']:.3f}, WAPE {metrics['WAPE']:.1f}%, accuracy score {metrics['Accuracy']:.1f}%.
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

Tamil Nadu district-wise predicted sales:
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


def local_chatbot_answer(question, df, kpis, sentiment_pct, issues, anomalies, india_forecast, live_forecast, tn_forecast, recommendations, metrics, crm_forecast):
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

    if any(word in lower for word in ["quota", "commit", "committed", "best case", "pipeline coverage", "forecast health", "attainment"]):
        categories = crm_forecast["categories"].set_index("category")["amount"].to_dict()
        risky = crm_forecast["territories"][crm_forecast["territories"]["status"] == "At risk"]
        risky_text = ", ".join(risky["region"].head(3).tolist()) if len(risky) else "no at-risk territory"
        return (
            f"For {crm_forecast['period_label']}, quota is {inr(crm_forecast['quota'])} and weighted forecast is "
            f"{inr(crm_forecast['weighted_forecast'])}, which is {crm_forecast['attainment']:.1f}% attainment. "
            f"Committed revenue is {inr(categories.get('Committed', 0))}, best case is {inr(categories.get('Best Case', 0))}, "
            f"and pipeline is {inr(categories.get('Pipeline', 0))}. Pipeline coverage is {crm_forecast['pipeline_coverage']:.1f}x, "
            f"confidence is {crm_forecast['confidence']}%, and the main territory risk is {risky_text}."
        )

    if any(phrase in lower for phrase in ["live sales now", "sales now", "current second", "this hour", "current hour"]):
        return (
            f"Live Tamil Nadu simulated sales for May 2026 are currently {inr(tn_forecast['month_counter'])}. "
            f"This hour is estimated at {inr(tn_forecast['hourly_prediction'])}, this minute at {inr(tn_forecast['minute_prediction'])}, "
            f"and this second at {inr(tn_forecast['second_prediction'])}. {tn_forecast['disclaimer']}"
        )
    if any(phrase in lower for phrase in ["tamil nadu", "tamilnadu", "tn sales", "may 2026", "this month", "current month"]):
        if "city" in lower or "district" in lower or "highest" in lower:
            return (
                f"{top_tn_city['city']} has the highest Tamil Nadu predicted sales at {inr(top_tn_city['predicted_sales'])}. "
                f"The top three districts/markets are "
                f"{', '.join([f'{row.city} ({inr(row.predicted_sales)})' for row in tn_forecast['city_sales'].head(3).itertuples()])}."
            )
        return (
            f"May 2026 Tamil Nadu e-commerce sales are forecast at {inr(tn_forecast['monthly_prediction'])}. "
            f"Today is {inr(tn_forecast['today_prediction'])}, growth is {tn_forecast['growth_rate']:.1f}%, "
            f"and confidence score is {tn_forecast['confidence_score']}%. {tn_forecast['insight']}"
        )
    if "download" in lower or "pdf" in lower:
        return "Go to the Reports page and click Download PDF report. It includes dataset summary, predictions, Tamil Nadu live forecast, district/category sales, recommendations, anomalies, and the predictive-estimate disclaimer."
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
        return f"The selected model is {metrics['Best Model']} with RMSE {metrics['RMSE']:.2f}, MAE {metrics['MAE']:.2f}, R2 {metrics['R2']:.3f}, WAPE {metrics['WAPE']:.1f}%, and an accuracy score of {metrics['Accuracy']:.1f}%. The platform compares multiple models and chooses the strongest time-aware validation result."
    if any(word in lower for word in ["recommend", "improve", "action", "strategy"]):
        return " ".join(recommendations)
    if any(word in lower for word in ["summary", "report", "overall"]):
        return f"Total revenue is {inr(kpis['total_revenue'])} from {kpis['total_units']:,.0f} units. {kpis['best_region']} and {kpis['best_category']} lead performance. Live forecast confidence is {live_forecast['confidence']}."

    return f"I can answer from the uploaded dataset. Key snapshot: revenue {inr(kpis['total_revenue'])}, best region {kpis['best_region']}, best category {kpis['best_category']}, forecast direction {live_forecast['direction']}, and top issue {top_issue}."


def ask_ai(question, context_text):
    groq_key = get_secret("GROQ_API_KEY")
    gemini_key = get_secret("GEMINI_API_KEY")
    prompt = f"""
You are a practical senior revenue operations analyst inside a professional sales forecasting workspace.
Answer the user's question using only the dataset context below.
Be specific, numeric when possible, and business-friendly.
If the question asks for forecast health, explain quota attainment, committed revenue, best case, pipeline coverage, territory risk, anomalies, discounts, and reviews.
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
                    {"role": "system", "content": "You answer sales forecasting and revenue operations questions from the provided dashboard context. Never invent rows, customers, or secret keys."},
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


def pdf_report(summary, recommendations, kpis, metrics, india_forecast, live_forecast, tn_forecast, crm_forecast, anomalies):
    pdf = FPDF()
    pdf.add_page()
    text_width = 180
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(text_width, 10, "Sales Forecast AI Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(text_width, 6, f"Total revenue: {inr(kpis['total_revenue'])}\nTotal units: {kpis['total_units']:,.0f}\nBest region: {kpis['best_region']}\nBest category: {kpis['best_category']}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "Predictive Intelligence Engine", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(text_width, 6, f"Best model: {metrics['Best Model']}\nAccuracy score: {metrics['Accuracy']:.1f}%\nWAPE: {metrics['WAPE']:.1f}%\nRMSE: {metrics['RMSE']:.2f}\nMAE: {metrics['MAE']:.2f}\nR2: {metrics['R2']:.3f}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "CRM Forecast Center", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    category_lines = "\n".join([f"{row.category}: {inr(row.amount)} weighted {inr(row.weighted_amount)}" for row in crm_forecast["categories"].itertuples()])
    pdf.multi_cell(
        text_width,
        6,
        (
            f"Period: {crm_forecast['period_label']} ({crm_forecast['date_range']})\n"
            f"Quota: {inr(crm_forecast['quota'])}\n"
            f"Weighted forecast: {inr(crm_forecast['weighted_forecast'])}\n"
            f"Closed revenue: {inr(crm_forecast['closed'])}\n"
            f"Quota gap: {inr(crm_forecast['quota_gap'])}\n"
            f"Attainment: {crm_forecast['attainment']:.1f}%\n"
            f"Pipeline coverage: {crm_forecast['pipeline_coverage']:.1f}x\n"
            f"Confidence: {crm_forecast['confidence']}%\n"
            f"{category_lines}"
        ),
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(text_width, 8, "Live Sales Forecast", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
            f"District-wise predicted sales:\n{city_lines}\n"
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


def forecast_card(label, value, detail):
    st.markdown(
        f"""
        <div class="forecast-card">
            <div class="forecast-label">{label}</div>
            <div class="forecast-value">{value}</div>
            <div class="forecast-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(text):
    return f'<span class="status-pill">{text}</span>'


def build_crm_forecast(df, live_forecast, anomalies):
    dated = df.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    dated = dated.dropna(subset=["date"]).sort_values("date")
    latest_date = dated["date"].max()
    quarter = int(((latest_date.month - 1) // 3) + 1)
    period_start = pd.Timestamp(year=latest_date.year, month=(quarter - 1) * 3 + 1, day=1)
    period_end = period_start + pd.offsets.QuarterEnd(0)
    previous_start = period_start - pd.DateOffset(months=3)
    previous_end = period_start - pd.Timedelta(days=1)

    current_rows = dated[(dated["date"] >= period_start) & (dated["date"] <= period_end)]
    previous_rows = dated[(dated["date"] >= previous_start) & (dated["date"] <= previous_end)]
    closed = float(current_rows["revenue"].sum())
    previous_closed = float(previous_rows["revenue"].sum())

    recent_start = latest_date - pd.Timedelta(days=30)
    recent_rows = dated[(dated["date"] >= recent_start) & (dated["date"] <= latest_date)]
    daily_average = float(recent_rows["revenue"].sum()) / max(1, min(30, max(1, len(recent_rows["date"].dt.date.unique()))))
    remaining_days = max(0, int((period_end - latest_date).days))
    pace_forecast = daily_average * remaining_days
    forecastable = max(float(live_forecast.get("next_30", 0)), pace_forecast, daily_average * 30)

    committed = forecastable * 0.34
    best_case = forecastable * 0.29
    pipeline = forecastable * 0.37
    omitted = forecastable * (0.07 if len(anomalies) else 0.03)

    categories = pd.DataFrame(
        [
            {"category": "Closed", "amount": closed, "weight": 1.00, "description": "Booked revenue in the period"},
            {"category": "Committed", "amount": committed, "weight": 0.90, "description": "High-confidence revenue"},
            {"category": "Best Case", "amount": best_case, "weight": 0.55, "description": "Upside revenue with close potential"},
            {"category": "Pipeline", "amount": pipeline, "weight": 0.25, "description": "Early-stage forecast pipeline"},
            {"category": "Omitted", "amount": omitted, "weight": 0.00, "description": "Excluded from the forecast"},
        ]
    )
    categories["weighted_amount"] = categories["amount"] * categories["weight"]
    weighted_forecast = float(categories["weighted_amount"].sum())
    quota = max(previous_closed * 1.12, closed * 1.18, weighted_forecast * 1.08, 1)
    quota_gap = max(0, quota - weighted_forecast)
    attainment = min(160, max(0, weighted_forecast / quota * 100))
    pipeline_coverage = min(9.9, (committed + best_case + pipeline) / max(quota_gap, quota * 0.1))
    confidence = int(min(94, max(52, 58 + attainment * 0.18 + min(pipeline_coverage, 4) * 5 - (omitted / quota * 100))))

    region_current = current_rows.groupby("region", as_index=False)["revenue"].sum()
    if region_current.empty:
        region_current = dated.groupby("region", as_index=False)["revenue"].sum()
    region_previous = previous_rows.groupby("region", as_index=False)["revenue"].sum().rename(columns={"revenue": "previous_revenue"})
    territories = region_current.merge(region_previous, on="region", how="left").fillna(0)
    total_region_closed = max(1, float(territories["revenue"].sum()))
    territories["share"] = territories["revenue"] / total_region_closed
    territories["manager"] = [FORECAST_MANAGERS[index % len(FORECAST_MANAGERS)] for index in range(len(territories))]
    territories["quota"] = np.maximum.reduce(
        [
            territories["previous_revenue"].to_numpy() * 1.12,
            territories["revenue"].to_numpy() * 1.2,
            quota * territories["share"].to_numpy(),
        ]
    )
    territories["committed"] = committed * territories["share"]
    territories["best_case"] = best_case * territories["share"]
    territories["pipeline"] = pipeline * territories["share"]
    territories["forecast"] = territories["revenue"] + territories["committed"] * 0.9 + territories["best_case"] * 0.55 + territories["pipeline"] * 0.25
    territories["attainment"] = territories["forecast"] / territories["quota"].replace(0, 1) * 100
    territories["status"] = np.where(
        territories["attainment"] >= 100,
        "On track",
        np.where(territories["attainment"] >= 82, "Needs commit", "At risk"),
    )
    territories = territories.sort_values("forecast", ascending=False)

    stages = pd.DataFrame(
        [
            {"stage": "Qualification", "probability": 15, "amount": pipeline * 0.42},
            {"stage": "Needs Analysis", "probability": 30, "amount": pipeline * 0.28},
            {"stage": "Proposal", "probability": 45, "amount": best_case * 0.44},
            {"stage": "Negotiation", "probability": 65, "amount": best_case * 0.56},
            {"stage": "Committed", "probability": 85, "amount": committed},
            {"stage": "Closed Won", "probability": 100, "amount": closed},
        ]
    )
    stages["weighted_amount"] = stages["amount"] * stages["probability"] / 100

    risk_signals = [
        f"{len(anomalies)} anomaly alert(s) need review before forecast commit." if len(anomalies) else "No major revenue anomaly is affecting the active forecast.",
        "Pipeline coverage is thin against quota gap." if pipeline_coverage < 1.5 else "Pipeline coverage is healthy for manager review.",
        f"{inr(quota_gap)} remains before weighted forecast reaches quota." if attainment < 100 else "Weighted forecast is above quota for the selected period.",
    ]

    return {
        "period_label": f"Q{quarter} FY{latest_date.year}",
        "date_range": f"{period_start:%d %b %Y} - {period_end:%d %b %Y}",
        "quota": quota,
        "closed": closed,
        "weighted_forecast": weighted_forecast,
        "quota_gap": quota_gap,
        "attainment": attainment,
        "pipeline_coverage": pipeline_coverage,
        "confidence": confidence,
        "categories": categories,
        "territories": territories,
        "stages": stages,
        "risk_signals": risk_signals,
    }


def render_forecast_center(forecast):
    st.subheader("Forecast Center")
    st.caption("Professional CRM-style sales forecasting with quota, commit, best case, pipeline, and territory rollups.")

    st.markdown(
        f"""
        <div class="forecast-panel">
            {status_pill(forecast["period_label"])} &nbsp; {status_pill(forecast["date_range"])}
            <h3 style="margin:.75rem 0 .2rem;color:#0F172A;">Quota, commit, and pipeline coverage</h3>
            <p style="margin:0;color:#475569;font-weight:600;">Built from uploaded revenue history, current-period closed revenue, recent pace, anomaly risk, and next-30-day forecast signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(5)
    kpis = [
        ("Quota", inr(forecast["quota"]), "Target for selected period"),
        ("Weighted Forecast", inr(forecast["weighted_forecast"]), f"{forecast['attainment']:.0f}% of quota"),
        ("Closed Revenue", inr(forecast["closed"]), "Booked revenue in period"),
        ("Quota Gap", inr(forecast["quota_gap"]), f"{forecast['pipeline_coverage']:.1f}x pipeline coverage"),
        ("Confidence", f"{forecast['confidence']}%", "Pace + coverage + risk"),
    ]
    for col, (label, value, detail) in zip(kpi_cols, kpis):
        with col:
            forecast_card(label, value, detail)

    left, right = st.columns([1.1, 0.9])
    with left:
        category_df = forecast["categories"].copy()
        fig = px.bar(
            category_df,
            x="amount",
            y="category",
            orientation="h",
            color="category",
            text=category_df["amount"].map(inr),
            title="Forecast Categories",
            color_discrete_map={
                "Closed": "#0F766E",
                "Committed": "#2563EB",
                "Best Case": "#F97316",
                "Pipeline": "#0F172A",
                "Omitted": "#94A3B8",
            },
        )
        fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=50, b=10))
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        stage_df = forecast["stages"].copy()
        fig = px.bar(
            stage_df,
            x="stage",
            y="weighted_amount",
            text=stage_df["weighted_amount"].map(inr),
            title="Stage Probability Weighted Revenue",
            color="probability",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Territory Forecast Rollup")
    territory_view = forecast["territories"][
        ["region", "manager", "quota", "revenue", "committed", "best_case", "pipeline", "forecast", "attainment", "status"]
    ].rename(
        columns={
            "region": "Territory",
            "manager": "Manager",
            "quota": "Quota",
            "revenue": "Closed",
            "committed": "Committed",
            "best_case": "Best Case",
            "pipeline": "Pipeline",
            "forecast": "Weighted Forecast",
            "attainment": "Attainment %",
            "status": "Status",
        }
    )
    st.dataframe(
        territory_view.style.format(
            {
                "Quota": inr,
                "Closed": inr,
                "Committed": inr,
                "Best Case": inr,
                "Pipeline": inr,
                "Weighted Forecast": inr,
                "Attainment %": "{:.1f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    risk_cols = st.columns(3)
    for col, signal in zip(risk_cols, forecast["risk_signals"]):
        with col:
            st.info(signal)


def render_project_assets(compact=False):
    if compact:
        st.markdown("### Project Links")
    else:
        st.subheader("Project Assets")
        st.caption("Use these assets to inspect the preprocessing workflow, explain the project, and review the source repository.")

    asset_cols = st.columns(3)
    with asset_cols[0]:
        forecast_card("Google Colab", "Preprocess Dataset", "Open the notebook, clean sales data, and export cleaned_sales_data.csv.")
        st.link_button("Open Colab Notebook", COLAB_URL, width="stretch")
    with asset_cols[1]:
        forecast_card("PowerPoint", "Project PPT", "Editable deck explaining the architecture, forecast center, Colab workflow, and Groq AI layer.")
        st.link_button("Open PPT From GitHub", PPT_URL, width="stretch")
        if os.path.exists(PPT_ASSET_PATH):
            with open(PPT_ASSET_PATH, "rb") as ppt_file:
                st.download_button(
                    "Download PPT",
                    ppt_file.read(),
                    "Sales_Forecast_AI_Project_Deck.pptx",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    width="stretch",
                )
    with asset_cols[2]:
        forecast_card("GitHub", "Source Repo", "Streamlit app, Colab notebook, sample data, requirements, and launch assets.")
        st.link_button("Open GitHub Repo", REPO_URL, width="stretch")

    if not compact:
        st.info("After pushing the latest files to GitHub, the Colab and PPT links above will resolve from the main branch.")


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
    st.caption("Statewide prediction coverage across major Tamil Nadu districts and commerce markets.")
    st.warning(forecast["disclaimer"])
    top_line = st.columns([1.4, 1, 1])
    with top_line[0]:
        metric_card("Live Statewide Counter", inr(forecast["month_counter"]), f"Updates every second at {forecast['timestamp']}")
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
    map_fig = px.scatter_map(
        forecast["city_sales"],
        lat="lat",
        lon="lon",
        size="predicted_sales",
        color="predicted_sales",
        hover_name="city",
        hover_data={"predicted_sales": ":,.0f", "share_pct": ":.2f", "second_velocity": ":,.2f", "lat": False, "lon": False},
        color_continuous_scale="YlOrRd",
        zoom=5.4,
        center={"lat": 11.1271, "lon": 78.6569},
        map_style="open-street-map",
        height=520,
        title="Live Sales Heatmap Across Tamil Nadu",
    )
    map_fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    st.plotly_chart(map_fig, width="stretch")

    city_col, category_col = st.columns(2)
    with city_col:
        st.plotly_chart(px.bar(forecast["city_sales"].head(18), x="city", y="predicted_sales", title="Tamil Nadu District-wise Predicted Sales"), width="stretch")
    with category_col:
        st.plotly_chart(px.bar(forecast["category_sales"], x="category", y="predicted_sales", title="Category-wise Predicted Sales"), width="stretch")
    with st.expander("All Tamil Nadu live district predictions"):
        st.dataframe(forecast["city_sales"], width="stretch")



st.markdown(
    """
    <style>
    .sidebar-brand {
        margin-bottom: 1rem;
    }
    .sidebar-brand h2 {
        margin: 0;
        color: #F8FAFC;
        font-size: 1.32rem;
    }
    .sidebar-brand p {
        margin: .35rem 0 0;
        color: rgba(255, 255, 255, .72);
        font-size: .9rem;
    }
    .sidebar-user {
        background: rgba(255, 255, 255, .08);
        border: 1px solid rgba(255, 255, 255, .12);
        border-radius: 8px;
        padding: .8rem .9rem;
        margin-bottom: .9rem;
    }
    .sidebar-user strong {
        display: block;
        color: #FFFFFF;
        margin-bottom: .18rem;
    }
    .sidebar-user span {
        display: block;
        color: rgba(255, 255, 255, .72);
        font-size: .82rem;
        line-height: 1.4;
    }
    .teamspace-line {
        display: flex;
        align-items: center;
        gap: .6rem;
        color: #FFFFFF;
        font-weight: 700;
        margin: 1rem 0 .7rem;
    }
    .teamspace-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.9rem;
        height: 1.9rem;
        border-radius: 6px;
        background: #0FB981;
        color: #FFFFFF;
        font-size: .82rem;
        font-weight: 800;
    }
    .nav-link {
        display: block;
        padding: .72rem .85rem;
        border-radius: 8px;
        margin: .2rem 0;
        color: rgba(255, 255, 255, .86);
        text-decoration: none;
        font-weight: 600;
        border: 1px solid transparent;
    }
    .nav-link:hover {
        background: rgba(255, 255, 255, .08);
        color: #FFFFFF;
    }
    .nav-link.active {
        background: #3C5180;
        color: #FFFFFF;
        border-color: rgba(255, 255, 255, .08);
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin: 0;
    }
    section[data-testid="stSidebar"] a.nav-link,
    section[data-testid="stSidebar"] a.nav-link:visited,
    section[data-testid="stSidebar"] a.nav-link:hover,
    section[data-testid="stSidebar"] a.nav-link:active {
        text-decoration: none !important;
    }
    .nav-caption {
        color: rgba(255, 255, 255, .56);
        font-size: .74rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
        margin: 1rem 0 .35rem;
    }
    .page-band {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid #D7DFEC;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
    }
    .page-band h1 {
        margin: 0;
        color: #091B33;
        font-size: 1.5rem;
    }
    .page-band p {
        margin: .4rem 0 0;
        color: #5F6F86;
        font-weight: 600;
    }
    .csv-upload-card {
        max-width: 740px;
        margin: 1.25rem auto 1rem;
        padding: 1.5rem;
        border: 1px solid #DDE6F2;
        border-radius: 8px;
        background: #FFFFFF;
        box-shadow: 0 18px 44px rgba(25, 43, 77, .08);
    }
    .csv-upload-card h2 {
        margin: 0 0 1rem;
        color: #111827;
        font-size: 1.22rem;
    }
    .csv-upload-card .upload-note {
        margin: .9rem 0 0;
        color: #6B7280;
        font-size: .88rem;
    }
    .csv-upload-card .upload-note strong {
        color: #111827;
    }
    div[data-testid="stFileUploader"] section {
        min-height: 190px;
        border: 1.5px dashed #20C6D8 !important;
        border-radius: 6px !important;
        background: #F8FCFF !important;
    }
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] small,
    div[data-testid="stFileUploaderDropzoneInstructions"] span:not(:first-child) {
        display: none !important;
    }
    .process-stage {
        padding: .75rem .9rem;
        border: 1px solid #D7DFEC;
        border-radius: 8px;
        background: #FFFFFF;
        color: #091B33;
        font-weight: 700;
        margin-bottom: .55rem;
    }
    .assistant-shell {
        min-height: 620px;
        border: 1px solid #D7DFEC;
        border-radius: 8px;
        background:
            linear-gradient(140deg, rgba(49, 91, 232, .08), rgba(15, 185, 129, .07)),
            #FFFFFF;
        padding: 1.4rem;
        margin-top: 1rem;
    }
    .assistant-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #D7DFEC;
        padding-bottom: .95rem;
        color: #091B33;
    }
    .assistant-brand {
        display: flex;
        align-items: center;
        gap: .7rem;
        font-weight: 800;
    }
    .assistant-mark {
        width: 2.1rem;
        height: 2.1rem;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #315BE8;
        color: #FFFFFF;
        font-size: .82rem;
        font-weight: 900;
    }
    .assistant-online {
        border: 1px solid rgba(15, 185, 129, .35);
        background: rgba(15, 185, 129, .1);
        color: #087A58;
        border-radius: 999px;
        padding: .35rem .85rem;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .08em;
    }
    .assistant-hero {
        text-align: center;
        max-width: 720px;
        margin: 4.5rem auto 2rem;
    }
    .assistant-avatar {
        width: 4.8rem;
        height: 4.8rem;
        border-radius: 50%;
        margin: 0 auto 1.4rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid rgba(49, 91, 232, .35);
        background: #F8FBFF;
        color: #315BE8;
        box-shadow: 0 12px 32px rgba(49, 91, 232, .18);
        font-size: 1.15rem;
        font-weight: 900;
    }
    .assistant-hero h1 {
        margin: 0;
        color: #091B33;
        font-size: 2.15rem;
        line-height: 1.15;
    }
    .assistant-hero p {
        margin: .8rem auto 0;
        max-width: 560px;
        color: #5F6F86;
        font-size: 1rem;
        line-height: 1.7;
        font-weight: 600;
    }
    .assistant-divider {
        border-top: 1px solid #D7DFEC;
        margin: 2.2rem 0 1.1rem;
    }
    .assistant-chat-card {
        max-width: 920px;
        margin: 0 auto;
        padding: 1rem;
        border: 1px solid #D7DFEC;
        border-radius: 8px;
        background: rgba(255, 255, 255, .86);
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_page_band(title, description):
    st.markdown(
        f"""
        <div class="page-band">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_processing_pipeline(summary, dataset_name):
    render_page_band(
        "Data Processing",
        "Drag and drop a dataset, then the platform maps fields, cleans records, engineers forecasting features, and prepares a model-ready output automatically.",
    )
    metric_cols = st.columns(5)
    pipeline_metrics = [
        ("Source Rows", f"{summary['raw_rows']:,}", dataset_name),
        ("Processed Rows", f"{summary['clean_rows']:,}", "Ready for forecast modeling"),
        ("Mapped Fields", str(summary["mapped_fields"]), "Detected or confirmed"),
        ("Engineered Features", str(summary["engineered_features"]), "Lag, rolling, and calendar signals"),
        ("Missing Cells Fixed", str(max(summary["missing_before"] - summary["missing_after"], 0)), "Recovered during cleaning"),
    ]
    for col, (label, value, detail) in zip(metric_cols, pipeline_metrics):
        with col:
            metric_card(label, value, detail)

    st.markdown("### Processing Workflow")
    workflow_cols = st.columns(4)
    workflow_steps = [
        ("1. Ingest", "CSV upload via drag and drop or click to upload"),
        ("2. Map", "Date, region, category, units, revenue, discount, and reviews"),
        ("3. Clean", "Parse dates, fill gaps, remove duplicates, normalize numerics"),
        ("4. Engineer", "Create lag, rolling, calendar, weekend, and quarter features"),
    ]
    for col, (title, detail) in zip(workflow_cols, workflow_steps):
        with col:
            forecast_card(title, "Completed", detail)

    st.info(
        f"Processed date range: {summary['date_start']} to {summary['date_end']}. "
        f"Duplicate rows removed: {summary['duplicates_removed']}. "
        f"Recognized raw date values: {summary['raw_date_coverage']:,}."
    )


PROCESSING_STAGES = [
    "Uploading",
    "Cleaning",
    "Preprocessing",
    "EDA",
    "Visualization",
    "Training",
    "Prediction",
    "Report",
    "Completed",
]


def build_future_predictions(live_forecast, forecast_chart):
    rows = [
        {"horizon": "Next day", "date": "", "predicted_revenue": live_forecast["tomorrow_forecast"]},
        {"horizon": "Next 7 days", "date": "", "predicted_revenue": live_forecast["next_7"]},
        {"horizon": "Next 30 days", "date": "", "predicted_revenue": live_forecast["next_30"]},
    ]
    daily_future = forecast_chart[forecast_chart["forecast"].notna()][["date", "forecast"]].copy()
    for row in daily_future.head(30).itertuples():
        rows.append({"horizon": "Daily forecast", "date": row.date.date().isoformat(), "predicted_revenue": float(row.forecast)})
    return pd.DataFrame(rows)


def generate_colab_notebook():
    code = r'''
!pip -q install pandas numpy matplotlib seaborn plotly scikit-learn xgboost prophet fpdf

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google.colab import files
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

try:
    from prophet import Prophet
except Exception:
    Prophet = None

uploaded = files.upload()
file_name = next(iter(uploaded))
raw_df = pd.read_csv(file_name, engine="python", on_bad_lines="skip")
raw_df.columns = [str(col).strip() for col in raw_df.columns]
display(raw_df.head())

ALIASES = {
    "date": ["date", "order_date", "sale_date", "created_at", "timestamp", "transaction_date", "invoice_date", "purchase_date"],
    "region": ["region", "state", "location", "city", "area", "district", "country", "market", "zone"],
    "product_category": ["product_category", "category", "product", "item", "product_type", "product_name", "sku", "department", "sub_category"],
    "units_sold": ["units_sold", "quantity", "qty", "units", "sales_count", "items_sold", "order_qty", "count"],
    "revenue": ["revenue", "sales", "amount", "total_price", "net_sales", "gross_sales", "order_value", "gmv", "subtotal"],
    "unit_price": ["unit_price", "selling_price", "price", "rate", "mrp", "item_price", "list_price"],
    "discount_pct": ["discount_pct", "discount", "discount_percent", "offer", "promo_discount", "coupon_discount"],
    "customer_reviews": ["customer_reviews", "review", "feedback", "comment", "customer_text", "customer_review", "remarks", "rating_text"],
}

def normalize(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

def auto_map(df):
    normalized = {normalize(col): col for col in df.columns}
    return {
        field: next((normalized[normalize(alias)] for alias in aliases if normalize(alias) in normalized), None)
        for field, aliases in ALIASES.items()
    }

mapping = auto_map(raw_df)
mapping

def preprocess(df, mapping):
    source = df.drop_duplicates().copy()
    clean = pd.DataFrame(index=source.index)
    if mapping.get("date"):
        clean["date"] = pd.to_datetime(source[mapping["date"]], errors="coerce")
        clean["date"] = clean["date"].fillna(clean["date"].dropna().min() if clean["date"].notna().any() else pd.Timestamp("2026-01-01"))
    else:
        clean["date"] = pd.date_range("2026-01-01", periods=len(clean), freq="D")
    clean["region"] = source[mapping["region"]].fillna("Overall").astype(str).str.strip() if mapping.get("region") else "Overall"
    clean["product_category"] = source[mapping["product_category"]].fillna("General").astype(str).str.strip() if mapping.get("product_category") else "General"
    clean["customer_reviews"] = source[mapping["customer_reviews"]].fillna("").astype(str) if mapping.get("customer_reviews") else ""
    clean["units_sold"] = pd.to_numeric(source[mapping["units_sold"]], errors="coerce") if mapping.get("units_sold") else np.nan
    clean["revenue"] = pd.to_numeric(source[mapping["revenue"]], errors="coerce") if mapping.get("revenue") else np.nan
    if clean["revenue"].isna().all() and mapping.get("unit_price"):
        unit_price = pd.to_numeric(source[mapping["unit_price"]], errors="coerce")
        clean["revenue"] = clean["units_sold"].fillna(1) * unit_price
    if clean["units_sold"].isna().all():
        clean["units_sold"] = np.where(clean["revenue"].notna(), np.maximum(clean["revenue"] / 100, 1), 1)
    if clean["revenue"].isna().all():
        clean["revenue"] = np.maximum(clean["units_sold"], 1) * 100
    clean["discount_pct"] = pd.to_numeric(source[mapping["discount_pct"]], errors="coerce") if mapping.get("discount_pct") else 0
    for col in ["units_sold", "revenue", "discount_pct"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(clean[col].median() if clean[col].notna().any() else 0)
    clean = clean.sort_values("date")
    clean["month"] = clean["date"].dt.month
    clean["year"] = clean["date"].dt.year
    clean["day_of_week"] = clean["date"].dt.dayofweek
    clean["day_of_month"] = clean["date"].dt.day
    clean["quarter"] = clean["date"].dt.quarter
    clean["is_weekend"] = clean["day_of_week"].isin([5, 6]).astype(int)
    clean["lag_7"] = clean["revenue"].shift(7)
    clean["lag_30"] = clean["revenue"].shift(30)
    clean["rolling_mean_7"] = clean["revenue"].rolling(7, min_periods=1).mean().shift(1)
    clean["rolling_mean_30"] = clean["revenue"].rolling(30, min_periods=1).mean().shift(1)
    clean["rolling_std_7"] = clean["revenue"].rolling(7, min_periods=2).std().shift(1)
    for col in ["lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30"]:
        clean[col] = clean[col].fillna(clean["revenue"].expanding().mean()).fillna(clean["revenue"].mean())
    clean["rolling_std_7"] = clean["rolling_std_7"].fillna(clean["rolling_std_7"].median() if clean["rolling_std_7"].notna().any() else 0)
    return clean.reset_index(drop=True)

clean_df = preprocess(raw_df, mapping)
display(clean_df.head())

daily = clean_df.groupby("date", as_index=False).agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
display(daily.describe())
px.line(daily, x="date", y="revenue", title="Daily Revenue").show()
px.bar(clean_df.groupby("region", as_index=False)["revenue"].sum(), x="region", y="revenue", title="Revenue by Region").show()
px.bar(clean_df.groupby("product_category", as_index=False)["revenue"].sum(), x="product_category", y="revenue", title="Revenue by Product").show()

features = [
    "units_sold", "discount_pct", "month", "year", "day_of_week", "day_of_month", "quarter",
    "is_weekend", "lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
    "region", "product_category",
]
x = clean_df[features]
y = clean_df["revenue"]
test_size = 0.2 if len(clean_df) >= 20 else 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)
preprocessor = ColumnTransformer([
    ("num", "passthrough", [col for col in features if col not in ["region", "product_category"]]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["region", "product_category"]),
])
models = {
    "RandomForest": RandomForestRegressor(n_estimators=260, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=260, learning_rate=.055, max_depth=4, objective="reg:squarederror", random_state=42, n_jobs=-1),
}
leaderboard = []
trained = {}
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else 0
    leaderboard.append({"model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    trained[name] = pipe
leaderboard = pd.DataFrame(leaderboard).sort_values("RMSE")
display(leaderboard)
best_model = trained[leaderboard.iloc[0]["model"]]

avg7 = daily["revenue"].tail(7).mean()
avg30 = daily["revenue"].tail(30).mean() if len(daily) >= 30 else avg7
trend = np.polyfit(np.arange(min(len(daily), 30)), daily["revenue"].tail(30), 1)[0] if len(daily) > 2 else 0
future_rows = []
last_date = daily["date"].max()
for day in range(1, 31):
    forecast_date = last_date + pd.Timedelta(days=day)
    estimate = max(0, avg7 * .65 + avg30 * .35 + trend * day)
    future_rows.append({"date": forecast_date, "predicted_revenue": estimate})
predictions = pd.DataFrame(future_rows)
display(predictions.head())
px.line(predictions, x="date", y="predicted_revenue", title="Future Sales Prediction").show()

if Prophet is not None and mapping.get("date"):
    prophet_df = daily.rename(columns={"date": "ds", "revenue": "y"})
    prophet = Prophet()
    prophet.fit(prophet_df)
    prophet_future = prophet.make_future_dataframe(periods=30)
    prophet_forecast = prophet.predict(prophet_future)
    prophet.plot(prophet_forecast)
    plt.show()

clean_df.to_csv("processed_sales_dataset.csv", index=False)
predictions.to_csv("future_sales_predictions.csv", index=False)
files.download("processed_sales_dataset.csv")
files.download("future_sales_predictions.csv")
'''
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Sales Forecast AI - Colab Processing Notebook\n", "Upload a messy CSV and run the full preprocessing, EDA, ML, and prediction workflow.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2).encode("utf-8")


def process_uploaded_sales_dataset(raw_df, dataset_name, column_mapping):
    progress = st.progress(0, text=PROCESSING_STAGES[0])
    stage_box = st.empty()

    def set_stage(index):
        progress.progress((index + 1) / len(PROCESSING_STAGES), text=PROCESSING_STAGES[index])
        stage_box.markdown(f"<div class='process-stage'>{PROCESSING_STAGES[index]}</div>", unsafe_allow_html=True)

    set_stage(0)
    set_stage(1)
    clean_df = preprocess_data(raw_df, column_mapping)
    if len(clean_df) < 2:
        raise ValueError("The dataset needs at least 2 usable rows after cleaning.")

    set_stage(2)
    processing_summary = build_processing_summary(raw_df, clean_df, column_mapping)

    set_stage(3)
    sentiment_pct, keywords, issues = analyze_reviews(clean_df)
    anomalies = detect_anomalies(clean_df)
    region_sales = clean_df.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    category_sales = clean_df.groupby("product_category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)

    set_stage(4)
    forecast_chart = forecast_daily_sales(clean_df)
    live_forecast = live_sales_prediction(clean_df)
    india_forecast = india_2026_forecast(clean_df)
    tn_forecast = tamil_nadu_live_prediction(clean_df)

    set_stage(5)
    model, metrics, validation, leaderboard = build_model(clean_df)

    set_stage(6)
    crm_forecast = build_crm_forecast(clean_df, live_forecast, anomalies)
    predictions_df = build_future_predictions(live_forecast, forecast_chart)
    default_pred_date = date(2026, 5, 5)
    default_region = sorted(clean_df["region"].unique())[0]
    default_product = sorted(clean_df["product_category"].unique())[0]
    default_units = float(clean_df["units_sold"].median())
    default_discount = float(clean_df["discount_pct"].median())
    default_predicted_revenue, _default_feature_row = predict_revenue_scenario(
        model,
        clean_df,
        default_pred_date,
        default_region,
        default_product,
        default_units,
        default_discount,
    )
    default_prediction_result = {
        "date": default_pred_date,
        "region": default_region,
        "product_category": default_product,
        "units_sold": default_units,
        "discount_pct": default_discount,
        "predicted_revenue": default_predicted_revenue,
    }

    set_stage(7)
    kpis = {
        "total_revenue": clean_df["revenue"].sum(),
        "total_units": clean_df["units_sold"].sum(),
        "best_region": region_sales.iloc[0]["region"],
        "best_region_revenue": region_sales.iloc[0]["revenue"],
        "best_category": category_sales.iloc[0]["product_category"],
    }
    summary, recommendations = business_summary(clean_df, kpis, sentiment_pct, issues, anomalies, india_forecast, tn_forecast)
    ai_context_text = build_ai_context(clean_df, kpis, sentiment_pct, keywords, issues, anomalies, india_forecast, live_forecast, tn_forecast, recommendations, metrics)
    ai_context_text += f"""

CRM forecast center:
Period: {crm_forecast['period_label']} ({crm_forecast['date_range']})
Quota: {inr(crm_forecast['quota'])}
Weighted forecast: {inr(crm_forecast['weighted_forecast'])}
Closed revenue: {inr(crm_forecast['closed'])}
Quota gap: {inr(crm_forecast['quota_gap'])}
Attainment: {crm_forecast['attainment']:.1f}%
Pipeline coverage: {crm_forecast['pipeline_coverage']:.1f}x
Forecast confidence: {crm_forecast['confidence']}%
Forecast categories: {', '.join([f"{row.category}: {inr(row.amount)}" for row in crm_forecast['categories'].itertuples()])}
Territory statuses: {', '.join([f"{row.region}: {row.status}" for row in crm_forecast['territories'].itertuples()])}
"""

    set_stage(8)
    return {
        "raw_df": raw_df,
        "dataset_name": dataset_name,
        "column_mapping": column_mapping,
        "clean_df": clean_df,
        "model": model,
        "metrics": metrics,
        "validation": validation,
        "leaderboard": leaderboard,
        "forecast_chart": forecast_chart,
        "live_forecast": live_forecast,
        "india_forecast": india_forecast,
        "tn_forecast": tn_forecast,
        "sentiment_pct": sentiment_pct,
        "keywords": keywords,
        "issues": issues,
        "anomalies": anomalies,
        "crm_forecast": crm_forecast,
        "region_sales": region_sales,
        "category_sales": category_sales,
        "kpis": kpis,
        "summary": summary,
        "recommendations": recommendations,
        "ai_context_text": ai_context_text,
        "processing_summary": processing_summary,
        "default_prediction_result": default_prediction_result,
        "predictions_df": predictions_df,
    }


def build_dataset_context(raw_df, dataset_name, column_mapping):
    clean_df = preprocess_data(raw_df, column_mapping)
    if len(clean_df) < 2:
        raise ValueError("The dataset needs at least 2 usable rows after cleaning.")

    processing_summary = build_processing_summary(raw_df, clean_df, column_mapping)
    sentiment_pct, keywords, issues = analyze_reviews(clean_df)
    anomalies = detect_anomalies(clean_df)
    region_sales = clean_df.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    category_sales = clean_df.groupby("product_category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    forecast_chart = forecast_daily_sales(clean_df)
    live_forecast = live_sales_prediction(clean_df)
    india_forecast = india_2026_forecast(clean_df)
    tn_forecast = tamil_nadu_live_prediction(clean_df)
    model, metrics, validation, leaderboard = build_model(clean_df)
    crm_forecast = build_crm_forecast(clean_df, live_forecast, anomalies)
    predictions_df = build_future_predictions(live_forecast, forecast_chart)

    default_pred_date = date(2026, 5, 5)
    default_region = sorted(clean_df["region"].unique())[0]
    default_product = sorted(clean_df["product_category"].unique())[0]
    default_units = float(clean_df["units_sold"].median())
    default_discount = float(clean_df["discount_pct"].median())
    default_predicted_revenue, _default_feature_row = predict_revenue_scenario(
        model,
        clean_df,
        default_pred_date,
        default_region,
        default_product,
        default_units,
        default_discount,
    )
    default_prediction_result = {
        "date": default_pred_date,
        "region": default_region,
        "product_category": default_product,
        "units_sold": default_units,
        "discount_pct": default_discount,
        "predicted_revenue": default_predicted_revenue,
    }
    kpis = {
        "total_revenue": clean_df["revenue"].sum(),
        "total_units": clean_df["units_sold"].sum(),
        "best_region": region_sales.iloc[0]["region"],
        "best_region_revenue": region_sales.iloc[0]["revenue"],
        "best_category": category_sales.iloc[0]["product_category"],
    }
    summary, recommendations = business_summary(clean_df, kpis, sentiment_pct, issues, anomalies, india_forecast, tn_forecast)
    ai_context_text = build_ai_context(clean_df, kpis, sentiment_pct, keywords, issues, anomalies, india_forecast, live_forecast, tn_forecast, recommendations, metrics)
    ai_context_text += f"""

CRM forecast center:
Period: {crm_forecast['period_label']} ({crm_forecast['date_range']})
Quota: {inr(crm_forecast['quota'])}
Weighted forecast: {inr(crm_forecast['weighted_forecast'])}
Closed revenue: {inr(crm_forecast['closed'])}
Quota gap: {inr(crm_forecast['quota_gap'])}
Attainment: {crm_forecast['attainment']:.1f}%
Pipeline coverage: {crm_forecast['pipeline_coverage']:.1f}x
Forecast confidence: {crm_forecast['confidence']}%
Forecast categories: {', '.join([f"{row.category}: {inr(row.amount)}" for row in crm_forecast['categories'].itertuples()])}
Territory statuses: {', '.join([f"{row.region}: {row.status}" for row in crm_forecast['territories'].itertuples()])}
"""
    return {
        "raw_df": raw_df,
        "dataset_name": dataset_name,
        "column_mapping": column_mapping,
        "clean_df": clean_df,
        "model": model,
        "metrics": metrics,
        "validation": validation,
        "leaderboard": leaderboard,
        "forecast_chart": forecast_chart,
        "live_forecast": live_forecast,
        "india_forecast": india_forecast,
        "tn_forecast": tn_forecast,
        "sentiment_pct": sentiment_pct,
        "keywords": keywords,
        "issues": issues,
        "anomalies": anomalies,
        "crm_forecast": crm_forecast,
        "region_sales": region_sales,
        "category_sales": category_sales,
        "kpis": kpis,
        "summary": summary,
        "recommendations": recommendations,
        "ai_context_text": ai_context_text,
        "processing_summary": processing_summary,
        "default_prediction_result": default_prediction_result,
        "predictions_df": predictions_df,
    }


def build_default_dataset_context():
    raw_df = load_sample_data()
    column_mapping = auto_detect_column_mapping(raw_df)
    return build_dataset_context(raw_df, "Sample dataset", column_mapping)


def render_processed_downloads(context):
    report_bytes = pdf_report(
        context["summary"],
        context["recommendations"],
        context["kpis"],
        context["metrics"],
        context["india_forecast"],
        context["live_forecast"],
        context["tn_forecast"],
        context["crm_forecast"],
        context["anomalies"],
    )
    download_cols = st.columns(4)
    with download_cols[0]:
        st.download_button("Download processed dataset CSV", context["clean_df"].to_csv(index=False).encode("utf-8"), "processed_sales_dataset.csv", "text/csv", width="stretch", key="download_processed_dataset")
    with download_cols[1]:
        st.download_button("Download predictions CSV", context["predictions_df"].to_csv(index=False).encode("utf-8"), "future_sales_predictions.csv", "text/csv", width="stretch", key="download_predictions_csv")
    with download_cols[2]:
        st.download_button("Download summary report PDF", report_bytes, "sales_forecast_summary_report.pdf", "application/pdf", width="stretch", key="download_summary_pdf")
    with download_cols[3]:
        st.download_button("Download Google Colab Notebook", generate_colab_notebook(), "sales_forecast_colab_pipeline.ipynb", "application/x-ipynb+json", width="stretch", key="download_processed_colab")


def render_csv_upload_processor():
    st.markdown(
        """
        <div class="csv-upload-card">
            <h2>Import Sales Dataset</h2>
            <div class="upload-note">Drag and Drop file here or <strong>Click to upload</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drag and Drop file here or Click to upload",
        type=["csv"],
        accept_multiple_files=False,
        key="sales_csv_upload",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("The dashboard is currently using the included sample dataset. Upload a CSV and click Process Dataset to replace it.")
        return None

    upload_signature = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
    if st.session_state.get("active_upload_signature") != upload_signature:
        st.session_state["active_upload_signature"] = upload_signature

    try:
        raw_df, dataset_name = read_uploaded_dataset(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        return None

    st.success(f"CSV loaded for review: {dataset_name} ({len(raw_df):,} rows, {len(raw_df.columns):,} columns).")
    detected_mapping = auto_detect_column_mapping(raw_df)
    mapping_warnings = mapping_completeness(detected_mapping)

    if mapping_warnings:
        st.warning("Some important columns were unclear. Confirm the mapping before processing.")
        for warning in mapping_warnings:
            st.caption(warning)
        column_mapping = render_column_mapping(raw_df, detected_mapping)
    else:
        st.success("Important sales columns were auto-detected.")
        with st.expander("Review or adjust detected column mapping", expanded=False):
            column_mapping = render_column_mapping(raw_df, detected_mapping)

    with st.expander("Preview uploaded CSV", expanded=False):
        st.dataframe(raw_df.head(25), width="stretch")

    process_clicked = st.button("Process Dataset", type="primary", width="stretch")
    if not process_clicked:
        return None

    with st.spinner("Processing uploaded CSV and building the dashboard..."):
        return process_uploaded_sales_dataset(raw_df, dataset_name, column_mapping)


def render_sidebar_navigation(current_page, user_info):
    provider_name, provider_state = ai_provider_status()
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2>Sales Forecast AI</h2>
                <p>Professional forecasting workspace</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="sidebar-user">
                <strong>{user_info['name']}</strong>
                <span>{user_info['email'] or 'Workspace access enabled'}</span>
                <span>{user_info['source']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if google_auth_configured() and getattr(getattr(st, "user", None), "is_logged_in", False):
            if st.button("Sign out", width="stretch"):
                st.logout()

        st.markdown(
            """
            <div class="teamspace-line">
                <span class="teamspace-badge">CT</span>
                <span>CRM Teamspace</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        nav_search = st.text_input("Search", placeholder="Search modules", label_visibility="collapsed")
        search_value = nav_search.strip().lower()

        for section, pages in PAGE_GROUPS.items():
            filtered_pages = [page for page in pages if not search_value or search_value in page.lower()]
            if not filtered_pages:
                continue
            st.markdown(f"<div class='nav-caption'>{section}</div>", unsafe_allow_html=True)
            for page in filtered_pages:
                state_class = "nav-link active" if page == current_page else "nav-link"
                st.markdown(
                    f"<a class='{state_class}' href='?page={page}' target='_self'>{NAV_LABELS[page]}</a>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.caption(f"AI provider: {provider_name} | {provider_state}")
        st.caption("Upload and process CSV files from the main workspace.")


def render_home_dashboard(clean_df, kpis, metrics, leaderboard, live_forecast, india_forecast, crm_forecast, recommendations, region_sales, category_sales):
    render_page_band("Org Overview", "A CRM-style summary of revenue, model health, pipeline coverage, and the next actions for this forecast cycle.")

    top_cols = st.columns(4)
    top_metrics = [
        ("Total Revenue", inr(kpis["total_revenue"]), "Uploaded data in current workspace"),
        ("Weighted Forecast", inr(crm_forecast["weighted_forecast"]), f"{crm_forecast['attainment']:.0f}% attainment"),
        ("Quota Gap", inr(crm_forecast["quota_gap"]), f"{crm_forecast['pipeline_coverage']:.1f}x coverage"),
        ("Confidence", f"{crm_forecast['confidence']}%", metrics["Best Model"]),
    ]
    for col, (label, value, detail) in zip(top_cols, top_metrics):
        with col:
            metric_card(label, value, detail)

    mid_cols = st.columns(4)
    mid_metrics = [
        ("Best Region", kpis["best_region"], inr(kpis["best_region_revenue"])),
        ("Best Category", kpis["best_category"], "Highest revenue contribution"),
        ("India 2026 Projection", inr(india_forecast["full_year"]), india_forecast["direction"]),
        ("Live 30-Day Forecast", inr(live_forecast["next_30"]), live_forecast["confidence"]),
    ]
    for col, (label, value, detail) in zip(mid_cols, mid_metrics):
        with col:
            metric_card(label, value, detail)

    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.plotly_chart(px.line(daily_revenue(clean_df), x="date", y="revenue", title="Revenue Trend"), width="stretch")
    with chart_b:
        st.plotly_chart(px.bar(region_sales, x="region", y="revenue", title="Region Contribution"), width="stretch")

    chart_c, chart_d = st.columns(2)
    with chart_c:
        st.plotly_chart(px.bar(category_sales, x="product_category", y="revenue", title="Category Performance"), width="stretch")
    with chart_d:
        st.plotly_chart(
            px.bar(
                leaderboard[["Model", "Accuracy"]],
                x="Model",
                y="Accuracy",
                title="Model Accuracy Score",
                color="Accuracy",
                color_continuous_scale="Blues",
            ),
            width="stretch",
        )

    st.markdown("### Recommended Actions")
    for rec in recommendations[:5]:
        st.info(rec)

    render_project_assets(compact=True)


def render_analytics_workspace(clean_df, validation, forecast_chart, region_sales, category_sales, live_forecast, metrics, anomalies):
    render_page_band("Analytics", "Forecast diagnostics, territory movement, validation quality, and live trend monitoring.")

    metric_cols = st.columns(4)
    analytics_metrics = [
        ("Validation Accuracy", f"{metrics['Accuracy']:.1f}%", f"WAPE {metrics['WAPE']:.1f}%"),
        ("Tracked Anomalies", str(len(anomalies)), "Revenue exceptions in uploaded data"),
        ("Today's Estimate", inr(live_forecast["today_estimate"]), live_forecast["confidence"]),
        ("Next 7 Days", inr(live_forecast["next_7"]), live_forecast["direction"]),
    ]
    for col, (label, value, detail) in zip(metric_cols, analytics_metrics):
        with col:
            metric_card(label, value, detail)

    upper_left, upper_right = st.columns(2)
    with upper_left:
        st.plotly_chart(px.line(validation, x="date", y=["actual", "predicted"], title="Validation: Actual vs Predicted"), width="stretch")
    with upper_right:
        st.plotly_chart(px.line(forecast_chart, x="date", y=["revenue", "forecast"], title="30-Day Forecast"), width="stretch")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.plotly_chart(px.bar(region_sales, x="region", y="revenue", title="Region-wise Sales"), width="stretch")
    with lower_right:
        st.plotly_chart(px.bar(category_sales, x="product_category", y="revenue", title="Category-wise Sales"), width="stretch")

    with st.expander("Open live Tamil Nadu commerce monitor", expanded=False):
        render_tamil_nadu_live_dashboard(clean_df)


def render_deals_workspace(model, clean_df, metrics, leaderboard, validation, default_prediction_result):
    render_page_band("Deals", "Model-assisted deal planning for scenario testing, weighted revenue review, and validation checks.")
    c1, c2, c3 = st.columns(3)
    with c1:
        pred_date = st.date_input("Prediction date", value=default_prediction_result["date"])
        region = st.selectbox("Region", sorted(clean_df["region"].unique()))
    with c2:
        product = st.selectbox("Product category", sorted(clean_df["product_category"].unique()))
        units = st.number_input("Units sold", min_value=0.0, value=float(clean_df["units_sold"].median()))
    with c3:
        discount = st.slider("Discount %", 0.0, 90.0, float(clean_df["discount_pct"].median()))

    predicted_revenue, _feature_row = predict_revenue_scenario(model, clean_df, pred_date, region, product, units, discount)
    st.session_state["latest_prediction"] = {
        "date": pred_date,
        "region": region,
        "product_category": product,
        "units_sold": units,
        "discount_pct": discount,
        "predicted_revenue": predicted_revenue,
    }

    preview_cols = st.columns(4)
    deal_metrics = [
        ("Predicted Revenue", inr(predicted_revenue), "Scenario output"),
        ("Best Model", metrics["Best Model"], "Auto-selected on validation"),
        ("Accuracy Score", f"{metrics['Accuracy']:.1f}%", f"Rows: {metrics['Validation Rows']}"),
        ("Validation WAPE", f"{metrics['WAPE']:.1f}%", "Lower is stronger"),
    ]
    for col, (label, value, detail) in zip(preview_cols, deal_metrics):
        with col:
            metric_card(label, value, detail)

    st.dataframe(leaderboard[["Model", "RMSE", "MAE", "R2", "WAPE", "Accuracy"]], width="stretch")
    st.plotly_chart(px.line(validation, x="date", y=["actual", "predicted"], title="Validation History"), width="stretch")


def render_documents_workspace(raw_df, clean_df, dataset_name, column_mapping):
    render_page_band("Documents", "Project assets, preprocessing links, schema mapping, and export-ready datasets in one place.")
    render_project_assets(compact=False)
    processing_summary = build_processing_summary(raw_df, clean_df, column_mapping)
    render_processing_pipeline(processing_summary, dataset_name)

    studio_cols = st.columns(4)
    with studio_cols[0]:
        metric_card("Raw Rows", f"{len(raw_df):,}", dataset_name)
    with studio_cols[1]:
        metric_card("Clean Rows", f"{len(clean_df):,}", "Ready for forecasting")
    with studio_cols[2]:
        metric_card("Columns", f"{len(raw_df.columns):,}", "Uploaded schema")
    with studio_cols[3]:
        metric_card("Mapped Fields", f"{sum(bool(value) for value in column_mapping.values())}", "Detected or selected")

    st.markdown("### Field Mapping")
    st.dataframe(pd.DataFrame([{"field": key, "mapped_column": value or "Fallback"} for key, value in column_mapping.items()]), width="stretch")
    st.markdown("### Raw Dataset Preview")
    st.dataframe(raw_df.head(100), width="stretch")
    st.markdown("### Cleaned Dataset Preview")
    preview_columns = list(dict.fromkeys(REQUIRED_COLUMNS + NUMERIC_FEATURES))
    st.dataframe(clean_df[preview_columns].head(100), width="stretch")
    st.download_button("Download cleaned dataset CSV", clean_df.to_csv(index=False).encode("utf-8"), "cleaned_sales_data.csv", "text/csv")


def render_table_workspace(title, description, table, metrics_row=None):
    render_page_band(title, description)
    if metrics_row:
        cols = st.columns(len(metrics_row))
        for col, (label, value, detail) in zip(cols, metrics_row):
            with col:
                metric_card(label, value, detail)
    st.dataframe(table, width="stretch", hide_index=True)


def answer_assistant_question(question, clean_df, kpis, sentiment_pct, issues, anomalies, india_forecast, live_forecast, tn_forecast, recommendations, metrics, crm_forecast, ai_context_text):
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
        crm_forecast,
    )
    try:
        answer, source = ask_ai(question, ai_context_text)
    except Exception as exc:
        answer, source = local_answer, f"Local insight engine ({exc})"
    return answer, source


def render_assistant_workspace(clean_df, kpis, sentiment_pct, issues, anomalies, india_forecast, live_forecast, tn_forecast, recommendations, metrics, crm_forecast, ai_context_text):
    provider_name, provider_state = ai_provider_status()
    st.markdown(
        f"""
        <div class="assistant-shell">
            <div class="assistant-topbar">
                <div class="assistant-brand">
                    <span class="assistant-mark">AI</span>
                    <div>
                        <div>Sales Forecast Assistant</div>
                        <div style="color:#5F6F86;font-size:.78rem;font-weight:700;letter-spacing:.08em;">REVENUE INTELLIGENCE</div>
                    </div>
                </div>
                <span class="assistant-online">{provider_state.upper()}</span>
            </div>
            <div class="assistant-hero">
                <div class="assistant-avatar">AI</div>
                <h1>Ask about your sales forecast</h1>
                <p>Use this assistant to understand quota health, revenue trends, territory risk, customer issues, model accuracy, and future sales predictions from the processed CSV.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_prompts = [
        "Summarize forecast health",
        "Which territories are at risk?",
        "Predict next 30 days sales",
        "Explain model accuracy",
        "Give business recommendations",
    ]
    prompt_cols = st.columns([1, 1, 1, 1, 1])
    pending_question = None
    for col, prompt in zip(prompt_cols, quick_prompts):
        with col:
            if st.button(prompt, width="stretch", key=f"assistant_prompt_{prompt}"):
                pending_question = prompt

    st.markdown("<div class='assistant-divider'></div>", unsafe_allow_html=True)
    st.caption(f"Provider: {provider_name}. Answers use the processed dataset and local fallback if the AI provider is unavailable.")

    if "assistant_chat_history" not in st.session_state:
        st.session_state.assistant_chat_history = []

    question = pending_question or st.chat_input("Ask about revenue, forecasts, pipeline, anomalies, or recommendations...")
    if question:
        answer, source = answer_assistant_question(
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
            crm_forecast,
            ai_context_text,
        )
        st.session_state.assistant_chat_history.append(("user", question, ""))
        st.session_state.assistant_chat_history.append(("assistant", answer, source))

    st.markdown("<div class='assistant-chat-card'>", unsafe_allow_html=True)
    if not st.session_state.assistant_chat_history:
        st.info("Start with a quick prompt or ask a custom question after processing your CSV.")
    for role, text, source in st.session_state.assistant_chat_history:
        with st.chat_message(role):
            st.write(text)
            if source:
                st.caption(source)
    st.markdown("</div>", unsafe_allow_html=True)


requested_page = st.query_params.get("page", "Analytics")
if isinstance(requested_page, list):
    requested_page = requested_page[0]
if requested_page not in PAGES:
    requested_page = "Analytics"
current_page = requested_page

user_info = render_google_auth_gate()
render_sidebar_navigation(current_page, user_info)
render_top_shell(user_info, current_page)

dataset_context = st.session_state.get("dataset_context")
if not dataset_context:
    dataset_context = build_default_dataset_context()
    st.session_state["dataset_context"] = dataset_context

raw_df = dataset_context["raw_df"]
dataset_name = dataset_context["dataset_name"]
column_mapping = dataset_context["column_mapping"]
clean_df = dataset_context["clean_df"]
model = dataset_context["model"]
metrics = dataset_context["metrics"]
validation = dataset_context["validation"]
leaderboard = dataset_context["leaderboard"]
forecast_chart = dataset_context["forecast_chart"]
live_forecast = dataset_context["live_forecast"]
india_forecast = dataset_context["india_forecast"]
tn_forecast = dataset_context["tn_forecast"]
sentiment_pct = dataset_context["sentiment_pct"]
keywords = dataset_context["keywords"]
issues = dataset_context["issues"]
anomalies = dataset_context["anomalies"]
crm_forecast = dataset_context["crm_forecast"]
region_sales = dataset_context["region_sales"]
category_sales = dataset_context["category_sales"]
kpis = dataset_context["kpis"]
summary = dataset_context["summary"]
recommendations = dataset_context["recommendations"]
ai_context_text = dataset_context["ai_context_text"]
processing_summary = dataset_context["processing_summary"]
default_prediction_result = dataset_context["default_prediction_result"]
predictions_df = dataset_context["predictions_df"]

if st.session_state.pop("dataset_processed_notice", False):
    st.success("Dataset processed successfully. Dashboard results are ready.")

if current_page in {"Home", "Documents"}:
    processed_context = render_csv_upload_processor()
    if processed_context:
        st.session_state["dataset_context"] = processed_context
        st.session_state["dataset_processed_notice"] = True
        st.rerun()

if current_page == "Documents":
    render_processed_downloads(dataset_context)

if current_page == "Home":
    render_home_dashboard(clean_df, kpis, metrics, leaderboard, live_forecast, india_forecast, crm_forecast, recommendations, region_sales, category_sales)
    render_processing_pipeline(processing_summary, dataset_name)

elif current_page == "Analytics":
    render_analytics_workspace(clean_df, validation, forecast_chart, region_sales, category_sales, live_forecast, metrics, anomalies)

elif current_page == "Assistant":
    render_assistant_workspace(clean_df, kpis, sentiment_pct, issues, anomalies, india_forecast, live_forecast, tn_forecast, recommendations, metrics, crm_forecast, ai_context_text)

elif current_page == "Forecasts":
    render_forecast_center(crm_forecast)

elif current_page == "Deals":
    render_deals_workspace(model, clean_df, metrics, leaderboard, validation, default_prediction_result)

elif current_page == "Documents":
    render_documents_workspace(raw_df, clean_df, dataset_name, column_mapping)

elif current_page == "Leads":
    leads_table = category_sales.rename(columns={"product_category": "Lead Segment", "revenue": "Open Revenue"}).copy()
    leads_table["Priority"] = ["High" if idx < 3 else "Monitor" for idx in range(len(leads_table))]
    leads_table["Expected Conversion"] = np.linspace(68, 34, len(leads_table)).round(0).astype(int).astype(str) + "%"
    render_table_workspace(
        "Leads",
        "Lead coverage by product segment using the uploaded revenue distribution as the active pipeline proxy.",
        leads_table,
        [
            ("Tracked Segments", str(len(leads_table)), "Active lead pools"),
            ("Top Segment", str(leads_table.iloc[0]["Lead Segment"]), inr(leads_table.iloc[0]["Open Revenue"])),
            ("Average Conversion", f"{np.linspace(68, 34, len(leads_table)).mean():.0f}%", "Weighted estimate"),
        ],
    )

elif current_page == "Contacts":
    contacts_table = crm_forecast["territories"][["manager", "region", "status", "forecast", "attainment"]].rename(
        columns={"manager": "Relationship Owner", "region": "Contact Territory", "status": "Coverage Status", "forecast": "Forecast Value", "attainment": "Attainment %"}
    )
    render_table_workspace(
        "Contacts",
        "Relationship coverage view for territory owners derived from the forecast rollup.",
        contacts_table.style.format({"Forecast Value": inr, "Attainment %": "{:.1f}%"}),
        [
            ("Owners", str(contacts_table["Relationship Owner"].nunique()), "Active territory coverage"),
            ("On-track Territories", str((crm_forecast["territories"]["status"] == "On track").sum()), "Healthy book of business"),
            ("Needs Attention", str((crm_forecast["territories"]["status"] != "On track").sum()), "Follow-up required"),
        ],
    )

elif current_page == "Accounts":
    accounts_table = region_sales.rename(columns={"region": "Account Region", "revenue": "Revenue"}).copy()
    accounts_table["Share %"] = (accounts_table["Revenue"] / max(accounts_table["Revenue"].sum(), 1) * 100).round(1)
    render_table_workspace(
        "Accounts",
        "Regional account concentration based on uploaded sales performance.",
        accounts_table.style.format({"Revenue": inr, "Share %": "{:.1f}%"}),
        [
            ("Regions", str(len(accounts_table)), "Tracked account territories"),
            ("Largest Region", str(accounts_table.iloc[0]["Account Region"]), inr(accounts_table.iloc[0]["Revenue"])),
            ("Revenue Spread", f"{accounts_table['Share %'].max():.1f}%", "Top-region concentration"),
        ],
    )

elif current_page == "Campaigns":
    campaigns_table = category_sales.rename(columns={"product_category": "Campaign Theme", "revenue": "Influenced Revenue"}).copy()
    campaigns_table["Focus"] = ["Scale" if idx < 2 else "Optimize" if idx < 5 else "Test" for idx in range(len(campaigns_table))]
    render_table_workspace(
        "Campaigns",
        "Category-led campaign planning based on the strongest commercial themes in the dataset.",
        campaigns_table.style.format({"Influenced Revenue": inr}),
        [
            ("Active Themes", str(len(campaigns_table)), "Campaign opportunities"),
            ("Scale Now", str((campaigns_table["Focus"] == "Scale").sum()), "High-performing themes"),
            ("Optimization Queue", str((campaigns_table["Focus"] == "Optimize").sum()), "Needs tuning"),
        ],
    )

elif current_page == "Tasks":
    risk_territories = crm_forecast["territories"][crm_forecast["territories"]["status"] != "On track"]
    task_rows = [{"Task": rec, "Owner": "Forecast Ops", "Priority": "High" if idx < 2 else "Medium"} for idx, rec in enumerate(recommendations[:5])]
    task_rows.extend(
        [{"Task": f"Review commit plan for {row.region}", "Owner": row.manager, "Priority": "High" if row.status == "At risk" else "Medium"} for row in risk_territories.itertuples()]
    )
    tasks_table = pd.DataFrame(task_rows).drop_duplicates().head(12)
    render_table_workspace(
        "Tasks",
        "Execution queue for closing the quarter with stronger quota coverage.",
        tasks_table,
        [
            ("Open Tasks", str(len(tasks_table)), "Actionable work items"),
            ("High Priority", str((tasks_table["Priority"] == "High").sum()), "Immediate follow-up"),
            ("At-risk Territories", str((crm_forecast["territories"]["status"] == "At risk").sum()), "Escalation candidates"),
        ],
    )

elif current_page == "Meetings":
    meetings_table = crm_forecast["territories"][["manager", "region", "status", "attainment"]].copy()
    meetings_table["Meeting Type"] = np.where(meetings_table["status"] == "At risk", "Recovery Review", "Forecast Cadence")
    meetings_table["Cadence"] = np.where(meetings_table["status"] == "On track", "Weekly", "Twice Weekly")
    meetings_table = meetings_table.rename(columns={"manager": "Host", "region": "Territory", "status": "Status", "attainment": "Attainment %"})
    render_table_workspace(
        "Meetings",
        "Forecast review rhythm for managers and territory owners.",
        meetings_table.style.format({"Attainment %": "{:.1f}%"}),
        [
            ("Planned Reviews", str(len(meetings_table)), "Scheduled forecast check-ins"),
            ("Recovery Reviews", str((meetings_table["Meeting Type"] == "Recovery Review").sum()), "Focused intervention"),
            ("Weekly Cadence", str((meetings_table["Cadence"] == "Weekly").sum()), "Healthy territories"),
        ],
    )

elif current_page == "Calls":
    calls_table = crm_forecast["territories"][["manager", "region", "status", "quota", "forecast"]].copy()
    calls_table["Call Purpose"] = np.where(calls_table["status"] == "At risk", "Close gap", "Confirm commit")
    calls_table = calls_table.rename(columns={"manager": "Rep", "region": "Territory", "status": "Status", "quota": "Quota", "forecast": "Forecast"})
    render_table_workspace(
        "Calls",
        "Call queue for managers working committed and best-case pipeline.",
        calls_table.style.format({"Quota": inr, "Forecast": inr}),
        [
            ("Rep Queue", str(len(calls_table)), "Territory call lists"),
            ("Commit Calls", str((calls_table["Call Purpose"] == "Confirm commit").sum()), "Protect current forecast"),
            ("Gap Calls", str((calls_table["Call Purpose"] == "Close gap").sum()), "Quota recovery"),
        ],
    )

elif current_page == "Reports":
    render_page_band("Reports", "Executive summary, forecast posture, exports, and a compact record of what the system recommends next.")
    st.write(summary)
    st.markdown("### CRM Forecast Summary")
    forecast_report_cols = st.columns(4)
    with forecast_report_cols[0]:
        metric_card("Quota", inr(crm_forecast["quota"]), crm_forecast["period_label"])
    with forecast_report_cols[1]:
        metric_card("Weighted Forecast", inr(crm_forecast["weighted_forecast"]), f"{crm_forecast['attainment']:.1f}% attainment")
    with forecast_report_cols[2]:
        metric_card("Quota Gap", inr(crm_forecast["quota_gap"]), f"{crm_forecast['pipeline_coverage']:.1f}x coverage")
    with forecast_report_cols[3]:
        metric_card("Forecast Confidence", f"{crm_forecast['confidence']}%", "Commit + pace + risk")
    st.markdown("### Tamil Nadu Live Prediction Summary")
    st.write(tn_forecast["insight"])
    tn_report_cols = st.columns(3)
    with tn_report_cols[0]:
        metric_card("May 2026 Forecast", inr(tn_forecast["monthly_prediction"]), f"Growth: {tn_forecast['growth_rate']:.1f}%")
    with tn_report_cols[1]:
        metric_card("Top Tamil Nadu District", tn_forecast["city_sales"].iloc[0]["city"], inr(tn_forecast["city_sales"].iloc[0]["predicted_sales"]))
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

    prediction_for_report = st.session_state.get("latest_prediction", default_prediction_result)
    prediction_output = pd.DataFrame(
        [
            {
                "date": prediction_for_report["date"],
                "region": prediction_for_report["region"],
                "product_category": prediction_for_report["product_category"],
                "units_sold": prediction_for_report["units_sold"],
                "discount_pct": prediction_for_report["discount_pct"],
                "predicted_revenue": prediction_for_report["predicted_revenue"],
            }
        ]
    ).to_csv(index=False).encode("utf-8")
    st.download_button("Download latest prediction CSV", prediction_output, "prediction_results.csv", "text/csv")

    report_bytes = pdf_report(summary, recommendations, kpis, metrics, india_forecast, live_forecast, tn_forecast, crm_forecast, anomalies)
    st.download_button("Download PDF report", report_bytes, "sales_prediction_ai_report.pdf", "application/pdf")
