import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

st.set_page_config("Water Dashboard", layout="wide")

# These are the columns every CSV needs.
CSV_FILE = "Data/field_ops_template.csv"
CODE_VERSION = "simple_app_v1"
COLS = ["site_id", "timestamp", "tds_ppm", "pH", "turbidity_ntu", "air_temp_c", "water_temp_c", "humidity_pct", "weather_cond"]
NUMBER_COLS = ["tds_ppm", "pH", "turbidity_ntu", "air_temp_c", "water_temp_c", "humidity_pct", "weather_cond"]


def fix_data(data):
    # Make sure the data has the right columns and types.
    data = data.reindex(columns=COLS + ["source"]).copy()
    data["source"] = data["source"].fillna("")
    data["site_id"] = data["site_id"].fillna("").astype(str).str.strip()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for col in NUMBER_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data[data["site_id"] != ""]
    return data.sort_values(["site_id", "timestamp"])


def save_data():
    # Only save the real CSV columns.
    st.session_state.data[COLS].to_csv(CSV_FILE, index=False)


def clear_everything():
    # Clear the website and the saved CSV.
    st.session_state.data = fix_data(pd.DataFrame(columns=COLS))
    st.session_state.extra_sites = []
    st.session_state.uploads = {}
    st.session_state.upload_box_number += 1
    save_data()


def make_forecast(site_data, col):
    # Prophet needs at l;east 2 points.
    prophet_data = site_data[["timestamp", col]].dropna()
    prophet_data = prophet_data.rename(columns={"timestamp": "ds", col: "y"})
    prophet_data = prophet_data.groupby("ds", as_index=False)["y"].mean()

    if len(prophet_data) < 2:
        return pd.DataFrame()

    points_to_add = math.ceil(len(prophet_data) * 0.5)
    time_gap = prophet_data["ds"].sort_values().diff().median()
    if pd.isna(time_gap):
        time_gap = pd.Timedelta(days=1)

    freq = "h" if time_gap < pd.Timedelta(days=2) else "D"
    days = (prophet_data["ds"].max() - prophet_data["ds"].min()).total_seconds() / 86400

    try:
        model = Prophet(
            daily_seasonality=time_gap < pd.Timedelta(days=1) and len(prophet_data) >= 12,
            weekly_seasonality=days >= 7,
            yearly_seasonality=False,
            changepoint_prior_scale=0.2,
            seasonality_prior_scale=15,
        )
        if len(prophet_data) >= 24:
            model.add_seasonality(name="extra_cycle", period=1, fourier_order=8)
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=points_to_add, freq=freq, include_history=False)
        return model.predict(future)
    except Exception:
        return pd.DataFrame()


def make_chart(site_data, col, title, suffix="", safe_ranges=None, temp_chart=False):
    # This makes one chart with real points and Prophet points.
    fig = go.Figure()

    if temp_chart:
        fig.add_trace(go.Scatter(x=site_data["timestamp"], y=site_data["air_temp_c"], mode="markers", name="Air temp", marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=site_data["timestamp"], y=site_data["water_temp_c"], mode="markers", name="Water temp", marker=dict(size=10)))
        for temp_col, name in [("air_temp_c", "Air forecast"), ("water_temp_c", "Water forecast")]:
            forecast = make_forecast(site_data, temp_col)
            if not forecast.empty:
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="markers", name=name, marker=dict(size=8, symbol="diamond")))
    else:
        fig.add_trace(go.Scatter(x=site_data["timestamp"], y=site_data[col], mode="markers", name=title, marker=dict(size=10)))
        forecast = make_forecast(site_data, col)
        if not forecast.empty:
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="markers", name="Prophet forecast", marker=dict(size=8, symbol="diamond")))

    if safe_ranges:
        for low, high, color, label in safe_ranges:
            fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.25, line_width=0, annotation_text=label)

    fig.update_layout(title=title, height=390, template="plotly_white", yaxis_ticksuffix=suffix)
    st.plotly_chart(fig, use_container_width=True)


# Start s ession data.
if st.session_state.get("code_version") != CODE_VERSION:
    st.session_state.data = fix_data(pd.read_csv(CSV_FILE))
    st.session_state.extra_sites = []
    st.session_state.uploads = {}
    st.session_state.upload_box_number = 0
    st.session_state.code_version = CODE_VERSION

st.session_state.setdefault("extra_sites", [])
st.session_state.setdefault("uploads", {})
st.session_state.setdefault("upload_box_number", 0)

st.title("Water Quality Dashboard")
st.write("Upload data, edit it like a spreadsheet, and forecast with Facebook Prophet.")

# Clear button area.
if st.button("Clear all data"):
    clear_everything()
    st.rerun()

# Upload CSV files.
files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True, key="uploader_" + str(st.session_state.upload_box_number))
for file in files:
    key = file.name + str(getattr(file, "size", 0))
    if key not in st.session_state.uploads:
        new_data = pd.read_csv(file)
        if all(col in new_data.columns for col in COLS):
            new_data["source"] = key
            new_data = fix_data(new_data)
            st.session_state.data = fix_data(pd.concat([st.session_state.data, new_data], ignore_index=True))
            st.session_state.uploads[key] = file.name
            save_data()
        else:
            st.error(file.name + " does not have the right columns.")

# Show uploaded files and delete buttons.
if st.session_state.uploads:
    st.subheader("Uploaded files")
    for key, name in list(st.session_state.uploads.items()):
        col1, col2 = st.columns([3, 1])
        col1.write(name)
        if col2.button("Delete", key="delete_" + key):
            st.session_state.data = st.session_state.data[st.session_state.data["source"] != key]
            del st.session_state.uploads[key]
            st.session_state.upload_box_number += 1
            save_data()
            st.rerun()

# Create empty site tabs.
with st.form("new_site_form"):
    new_site = st.text_input("Create site")
    made_site = st.form_submit_button("Create")
    if made_site and new_site:
        st.session_state.extra_sites.append(new_site.strip())
        st.rerun()

st.download_button("Export all CSV", st.session_state.data[COLS].to_csv(index=False), "water_quality.csv", "text/csv")

all_sites = sorted(set(st.session_state.data["site_id"].unique()) | set(st.session_state.extra_sites))

if not all_sites:
    st.info("No data yet. Upload a CSV or create a site.")
else:
    tabs = st.tabs(all_sites)

    for tab, site in zip(tabs, all_sites):
        with tab:
            site_data = st.session_state.data[st.session_state.data["site_id"] == site]

            if st.button("Delete site", key="site_delete_" + site):
                st.session_state.data = st.session_state.data[st.session_state.data["site_id"] != site]
                st.session_state.extra_sites = [x for x in st.session_state.extra_sites if x != site]
                save_data()
                st.rerun()

            st.subheader(site + " spreadsheet")
            edit_cols = COLS[1:]
            edited = st.data_editor(site_data[edit_cols], num_rows="dynamic", use_container_width=True, key="edit_" + site)

            if st.button("Save spreadsheet", key="save_" + site):
                edited = edited.dropna(how="all")
                edited.insert(0, "site_id", site)
                edited["source"] = ""
                old_data = st.session_state.data[st.session_state.data["site_id"] != site]
                st.session_state.data = fix_data(pd.concat([old_data, edited], ignore_index=True))
                save_data()
                st.rerun()

            st.download_button("Export this site", site_data[COLS].to_csv(index=False), site + ".csv", "text/csv", key="download_" + site)

            if site_data.empty:
                st.info("Add rows and save to see charts.")
            else:
                make_chart(site_data, "turbidity_ntu", "Turbidity", " NTU", [
                    (0, 1, "lightgreen", "Clear"),
                    (2, 6, "khaki", "Cloudy"),
                    (6, max(8, site_data["turbidity_ntu"].max() + 2), "salmon", "High risk"),
                ])
                make_chart(site_data, "pH", "pH", "", [(6.5, 8.5, "lightgreen", "Safe range")])
                make_chart(site_data, "tds_ppm", "Total Dissolved Solids", " ppm", [
                    (0, 500, "lightgreen", "Safe range"),
                    (500, max(650, site_data["tds_ppm"].max() + 100), "salmon", "High"),
                ])
                make_chart(site_data, "", "Air and Water Temperature", " C", temp_chart=True)
                make_chart(site_data, "humidity_pct", "Humidity", "%")
