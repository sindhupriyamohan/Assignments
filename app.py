"""
Shipment Delay Predictor — Streamlit app.

Loads the Gradient Boosting model trained in Final Code.ipynb, reproduces the
exact same feature-engineering pipeline for a single user-entered shipment,
predicts delay risk, and surfaces the specific risk factors driving that
prediction along with actionable options to avoid the delay.

Run: streamlit run app.py   (from inside the `app` folder, after the notebook's
export cell has created `model_artifacts.pkl` in this same folder)
"""

from datetime import date

import joblib
import pandas as pd
import streamlit as st

import db

st.set_page_config(page_title="Shipment Delay Predictor", page_icon="🚚", layout="wide")

db.init_db()


@st.cache_resource
def load_artifacts():
    return joblib.load("model_artifacts.pkl")


@st.cache_resource
def load_penalty_artifacts():
    return joblib.load("penalty_risk_artifacts.pkl")


artifacts = load_artifacts()
model = artifacts["model"]
selected_features = artifacts["selected_features"]
all_columns = artifacts["all_columns"]
nominal_cols = artifacts["nominal_cols"]
priority_map = artifacts["priority_map"]
tier_map = artifacts["tier_map"]
rainfall_p75 = artifacts["rainfall_p75"]
category_values = artifacts["category_values"]
risk_thresholds = artifacts["risk_thresholds"]

penalty_artifacts = load_penalty_artifacts()
severity_model = penalty_artifacts["severity_model"]
severity_selected_features = penalty_artifacts["severity_selected_features"]
severity_all_columns = penalty_artifacts["severity_all_columns"]
prior_beta = penalty_artifacts["beta"]
prior_pi = penalty_artifacts["pi"]
mean_match_scale = penalty_artifacts["mean_match_scale"]


def build_features(raw: dict):
    """Reproduce the notebook's feature engineering + encoding for one raw shipment."""
    df_row = pd.DataFrame([raw])

    dt = pd.to_datetime(df_row["shipment_date"])
    df_row["shipment_month"] = dt.dt.month
    df_row["shipment_quarter"] = dt.dt.quarter
    df_row["shipment_dayofweek"] = dt.dt.dayofweek
    df_row["shipment_is_weekend"] = (df_row["shipment_dayofweek"] >= 5).astype(int)
    df_row = df_row.drop(columns=["shipment_date"])

    eps = 1e-6
    df_row["shipment_density"] = df_row["shipment_weight_kg"] / (df_row["shipment_volume_cbm"] + eps)
    df_row["transport_cost_per_kg"] = df_row["transportation_cost"] / (df_row["shipment_weight_kg"] + eps)
    df_row["avg_speed_kmph"] = df_row["route_distance_km"] / (df_row["estimated_travel_time_hrs"] + eps)
    df_row["total_lead_time_hrs"] = df_row["warehouse_processing_time_hrs"] + df_row["estimated_travel_time_hrs"]
    df_row["maintenance_overdue_flag"] = (df_row["maintenance_overdue_days"] > 0).astype(int)
    df_row["vehicle_risk_score"] = df_row["vehicle_age_years"] + df_row["maintenance_overdue_days"]
    df_row["adverse_weather_flag"] = (
        (df_row["storm_warning"] == 1) | (df_row["rainfall_mm"] > rainfall_p75)
    ).astype(int)
    df_row["supplier_risk_index"] = df_row["supplier_previous_delay_count"] / (df_row["supplier_rating"] + 1)
    df_row["warehouse_units_in_use"] = df_row["warehouse_capacity"] * (df_row["warehouse_utilization_pct"] / 100)

    df_row["shipment_priority"] = df_row["shipment_priority"].map(priority_map)
    df_row["customer_tier"] = df_row["customer_tier"].map(tier_map)

    # Manually one-hot encode against the TRAINING-TIME column set.
    # pd.get_dummies on a single row is unsafe here: with drop_first=True it would
    # always treat that row's one and only category as "first" and drop it,
    # silently encoding every input as the reference category regardless of its
    # real value. Setting the matching dummy column to 1 (when it exists) avoids that.
    for col in nominal_cols:
        dummy_col = f"{col}_{raw[col]}"
        if dummy_col in all_columns:
            df_row[dummy_col] = 1
        df_row = df_row.drop(columns=[col])

    df_row = df_row.reindex(columns=all_columns, fill_value=0)
    return df_row[selected_features], df_row


def predict_penalty_exposure(raw_delay_proba: float, full_row: pd.DataFrame):
    """Frequency x severity penalty exposure for one shipment, reusing the notebook's
    prior-correction + mean-matching calibration (built for exactly this single-row case)
    and applying the severity model to the same engineered row used for the delay model."""
    odds = raw_delay_proba * (prior_pi / prior_beta)
    odds_complement = (1 - raw_delay_proba) * ((1 - prior_pi) / (1 - prior_beta))
    p_delay_corrected = odds / (odds + odds_complement)
    p_delay = min(max(p_delay_corrected * mean_match_scale, 0.0), 1.0)

    # severity_all_columns is a strict subset of the delay model's all_columns (it excludes
    # the cost-leakage columns), so the already-engineered full_row can be reused directly.
    X_severity = full_row.reindex(columns=severity_all_columns, fill_value=0)[severity_selected_features]
    predicted_severity = max(float(severity_model.predict(X_severity)[0]), 0.0)

    return {
        "p_delay": p_delay,
        "predicted_severity": predicted_severity,
        "expected_penalty_exposure": p_delay * predicted_severity,
    }


def get_risk_flags(full_row: pd.DataFrame):
    r = full_row.iloc[0]
    flags = []
    if r.get("adverse_weather_flag", 0) == 1:
        flags.append((
            "Adverse weather",
            "A storm warning or heavy rainfall is active on this route. Delay dispatch until "
            "conditions clear, or route through an unaffected region.",
        ))
    if r.get("maintenance_overdue_flag", 0) == 1:
        flags.append((
            "Vehicle maintenance overdue",
            f"This vehicle has {int(r['maintenance_overdue_days'])} day(s) of overdue maintenance. "
            "Service it before dispatch or assign a different vehicle.",
        ))
    if r.get("supplier_risk_index", 0) >= risk_thresholds["supplier_risk_index_p75"]:
        flags.append((
            "High supplier risk",
            "This supplier has a high rate of past delays relative to its rating. Confirm "
            "readiness directly with the supplier or add schedule buffer.",
        ))
    if r.get("route_risk_score", 0) >= risk_thresholds["route_risk_score_p75"]:
        flags.append((
            "High route risk",
            "The selected route has an above-average risk score. Consider an alternate route "
            "or carrier for this shipment.",
        ))
    if r.get("warehouse_utilization_pct", 0) >= risk_thresholds["warehouse_utilization_p75"]:
        flags.append((
            "Warehouse congestion",
            "The origin warehouse is heavily utilized. Consider an off-peak dispatch window "
            "or routing through an alternate warehouse.",
        ))
    if r.get("vehicle_risk_score", 0) >= risk_thresholds["vehicle_risk_score_p75"]:
        flags.append((
            "Aging / high-risk vehicle",
            "This vehicle is old and/or overdue for maintenance. Prioritize a newer vehicle "
            "for high-priority shipments.",
        ))
    return flags


st.title("🚚 Shipment Delay Predictor")
st.caption(
    "Enter shipment parameters to predict delay risk with the trained Gradient Boosting model, "
    "then see the specific risk factors driving that prediction and how to avoid the delay."
)

tab_predict, tab_explore, tab_history = st.tabs(["Predict", "What-if Explorer", "History"])

with tab_predict:
    with st.form("shipment_form"):
        st.subheader("Shipment")
        c1, c2, c3 = st.columns(3)
        shipment_date = c1.date_input("Shipment date", value=date.today())
        shipment_type = c2.selectbox("Shipment type", category_values["shipment_type"])
        shipment_priority = c3.selectbox("Priority", list(priority_map.keys()))
        c1, c2 = st.columns(2)
        shipment_weight_kg = c1.number_input("Weight (kg)", min_value=0.0, value=10000.0)
        shipment_volume_cbm = c2.number_input("Volume (cbm)", min_value=0.01, value=50.0)
        c1, c2, c3 = st.columns(3)
        shipment_color_code = c1.selectbox("Shipment color code", category_values["shipment_color_code"])
        routing_template_version = c2.selectbox("Routing template version", category_values["routing_template_version"])
        operational_cluster_id = c3.selectbox("Operational cluster ID", category_values["operational_cluster_id"])

        st.subheader("Costs")
        c1, c2, c3 = st.columns(3)
        transportation_cost = c1.number_input("Transportation cost", min_value=0.0, value=5000.0)
        warehouse_cost = c2.number_input("Warehouse cost", min_value=0.0, value=1000.0)
        fuel_cost = c3.number_input("Fuel cost", min_value=0.0, value=2000.0)

        st.subheader("Supplier")
        c1, c2, c3 = st.columns(3)
        supplier_rating = c1.number_input("Supplier rating (0-100)", min_value=0.0, max_value=100.0, value=75.0)
        supplier_region = c2.selectbox("Supplier region", category_values["supplier_region"])
        supplier_previous_delay_count = c3.number_input(
            "Supplier's previous delay count", min_value=0, value=1, step=1
        )

        st.subheader("Warehouse")
        c1, c2, c3 = st.columns(3)
        warehouse_capacity = c1.number_input("Warehouse capacity", min_value=1, value=40000, step=1)
        warehouse_utilization_pct = c2.number_input(
            "Warehouse utilization (%)", min_value=0.0, max_value=100.0, value=75.0
        )
        warehouse_processing_time_hrs = c3.number_input("Warehouse processing time (hrs)", min_value=0.0, value=12.0)

        st.subheader("Vehicle & Route")
        c1, c2, c3 = st.columns(3)
        vehicle_type = c1.selectbox("Vehicle type", category_values["vehicle_type"])
        vehicle_age_years = c2.number_input("Vehicle age (years)", min_value=0, value=5, step=1)
        maintenance_overdue_days = c3.number_input("Maintenance overdue (days)", min_value=0, value=0, step=1)
        c1, c2, c3 = st.columns(3)
        route_distance_km = c1.number_input("Route distance (km)", min_value=1, value=1000, step=1)
        estimated_travel_time_hrs = c2.number_input("Estimated travel time (hrs)", min_value=0.1, value=20.0)
        route_risk_score = c3.number_input("Route risk score (0-100)", min_value=0.0, max_value=100.0, value=50.0)

        st.subheader("Weather")
        c1, c2, c3 = st.columns(3)
        temperature = c1.number_input("Temperature (°C)", value=25.0)
        rainfall_mm = c2.number_input("Rainfall (mm)", min_value=0.0, value=10.0)
        storm_warning = c3.selectbox("Storm warning", [0, 1], format_func=lambda x: "Yes" if x else "No")

        st.subheader("Inventory & Customer")
        c1, c2, c3, c4 = st.columns(4)
        inventory_available = c1.number_input("Inventory available", min_value=0, value=300, step=1)
        inventory_shortage_flag = c2.selectbox(
            "Inventory shortage", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )
        customer_tier = c3.selectbox("Customer tier", list(tier_map.keys()))
        customer_region = c4.selectbox("Customer region", category_values["customer_region"])

        submitted = st.form_submit_button("Predict delay risk")

    if submitted:
        raw = dict(
            shipment_date=shipment_date, shipment_type=shipment_type, shipment_priority=shipment_priority,
            shipment_weight_kg=shipment_weight_kg, shipment_volume_cbm=shipment_volume_cbm,
            shipment_color_code=shipment_color_code, routing_template_version=routing_template_version,
            operational_cluster_id=operational_cluster_id,
            transportation_cost=transportation_cost, warehouse_cost=warehouse_cost, fuel_cost=fuel_cost,
            supplier_rating=supplier_rating, supplier_region=supplier_region,
            supplier_previous_delay_count=supplier_previous_delay_count,
            warehouse_capacity=warehouse_capacity, warehouse_utilization_pct=warehouse_utilization_pct,
            warehouse_processing_time_hrs=warehouse_processing_time_hrs,
            vehicle_type=vehicle_type, vehicle_age_years=vehicle_age_years,
            maintenance_overdue_days=maintenance_overdue_days,
            route_distance_km=route_distance_km, estimated_travel_time_hrs=estimated_travel_time_hrs,
            route_risk_score=route_risk_score, temperature=temperature, rainfall_mm=rainfall_mm,
            storm_warning=storm_warning, inventory_available=inventory_available,
            inventory_shortage_flag=inventory_shortage_flag, customer_tier=customer_tier,
            customer_region=customer_region,
        )

        X_input, full_row = build_features(raw)
        proba = model.predict_proba(X_input)[0, 1]
        pred = int(proba >= 0.5)
        flags = get_risk_flags(full_row)
        penalty = predict_penalty_exposure(proba, full_row)
        # Hard two-stage gate for single-shipment display (per Penalty_Risk_Exposure.ipynb's
        # own guidance): a clean $0-or-a-number figure tied to the DELAYED/ON TIME call above,
        # rather than the always-nonzero soft score, which is only meant for portfolio totals.
        predicted_penalty_two_stage = penalty["predicted_severity"] if pred == 1 else 0.0

        st.session_state["last_raw"] = raw
        db.insert_prediction(raw, proba, pred, flags)

        st.divider()
        col_result, col_gauge = st.columns([2, 1])
        with col_result:
            if pred == 1:
                st.error(f"### ⚠️ Predicted: DELAYED\nProbability of delay: **{proba:.1%}**")
            else:
                st.success(f"### ✅ Predicted: ON TIME\nProbability of delay: **{proba:.1%}**")
        with col_gauge:
            st.metric("Delay probability", f"{proba:.1%}")
            st.progress(min(proba, 1.0))

        st.subheader("Financial Impact")
        c1, c2 = st.columns(2)
        c1.metric("Transportation cost (input)", f"${transportation_cost:,.2f}")
        c2.metric("Predicted penalty cost (if delayed)", f"${predicted_penalty_two_stage:,.2f}")

        st.subheader("Risk factors & recommended actions")
        if not flags:
            st.info("No major risk factors detected for this shipment.")
        else:
            for name, rec in flags:
                st.warning(f"**{name}** — {rec}")

with tab_explore:
    st.subheader("What-if Explorer")
    st.caption(
        "Adjust the levers below to see how they change delay risk for the shipment you just "
        "predicted in the Predict tab — this is where to explore options to avoid the delay."
    )

    if "last_raw" not in st.session_state:
        st.info("Submit a prediction in the Predict tab first, then come back here.")
    else:
        base_raw = st.session_state["last_raw"]

        st.markdown("**Delay risk levers**")
        c1, c2 = st.columns(2)
        new_maintenance = c1.number_input(
            "Maintenance overdue (days)", min_value=0,
            value=int(base_raw["maintenance_overdue_days"]), step=1, key="wi_maint",
        )
        new_route_risk = c2.number_input(
            "Route risk score (0-100)", min_value=0.0, max_value=100.0,
            value=float(base_raw["route_risk_score"]), key="wi_route",
        )
        c3, c4 = st.columns(2)
        new_warehouse_util = c3.number_input(
            "Warehouse utilization (%)", min_value=0.0, max_value=100.0,
            value=float(base_raw["warehouse_utilization_pct"]), key="wi_wh",
        )
        new_storm = c4.selectbox(
            "Storm warning", [0, 1], index=int(base_raw["storm_warning"]),
            format_func=lambda x: "Yes" if x else "No", key="wi_storm",
        )
        c5, c6 = st.columns(2)
        new_rainfall = c5.number_input(
            "Rainfall (mm)", min_value=0.0,
            value=float(base_raw["rainfall_mm"]), key="wi_rainfall",
        )
        new_supplier_rating = c6.number_input(
            "Supplier rating (0-100)", min_value=0.0, max_value=100.0,
            value=float(base_raw["supplier_rating"]), key="wi_supplier_rating",
        )
        c7, _ = st.columns(2)
        new_supplier_delays = c7.number_input(
            "Supplier's previous delay count", min_value=0,
            value=int(base_raw["supplier_previous_delay_count"]), step=1, key="wi_supplier_delays",
        )

        st.markdown("**Cost & penalty levers**")
        st.caption(
            "These are the features that most drive the predicted transportation cost efficiency "
            "(delay model) and the predicted penalty severity (severity model)."
        )
        c1, c2 = st.columns(2)
        new_transportation_cost = c1.number_input(
            "Transportation cost", min_value=0.0,
            value=float(base_raw["transportation_cost"]), key="wi_transport_cost",
        )
        new_shipment_weight = c2.number_input(
            "Shipment weight (kg)", min_value=0.0,
            value=float(base_raw["shipment_weight_kg"]), key="wi_weight",
        )
        c3, c4 = st.columns(2)
        new_warehouse_processing = c3.number_input(
            "Warehouse processing time (hrs)", min_value=0.0,
            value=float(base_raw["warehouse_processing_time_hrs"]), key="wi_wh_processing",
        )
        new_inventory = c4.number_input(
            "Inventory available", min_value=0,
            value=int(base_raw["inventory_available"]), step=1, key="wi_inventory",
        )
        c5, _ = st.columns(2)
        new_warehouse_capacity = c5.number_input(
            "Warehouse capacity", min_value=1,
            value=int(base_raw["warehouse_capacity"]), step=1, key="wi_wh_capacity",
        )

        what_if_raw = dict(base_raw)
        what_if_raw["maintenance_overdue_days"] = new_maintenance
        what_if_raw["route_risk_score"] = new_route_risk
        what_if_raw["warehouse_utilization_pct"] = new_warehouse_util
        what_if_raw["storm_warning"] = new_storm
        what_if_raw["rainfall_mm"] = new_rainfall
        what_if_raw["supplier_rating"] = new_supplier_rating
        what_if_raw["supplier_previous_delay_count"] = new_supplier_delays
        what_if_raw["transportation_cost"] = new_transportation_cost
        what_if_raw["shipment_weight_kg"] = new_shipment_weight
        what_if_raw["warehouse_processing_time_hrs"] = new_warehouse_processing
        what_if_raw["inventory_available"] = new_inventory
        what_if_raw["warehouse_capacity"] = new_warehouse_capacity

        X_base, full_row_base = build_features(base_raw)
        X_whatif, full_row_whatif = build_features(what_if_raw)
        base_proba = model.predict_proba(X_base)[0, 1]
        whatif_proba = model.predict_proba(X_whatif)[0, 1]

        base_penalty = predict_penalty_exposure(base_proba, full_row_base)
        whatif_penalty = predict_penalty_exposure(whatif_proba, full_row_whatif)
        base_penalty_two_stage = base_penalty["predicted_severity"] if base_proba >= 0.5 else 0.0
        whatif_penalty_two_stage = whatif_penalty["predicted_severity"] if whatif_proba >= 0.5 else 0.0

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Original delay probability", f"{base_proba:.1%}")
        c2.metric(
            "Adjusted delay probability", f"{whatif_proba:.1%}",
            delta=f"{(whatif_proba - base_proba):+.1%}", delta_color="inverse",
        )

        c1, c2 = st.columns(2)
        c1.metric("Original transportation cost", f"${base_raw['transportation_cost']:,.2f}")
        c2.metric(
            "Adjusted transportation cost", f"${new_transportation_cost:,.2f}",
            delta=f"{(new_transportation_cost - base_raw['transportation_cost']):+,.2f}",
        )

        c1, c2 = st.columns(2)
        c1.metric("Original predicted penalty cost", f"${base_penalty_two_stage:,.2f}")
        c2.metric(
            "Adjusted predicted penalty cost", f"${whatif_penalty_two_stage:,.2f}",
            delta=f"{(whatif_penalty_two_stage - base_penalty_two_stage):+,.2f}", delta_color="inverse",
        )

with tab_history:
    st.subheader("Prediction History")
    st.caption("Every prediction made through the Predict tab is logged to shipment_predictions.db (SQLite).")

    history_df = db.fetch_predictions()

    if history_df.empty:
        st.info("No predictions logged yet — submit one in the Predict tab.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total logged", len(history_df))
        c2.metric("Predicted delayed", int((history_df["predicted_label"] == 1).sum()))
        c3.metric("Predicted on time", int((history_df["predicted_label"] == 0).sum()))

        st.dataframe(history_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="shipment_predictions.csv",
            mime="text/csv",
        )

        with st.expander("⚠️ Clear all history"):
            st.warning("This permanently deletes every logged prediction.")
            if st.button("Delete all history", type="primary"):
                db.clear_predictions()
                st.rerun()
