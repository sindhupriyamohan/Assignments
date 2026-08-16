"""SQLite persistence for shipment delay predictions made through the app.

Opens a fresh connection per operation rather than holding one open across
Streamlit reruns — SQLite connections aren't safe to share across threads,
and Streamlit can service a session from a different thread on each rerun.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "shipment_predictions.db"

RAW_COLUMNS = [
    "shipment_date", "shipment_type", "shipment_priority", "shipment_weight_kg",
    "shipment_volume_cbm", "supplier_rating", "supplier_region",
    "supplier_previous_delay_count", "warehouse_capacity", "warehouse_utilization_pct",
    "warehouse_processing_time_hrs", "vehicle_type", "vehicle_age_years",
    "maintenance_overdue_days", "route_distance_km", "estimated_travel_time_hrs",
    "route_risk_score", "temperature", "rainfall_mm", "storm_warning",
    "inventory_available", "inventory_shortage_flag", "customer_tier",
    "customer_region", "transportation_cost", "warehouse_cost", "fuel_cost",
    "shipment_color_code", "routing_template_version", "operational_cluster_id",
]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        raw_columns_sql = ",\n".join(f'"{col}" TEXT' for col in RAW_COLUMNS)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                {raw_columns_sql},
                predicted_label INTEGER NOT NULL,
                delay_probability REAL NOT NULL,
                risk_flags TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_prediction(raw: dict, proba: float, pred: int, flags: list) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        columns = ["created_at"] + RAW_COLUMNS + ["predicted_label", "delay_probability", "risk_flags"]
        values = (
            [datetime.now().isoformat(timespec="seconds")]
            + [str(raw[col]) for col in RAW_COLUMNS]
            + [pred, float(proba), "; ".join(name for name, _ in flags)]
        )
        col_sql = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        conn.execute(f'INSERT INTO predictions ({col_sql}) VALUES ({placeholders})', values)
        conn.commit()
    finally:
        conn.close()


def fetch_predictions(limit: int = 200) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )
    finally:
        conn.close()


def count_predictions() -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()


def clear_predictions() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    finally:
        conn.close()
