from __future__ import annotations

import html
import math
from calendar import month_name
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("model.pkl")


st.set_page_config(
    page_title="HDB Price Compass",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #16302c;
            --muted: #617873;
            --brand: #08786d;
            --brand-dark: #075f57;
            --mint: #e8f6f1;
            --warm: #fff3df;
            --line: #d9e8e3;
        }

        .stApp {
            background:
                radial-gradient(circle at 95% 4%, #dff4ed 0, transparent 25rem),
                radial-gradient(circle at 4% 50%, #fff1dd 0, transparent 22rem),
                #fbfdfc;
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none;
        }

        .block-container {
            max-width: 1040px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        .trust-strip {
            display: flex;
            justify-content: center;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin-bottom: 0.9rem;
        }

        .trust-pill {
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.88);
            border-radius: 999px;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 650;
            padding: 0.42rem 0.72rem;
        }

        .hero {
            overflow: hidden;
            background: linear-gradient(130deg, #083f3a 0%, #08786d 78%);
            border-radius: 26px;
            color: white;
            padding: 2.35rem 2.55rem;
            box-shadow: 0 20px 48px rgba(8, 75, 68, 0.17);
            margin-bottom: 1.45rem;
        }

        .hero-kicker,
        .eyebrow,
        .result-label {
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero-kicker,
        .result-label {
            color: #bce8df;
        }

        .hero h1 {
            color: white;
            font-size: clamp(2.1rem, 5vw, 3.45rem);
            letter-spacing: -0.045em;
            line-height: 1.04;
            margin: 0.55rem 0 0.7rem;
            max-width: 720px;
        }

        .hero p {
            color: #e1f5f0;
            font-size: 1.05rem;
            line-height: 1.55;
            max-width: 680px;
            margin: 0;
        }

        .form-intro {
            text-align: center;
            margin: 1.6rem auto 1rem;
        }

        .eyebrow {
            color: var(--brand);
        }

        .form-intro h2 {
            color: var(--ink);
            letter-spacing: -0.025em;
            margin: 0.3rem 0 0;
        }

        .form-intro p {
            color: var(--muted);
            margin: 0.4rem 0 0;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--line) !important;
            border-radius: 22px;
            box-shadow: 0 12px 32px rgba(22, 48, 44, 0.07);
        }

        div[data-testid="stButton"] button {
            min-height: 3.25rem;
            border: 0 !important;
            border-radius: 15px;
            background: var(--brand) !important;
            color: white !important;
            font-size: 1rem;
            font-weight: 750;
            box-shadow: 0 9px 22px rgba(8, 120, 109, 0.22);
        }

        div[data-testid="stButton"] button:hover {
            background: var(--brand-dark) !important;
        }

        .result-card {
            background: linear-gradient(135deg, #083f3a 0%, #08786d 100%);
            color: white;
            border-radius: 25px;
            padding: 1.9rem 2.1rem;
            box-shadow: 0 18px 45px rgba(8, 75, 68, 0.19);
            margin: 1.2rem 0 1rem;
        }

        .result-price {
            font-size: clamp(2.7rem, 7vw, 4.7rem);
            font-weight: 850;
            letter-spacing: -0.055em;
            line-height: 1.03;
            margin: 0.3rem 0 0.5rem;
        }

        .result-property {
            color: #e1f5f0;
            margin: 0;
        }

        .footer {
            color: var(--muted);
            text-align: center;
            font-size: 0.8rem;
            margin-top: 1.5rem;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 0.7rem;
            }

            .hero,
            .result-card {
                padding: 1.55rem;
                border-radius: 20px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# 1. Load the trained model, matching the lecture deployment workflow.
@st.cache_resource(show_spinner="Preparing your price estimator...")
def load_model() -> dict:
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError("model.pkl is missing or empty.")

    bundle = joblib.load(MODEL_PATH)
    required_keys = {
        "model",
        "feature_names",
        "categories",
        "categorical_features",
        "model_features",
        "metadata",
    }
    if not required_keys.issubset(bundle):
        raise ValueError("model.pkl does not contain the required model data.")
    return bundle


def format_currency(value: float) -> str:
    return f"S${value:,.0f}"


def storey_midpoint(storey_range: str) -> float:
    lower, _, upper = storey_range.split()
    return (float(lower) + float(upper)) / 2


def prepare_input(raw_input: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Apply the same pandas categorical conversion used during training."""
    processed = raw_input.copy()

    for column in bundle["categorical_features"]:
        processed[column] = pd.Categorical(
            processed[column].astype(str),
            categories=bundle["categories"][column],
        )

    processed = pd.get_dummies(
        processed,
        columns=bundle["categorical_features"],
        drop_first=True,
        dtype=float,
    )
    return processed.reindex(
        columns=bundle["feature_names"],
        fill_value=0.0,
    )


def clear_prediction() -> None:
    st.session_state.pop("prediction", None)


try:
    model_bundle = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

metadata = model_bundle["metadata"]
latest_year = int(metadata["data_year_max"])
latest_month = int(metadata["data_month_max"])
latest_period = f"{month_name[latest_month]} {latest_year}"

st.markdown(
    f"""
    <div class="trust-strip">
        <span class="trust-pill">✓ {metadata['dataset_rows']:,} transactions</span>
        <span class="trust-pill">✓ Updated through {latest_period}</span>
        <span class="trust-pill">✓ Ang Mo Kio · Bishan · Toa Payoh</span>
    </div>
    <section class="hero">
        <div class="hero-kicker">HDB resale price guide</div>
        <h1>Know your price range before making an offer.</h1>
        <p>
            Get a quick, data-backed estimate and use it as a starting
            point when comparing resale flats.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="form-intro">
        <div class="eyebrow">Property details</div>
        <h2>Tell us about the flat</h2>
        <p>Use the information shown in the property listing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Collect user inputs.
with st.container(border=True):
    location_left, location_right = st.columns(2, gap="large")
    with location_left:
        town = st.selectbox(
            "Town *",
            metadata["towns"],
            index=None,
            placeholder="Choose a town",
            on_change=clear_prediction,
        )

    with location_right:
        streets = metadata["streets_by_town"].get(town, []) if town else []
        street = st.selectbox(
            "Street *",
            streets,
            index=None,
            placeholder="Choose a street" if town else "Select a town first",
            disabled=town is None,
            on_change=clear_prediction,
        )

    type_left, type_right = st.columns(2, gap="large")
    with type_left:
        flat_type = st.selectbox(
            "Flat type *",
            metadata["flat_types"],
            index=None,
            placeholder="Choose a flat type",
            on_change=clear_prediction,
        )

    with type_right:
        flat_models = (
            metadata["flat_models_by_flat_type"].get(flat_type, [])
            if flat_type
            else []
        )
        flat_model = st.selectbox(
            "Flat model *",
            flat_models,
            index=None,
            placeholder=(
                "Choose a flat model"
                if flat_type
                else "Select a flat type first"
            ),
            disabled=flat_type is None,
            on_change=clear_prediction,
        )

    detail_left, detail_middle, detail_right = st.columns(3, gap="medium")
    with detail_left:
        floor_area = st.number_input(
            "Floor area (m²)",
            min_value=float(metadata["floor_area_min"]),
            max_value=float(metadata["floor_area_max"]),
            value=90.0,
            step=1.0,
            help="Enter the floor area shown in the property listing.",
            on_change=clear_prediction,
        )

    with detail_middle:
        storey_range = st.selectbox(
            "Storey range *",
            metadata["storey_ranges"],
            index=None,
            placeholder="Choose a range",
            on_change=clear_prediction,
        )

    with detail_right:
        lease_years = st.number_input(
            "Remaining lease (years)",
            min_value=float(metadata["lease_years_min"]),
            max_value=float(metadata["lease_years_max"]),
            value=min(75.0, float(metadata["lease_years_max"])),
            step=0.25,
            help="Enter the remaining lease shown in the property listing.",
            on_change=clear_prediction,
        )

    button_left, button_middle, button_right = st.columns([1, 1.35, 1])
    with button_middle:
        predict_clicked = st.button(
            "Show my price estimate",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        f"Uses the latest model period ({latest_period}). "
        "Required selections are marked with *."
    )


if predict_clicked:
    required_values = {
        "town": town,
        "street": street,
        "flat type": flat_type,
        "flat model": flat_model,
        "storey range": storey_range,
    }
    missing = [
        label for label, value in required_values.items() if value is None
    ]

    if missing:
        st.error("Please complete: " + ", ".join(missing) + ".")
    else:
        # 3. Convert the selected values into a one-row DataFrame.
        raw_input = pd.DataFrame(
            [
                {
                    "town": town,
                    "flat_type": flat_type,
                    "street_name": street,
                    "flat_model": flat_model,
                    "floor_area_sqm": floor_area,
                    "remaining_lease_years": lease_years,
                    "storey_mid": storey_midpoint(storey_range),
                    "sale_year": latest_year,
                    "sale_month": latest_month,
                }
            ],
            columns=model_bundle["model_features"],
        )

        try:
            # 4. Apply the training columns, then 5. predict with model.pkl.
            processed_input = prepare_input(raw_input, model_bundle)
            predicted_price = float(
                model_bundle["model"].predict(processed_input)[0]
            )

            if not math.isfinite(predicted_price) or predicted_price <= 0:
                raise ValueError("The model returned an invalid price.")
        except Exception:
            st.session_state.pop("prediction", None)
            st.error(
                "We could not calculate an estimate for these details. "
                "Please check your selections and try again."
            )
        else:
            st.session_state["prediction"] = {
                "price": predicted_price,
                "town": town,
                "street": street,
                "flat_type": flat_type,
                "floor_area": floor_area,
            }


result = st.session_state.get("prediction")
if result:
    price = float(result["price"])

    st.markdown(
        f"""
        <section class="result-card">
            <div class="result-label">Estimated resale price</div>
            <div class="result-price">{format_currency(price)}</div>
            <p class="result-property">
                {html.escape(result['flat_type'])} ·
                {result['floor_area']:.0f} m² ·
                {html.escape(result['street'])}
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
        HDB Price Compass · Student machine-learning decision-support project
    </div>
    """,
    unsafe_allow_html=True,
)
