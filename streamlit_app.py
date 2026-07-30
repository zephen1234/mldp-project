"""Buyer-friendly Streamlit interface for the HDB resale price model."""

from __future__ import annotations

import html
import json
from calendar import month_name

import joblib
import pandas as pd
import streamlit as st

from model_utils import MODEL_FEATURES, PROJECT_DIR, storey_midpoint


MODEL_PATH = PROJECT_DIR / "hdb_price_pipeline.joblib"
METADATA_PATH = PROJECT_DIR / "model_metadata.json"


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
            --warm: #fff5e8;
            --gold: #d88a26;
            --line: #d9e8e3;
            --white: #ffffff;
        }

        .stApp {
            background:
                radial-gradient(circle at 95% 3%, #dff4ed 0, transparent 25rem),
                radial-gradient(circle at 4% 42%, #fff1dd 0, transparent 22rem),
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
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
        }

        .trust-strip {
            display: flex;
            justify-content: center;
            gap: 0.7rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }

        .trust-pill {
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.82);
            border-radius: 999px;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 650;
            padding: 0.42rem 0.75rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(130deg, #083f3a 0%, #08786d 75%);
            border-radius: 28px;
            color: white;
            padding: 2.5rem 2.7rem;
            box-shadow: 0 22px 55px rgba(8, 75, 68, 0.17);
            margin-bottom: 1.5rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            right: -90px;
            top: -115px;
            background: rgba(255, 255, 255, 0.08);
        }

        .hero-kicker {
            color: #bce8df;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .hero h1 {
            color: white;
            font-size: clamp(2.1rem, 5vw, 3.55rem);
            letter-spacing: -0.045em;
            line-height: 1.03;
            margin: 0 0 0.75rem;
            max-width: 720px;
        }

        .hero p {
            color: #e1f5f0;
            font-size: 1.08rem;
            line-height: 1.55;
            max-width: 680px;
            margin: 0;
        }

        .form-intro {
            text-align: center;
            margin: 1.8rem auto 1.1rem;
            max-width: 650px;
        }

        .eyebrow {
            color: var(--brand);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .form-intro h2,
        .section-heading h2 {
            color: var(--ink);
            letter-spacing: -0.025em;
            margin: 0;
        }

        .form-intro p,
        .section-heading p {
            color: var(--muted);
            margin: 0.45rem 0 0;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line) !important;
            border-radius: 24px;
            box-shadow: 0 14px 38px rgba(22, 48, 44, 0.07);
        }

        .step-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--ink);
            font-weight: 750;
            margin: 0.1rem 0 0.65rem;
        }

        .step-number {
            display: inline-grid;
            place-items: center;
            width: 1.7rem;
            height: 1.7rem;
            border-radius: 50%;
            background: var(--mint);
            color: var(--brand);
            font-size: 0.8rem;
            font-weight: 850;
        }

        div[data-testid="stButton"] button {
            min-height: 3.35rem;
            border: 0 !important;
            border-radius: 15px;
            background: var(--brand) !important;
            color: white !important;
            font-size: 1rem;
            font-weight: 760;
            box-shadow: 0 10px 24px rgba(8, 120, 109, 0.22);
        }

        div[data-testid="stButton"] button:hover,
        div[data-testid="stButton"] button:focus,
        div[data-testid="stButton"] button:active {
            background: var(--brand-dark) !important;
            color: white !important;
            transform: translateY(-1px);
        }

        .result-card {
            background: linear-gradient(135deg, #083f3a 0%, #08786d 100%);
            color: white;
            border-radius: 26px;
            padding: 2rem 2.2rem;
            box-shadow: 0 20px 50px rgba(8, 75, 68, 0.19);
            margin: 1.2rem 0 1rem;
        }

        .result-label {
            color: #bce8df;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .result-price {
            font-size: clamp(2.65rem, 7vw, 4.8rem);
            font-weight: 850;
            letter-spacing: -0.055em;
            line-height: 1.03;
            margin: 0.35rem 0 0.55rem;
        }

        .result-property {
            color: #e1f5f0;
            font-size: 0.95rem;
            margin: 0;
        }

        .insight-card {
            height: 100%;
            border: 1px solid var(--line);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.9);
            padding: 1.15rem 1.2rem;
        }

        .insight-label {
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .insight-value {
            color: var(--ink);
            font-size: 1.28rem;
            font-weight: 790;
            line-height: 1.25;
        }

        .insight-note {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
            margin-top: 0.35rem;
        }

        .comparison-good,
        .comparison-warm {
            display: inline-block;
            border-radius: 999px;
            padding: 0.28rem 0.65rem;
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .comparison-good {
            color: #08665d;
            background: #dff5ee;
        }

        .comparison-warm {
            color: #995a0b;
            background: #fff0d8;
        }

        .section-heading {
            margin: 2rem 0 0.85rem;
        }

        .action-card {
            min-height: 150px;
            border: 1px solid var(--line);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.9);
            padding: 1.15rem 1.2rem;
        }

        .action-icon {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }

        .action-title {
            color: var(--ink);
            font-weight: 790;
            margin-bottom: 0.3rem;
        }

        .action-copy {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.48;
        }

        .small-print {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.55;
        }

        .footer {
            color: var(--muted);
            text-align: center;
            font-size: 0.8rem;
            margin-top: 1.6rem;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 0.8rem;
            }

            .hero {
                padding: 1.7rem;
                border-radius: 21px;
            }

            .result-card {
                padding: 1.55rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing your price estimator...")
def load_artifacts() -> tuple[object, dict]:
    """Load the trained model and buyer-interface metadata."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python train_model.py` once."
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def format_currency(value: float) -> str:
    return f"S${value:,.0f}"


def clear_prediction() -> None:
    """Clear the old result whenever a buyer changes an input."""
    st.session_state.pop("prediction_result", None)


def validate_inputs(
    town: str | None,
    flat_type: str | None,
    street_name: str | None,
    flat_model: str | None,
    storey_range: str | None,
) -> list[str]:
    """Return short, buyer-friendly validation messages."""
    required_values = {
        "town": town,
        "street": street_name,
        "flat type": flat_type,
        "flat model": flat_model,
        "storey range": storey_range,
    }
    return [
        label
        for label, value in required_values.items()
        if value is None
    ]


try:
    prediction_model, model_metadata = load_artifacts()
except Exception as error:
    st.error(f"Unable to load the price estimator. {error}")
    st.info("Run `python train_model.py`, then restart Streamlit.")
    st.stop()


latest_year = int(model_metadata["data_year_max"])
latest_month = int(model_metadata.get("data_month_max", 5))
latest_period = f"{month_name[latest_month]} {latest_year}"

st.markdown(
    f"""
    <div class="trust-strip">
        <span class="trust-pill">✓ {model_metadata['dataset_rows']:,} transactions</span>
        <span class="trust-pill">✓ Updated through {latest_period}</span>
        <span class="trust-pill">✓ Ang Mo Kio · Bishan · Toa Payoh</span>
    </div>
    <section class="hero">
        <div class="hero-kicker">HDB resale price guide</div>
        <h1>Know your price range before making an offer.</h1>
        <p>
            Get a quick, data-backed estimate for a resale flat and use it
            as a starting point for your buying decision.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="form-intro">
        <div class="eyebrow">Takes about one minute</div>
        <h2>Tell us about the flat</h2>
        <p>Use the details shown in the property listing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        '<div class="step-title"><span class="step-number">1</span>'
        "Where is the flat?</div>",
        unsafe_allow_html=True,
    )

    location_one, location_two = st.columns(2, gap="large")
    with location_one:
        town = st.selectbox(
            "Town",
            options=model_metadata["towns"],
            index=None,
            placeholder="Choose a town",
            on_change=clear_prediction,
        )

    with location_two:
        street_options = (
            model_metadata["streets_by_town"].get(town, [])
            if town
            else []
        )
        street_name = st.selectbox(
            "Street",
            options=street_options,
            index=None,
            placeholder=(
                "Choose a street" if town else "Select a town first"
            ),
            disabled=town is None,
            on_change=clear_prediction,
        )

    st.write("")
    st.markdown(
        '<div class="step-title"><span class="step-number">2</span>'
        "What type of flat is it?</div>",
        unsafe_allow_html=True,
    )

    type_one, type_two = st.columns(2, gap="large")
    with type_one:
        flat_type = st.selectbox(
            "Flat type",
            options=model_metadata["flat_types"],
            index=None,
            placeholder="Choose a flat type",
            on_change=clear_prediction,
        )

    with type_two:
        flat_model_options = (
            model_metadata["flat_models_by_flat_type"].get(flat_type, [])
            if flat_type
            else []
        )
        flat_model = st.selectbox(
            "Flat model",
            options=flat_model_options,
            index=None,
            placeholder=(
                "Choose a flat model"
                if flat_type
                else "Select a flat type first"
            ),
            disabled=flat_type is None,
            help="Examples include Improved, Model A and New Generation.",
            on_change=clear_prediction,
        )

    st.write("")
    st.markdown(
        '<div class="step-title"><span class="step-number">3</span>'
        "Add the key details</div>",
        unsafe_allow_html=True,
    )

    detail_one, detail_two, detail_three = st.columns(3, gap="medium")
    with detail_one:
        floor_area = st.number_input(
            "Floor area (m²)",
            min_value=float(model_metadata["floor_area_min"]),
            max_value=float(model_metadata["floor_area_max"]),
            value=90.0,
            step=1.0,
            help="Use the floor area stated in the listing.",
            on_change=clear_prediction,
        )

    with detail_two:
        storey_range = st.selectbox(
            "Storey range",
            options=model_metadata["storey_ranges"],
            index=None,
            placeholder="Choose a range",
            on_change=clear_prediction,
        )

    with detail_three:
        lease_years = st.number_input(
            "Remaining lease (years)",
            min_value=float(model_metadata["lease_years_min"]),
            max_value=float(model_metadata["lease_years_max"]),
            value=min(75.0, float(model_metadata["lease_years_max"])),
            step=0.25,
            help="Example: 74 years 6 months is 74.5 years.",
            on_change=clear_prediction,
        )

    st.write("")
    button_left, button_centre, button_right = st.columns([1, 1.35, 1])
    with button_centre:
        estimate_clicked = st.button(
            "Show my price estimate",
            type="primary",
            width="stretch",
        )
    st.caption(
        f"The estimate uses the latest model period ({latest_period}). "
        "It is a guide, not an official valuation."
    )


if estimate_clicked:
    missing_inputs = validate_inputs(
        town=town,
        flat_type=flat_type,
        street_name=street_name,
        flat_model=flat_model,
        storey_range=storey_range,
    )

    if missing_inputs:
        st.error(
            "Please complete: " + ", ".join(missing_inputs) + "."
        )
    else:
        prediction_input = pd.DataFrame(
            [
                {
                    "town": town,
                    "flat_type": flat_type,
                    "street_name": street_name,
                    "flat_model": flat_model,
                    "floor_area_sqm": floor_area,
                    "remaining_lease_years": lease_years,
                    "storey_mid": storey_midpoint(storey_range),
                    "sale_year": latest_year,
                    "sale_month": latest_month,
                }
            ],
            columns=MODEL_FEATURES,
        )

        try:
            predicted_price = float(
                prediction_model.predict(prediction_input)[0]
            )
            median_key = f"{town}|{flat_type}"
            market_median = model_metadata["market_medians"].get(median_key)

            st.session_state["prediction_result"] = {
                "price": predicted_price,
                "market_median": market_median,
                "town": town,
                "street": street_name,
                "flat_type": flat_type,
                "flat_model": flat_model,
                "floor_area": floor_area,
                "storey_range": storey_range,
                "lease_years": lease_years,
            }
        except Exception:
            st.error(
                "The estimate could not be generated. Please check the "
                "details and try again."
            )


prediction_result = st.session_state.get("prediction_result")
if prediction_result is not None:
    predicted_price = prediction_result["price"]
    typical_error = float(model_metadata["metrics"]["test_mae"])
    lower_estimate = max(0, predicted_price - typical_error)
    upper_estimate = predicted_price + typical_error

    property_summary = (
        f"{html.escape(prediction_result['flat_type'])} · "
        f"{prediction_result['floor_area']:.0f} m² · "
        f"{html.escape(prediction_result['street'])}"
    )

    st.markdown(
        f"""
        <section class="result-card">
            <div class="result-label">Estimated resale price</div>
            <div class="result-price">{format_currency(predicted_price)}</div>
            <p class="result-property">{property_summary}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    market_median = prediction_result["market_median"]
    if market_median is None:
        comparison_label = "Market comparison unavailable"
        comparison_value = "Not enough recent comparable data"
        comparison_note = "Use recent nearby transactions as a reference."
        comparison_class = "comparison-warm"
    else:
        difference = predicted_price - market_median
        difference_percentage = difference / market_median * 100
        if abs(difference_percentage) <= 5:
            comparison_label = "Close to recent median"
            comparison_class = "comparison-good"
        elif difference > 0:
            comparison_label = "Above recent median"
            comparison_class = "comparison-warm"
        else:
            comparison_label = "Below recent median"
            comparison_class = "comparison-good"

        comparison_value = (
            f"{format_currency(abs(difference))} "
            f"{'higher' if difference > 0 else 'lower'}"
        )
        comparison_note = (
            f"Compared with the {latest_year} median for "
            f"{prediction_result['flat_type']} flats in "
            f"{prediction_result['town']}."
        )

    insight_one, insight_two = st.columns(2, gap="medium")
    with insight_one:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">Buyer planning range</div>
                <div class="insight-value">
                    {format_currency(lower_estimate)} –
                    {format_currency(upper_estimate)}
                </div>
                <div class="insight-note">
                    A practical guide using the model's typical test error.
                    It is not a guaranteed sale-price range.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with insight_two:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="{comparison_class}">{comparison_label}</div>
                <div class="insight-value">{comparison_value}</div>
                <div class="insight-note">{comparison_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-heading">
            <div class="eyebrow">Before you make an offer</div>
            <h2>Three useful next steps</h2>
            <p>The model is a starting point. Complete these checks before deciding.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_one, action_two, action_three = st.columns(3, gap="medium")
    with action_one:
        st.markdown(
            """
            <div class="action-card">
                <div class="action-icon">🔎</div>
                <div class="action-title">Check recent transactions</div>
                <div class="action-copy">
                    Compare similar flats on the same street and nearby blocks,
                    especially recent sales with a similar floor area.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_two:
        st.markdown(
            """
            <div class="action-card">
                <div class="action-icon">💰</div>
                <div class="action-title">Confirm your full budget</div>
                <div class="action-copy">
                    Check financing, valuation, renovation and transaction
                    costs before deciding what you can comfortably offer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_three:
        st.markdown(
            """
            <div class="action-card">
                <div class="action-icon">🏡</div>
                <div class="action-title">Inspect what data misses</div>
                <div class="action-copy">
                    Consider condition, orientation, noise, view and nearby
                    amenities because these are not included in the model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("How reliable is this estimate?"):
        reliability_one, reliability_two = st.columns(2)
        with reliability_one:
            st.markdown("**What supports the estimate**")
            st.write(
                f"Trained on {model_metadata['dataset_rows']:,} cleaned "
                f"transactions from {model_metadata['data_year_min']} to "
                f"{latest_period}."
            )
            st.write(
                "The model's typical test error is approximately "
                f"{format_currency(typical_error)}."
            )

        with reliability_two:
            st.markdown("**What the model cannot see**")
            st.write(
                "Renovation quality, unit condition, exact orientation, "
                "noise, views, policy changes and interest rates."
            )
            st.write(
                "Use the result for early research, not as an official "
                "valuation or financial recommendation."
            )

        st.caption(
            "Technical reference: "
            f"test R² {model_metadata['metrics']['test_r2']:.4f} · "
            f"test RMSE "
            f"{format_currency(model_metadata['metrics']['test_rmse'])}"
        )

st.markdown(
    """
    <div class="footer">
        HDB Price Compass · A student machine-learning decision-support project
    </div>
    """,
    unsafe_allow_html=True,
)
