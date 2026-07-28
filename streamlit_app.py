"""Streamlit interface for the HDB resale price prediction model."""

from __future__ import annotations

import json
from calendar import month_name
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model_utils import MODEL_FEATURES, PROJECT_DIR, storey_midpoint


MODEL_PATH = PROJECT_DIR / "hdb_price_pipeline.joblib"
METADATA_PATH = PROJECT_DIR / "model_metadata.json"

MONTH_OPTIONS = {
    month_name[month_number]: month_number
    for month_number in range(1, 13)
}


st.set_page_config(
    page_title="HDB Price Compass",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #12332f;
            --muted: #56706c;
            --brand: #0b766d;
            --brand-dark: #075a53;
            --accent: #f4a261;
            --surface: #f5faf8;
            --line: #dce9e5;
        }

        .stApp {
            background:
                radial-gradient(circle at 92% 8%, #dff3ed 0, transparent 26rem),
                linear-gradient(180deg, #ffffff 0%, #f7fbfa 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.4rem;
            padding-bottom: 4rem;
        }

        .hero {
            background: linear-gradient(125deg, #073f3a 0%, #0b766d 72%);
            color: white;
            border-radius: 26px;
            padding: 2.2rem 2.4rem;
            box-shadow: 0 22px 55px rgba(7, 63, 58, 0.18);
            margin-bottom: 1.5rem;
        }

        .hero-kicker {
            color: #a8ded4;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .hero h1 {
            color: white;
            font-size: clamp(2.1rem, 5vw, 3.8rem);
            line-height: 1.02;
            letter-spacing: -0.045em;
            margin: 0 0 0.8rem 0;
        }

        .hero p {
            color: #dff4ef;
            font-size: 1.05rem;
            max-width: 760px;
            margin: 0;
        }

        .section-label {
            color: var(--brand);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin: 0.2rem 0 0.3rem;
        }

        .result-card {
            background: linear-gradient(135deg, #ffffff 0%, #edf8f5 100%);
            border: 1px solid #cfe5df;
            border-radius: 24px;
            padding: 1.7rem 1.8rem;
            box-shadow: 0 16px 40px rgba(18, 51, 47, 0.09);
            margin: 0.5rem 0 1.2rem;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .result-price {
            color: var(--ink);
            font-size: clamp(2.3rem, 6vw, 4.2rem);
            font-weight: 820;
            letter-spacing: -0.045em;
            line-height: 1.08;
            margin: 0.25rem 0 0.5rem;
        }

        .result-note {
            color: var(--muted);
            margin: 0;
        }

        .empty-state {
            border: 1px dashed #b9d5ce;
            border-radius: 22px;
            padding: 2rem;
            background: rgba(245, 250, 248, 0.78);
            color: var(--muted);
            text-align: center;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
        }

        div[data-testid="stButton"] button {
            min-height: 3.2rem;
            border-radius: 14px;
            border: 0 !important;
            background: var(--brand) !important;
            color: white !important;
            font-weight: 750;
            box-shadow: 0 9px 20px rgba(11, 118, 109, 0.2);
        }

        div[data-testid="stButton"] button:hover,
        div[data-testid="stButton"] button:focus,
        div[data-testid="stButton"] button:active {
            background: var(--brand-dark) !important;
            color: white !important;
        }

        [data-testid="stSidebar"] {
            background: #eef7f4;
            border-right: 1px solid var(--line);
        }

        .small-print {
            color: var(--muted);
            font-size: 0.83rem;
            line-height: 1.55;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 1.1rem;
            }
            .hero {
                padding: 1.6rem;
                border-radius: 20px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the valuation model...")
def load_artifacts() -> tuple[object, dict]:
    """Load the pre-trained pipeline and its user-interface metadata."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python train_model.py` once, "
            "then restart the app."
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def format_currency(value: float) -> str:
    return f"S${value:,.0f}"


def format_signed_currency(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}S${abs(value):,.0f}"


def clear_prediction() -> None:
    """Prevent an old result from being shown after inputs change."""
    st.session_state.pop("prediction_result", None)


def validate_inputs(
    town: str | None,
    flat_type: str | None,
    street_name: str | None,
    flat_model: str | None,
    storey_range: str | None,
    floor_area: float,
    lease_years: float,
    metadata: dict,
) -> list[str]:
    """Return clear user-facing validation messages."""
    required_values = {
        "town": town,
        "flat type": flat_type,
        "street": street_name,
        "flat model": flat_model,
        "storey range": storey_range,
    }
    errors = [
        f"Please select a {label}."
        for label, value in required_values.items()
        if value is None
    ]

    if town and street_name:
        valid_streets = metadata["streets_by_town"].get(town, [])
        if street_name not in valid_streets:
            errors.append("The selected street does not belong to this town.")

    if flat_type and flat_model:
        valid_models = metadata["flat_models_by_flat_type"].get(
            flat_type, []
        )
        if flat_model not in valid_models:
            errors.append(
                "The selected flat model is not available for this flat type."
            )

    if not metadata["floor_area_min"] <= floor_area <= metadata[
        "floor_area_max"
    ]:
        errors.append(
            "Floor area must stay within the observed training range of "
            f"{metadata['floor_area_min']:.0f} to "
            f"{metadata['floor_area_max']:.0f} m²."
        )

    if not metadata["lease_years_min"] <= lease_years <= metadata[
        "lease_years_max"
    ]:
        errors.append(
            "Remaining lease must stay within the observed training range of "
            f"{metadata['lease_years_min']:.1f} to "
            f"{metadata['lease_years_max']:.1f} years."
        )

    return errors


try:
    prediction_model, model_metadata = load_artifacts()
except Exception as error:
    st.error(f"Unable to load the prediction model. {error}")
    st.info(
        "Open a terminal in this project folder and run "
        "`python train_model.py`, then start Streamlit again."
    )
    st.stop()


with st.sidebar:
    st.markdown("### About this estimator")
    st.write(
        "A decision-support tool for buyers and sellers comparing HDB resale "
        "flats in three mature estates."
    )
    st.markdown("---")
    st.markdown("**Model coverage**")
    st.caption("Ang Mo Kio · Bishan · Toa Payoh")
    st.markdown("**Historical data**")
    st.caption(
        f"{model_metadata['data_year_min']} to "
        f"{model_metadata['data_year_max']} · "
        f"{model_metadata['dataset_rows']:,} cleaned transactions"
    )
    st.markdown("**Model performance**")
    st.caption(
        f"Test MAE {format_currency(model_metadata['metrics']['test_mae'])} "
        f"· R² {model_metadata['metrics']['test_r2']:.4f}"
    )
    st.markdown("---")
    st.markdown(
        '<p class="small-print">This estimate supports early price research. '
        "It is not an official HDB valuation and does not include renovation "
        "condition, unit orientation, amenities, policy changes or interest "
        "rates.</p>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Machine-learning decision support</div>
        <h1>HDB Price Compass</h1>
        <p>
            Explore a data-backed resale price estimate for homes in
            Ang Mo Kio, Bishan and Toa Payoh.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Property details</div>', unsafe_allow_html=True)
st.subheader("Tell us about the flat")
st.caption(
    "Choose every required field, then select **Estimate resale price**. "
    "Changing any input clears the previous result."
)

location_column, property_column = st.columns(2, gap="large")

with location_column:
    town = st.selectbox(
        "Town *",
        options=model_metadata["towns"],
        index=None,
        placeholder="Choose a town",
        on_change=clear_prediction,
    )

    street_options = (
        model_metadata["streets_by_town"].get(town, []) if town else []
    )
    street_name = st.selectbox(
        "Street *",
        options=street_options,
        index=None,
        placeholder=(
            "Choose a street" if town else "Select a town first"
        ),
        disabled=town is None,
        on_change=clear_prediction,
    )

    storey_range = st.selectbox(
        "Storey range *",
        options=model_metadata["storey_ranges"],
        index=None,
        placeholder="Choose a storey range",
        on_change=clear_prediction,
    )

    sale_year = st.selectbox(
        "Transaction year",
        options=list(
            range(
                model_metadata["data_year_max"],
                model_metadata["data_year_min"] - 1,
                -1,
            )
        ),
        index=0,
        help="Predictions are limited to years represented in the training data.",
        on_change=clear_prediction,
    )

    sale_month_name = st.selectbox(
        "Transaction month",
        options=list(MONTH_OPTIONS),
        index=0,
        on_change=clear_prediction,
    )

with property_column:
    flat_type = st.selectbox(
        "Flat type *",
        options=model_metadata["flat_types"],
        index=None,
        placeholder="Choose a flat type",
        on_change=clear_prediction,
    )

    flat_model_options = (
        model_metadata["flat_models_by_flat_type"].get(flat_type, [])
        if flat_type
        else []
    )
    flat_model = st.selectbox(
        "Flat model *",
        options=flat_model_options,
        index=None,
        placeholder=(
            "Choose a flat model"
            if flat_type
            else "Select a flat type first"
        ),
        disabled=flat_type is None,
        on_change=clear_prediction,
    )

    floor_area = st.number_input(
        "Floor area (m²)",
        min_value=float(model_metadata["floor_area_min"]),
        max_value=float(model_metadata["floor_area_max"]),
        value=90.0,
        step=1.0,
        help="Enter the floor area stated in the property listing.",
        on_change=clear_prediction,
    )

    lease_years = st.number_input(
        "Remaining lease (years)",
        min_value=float(model_metadata["lease_years_min"]),
        max_value=float(model_metadata["lease_years_max"]),
        value=min(75.0, float(model_metadata["lease_years_max"])),
        step=0.25,
        help="Example: 74 years and 6 months is 74.5 years.",
        on_change=clear_prediction,
    )

st.write("")
button_column, helper_column = st.columns([1, 2], vertical_alignment="center")
with button_column:
    estimate_clicked = st.button(
        "Estimate resale price",
        type="primary",
        width="stretch",
    )
with helper_column:
    st.caption(
        "Required selections are marked with *. Your information is processed "
        "only inside this prediction session."
    )

if estimate_clicked:
    validation_errors = validate_inputs(
        town=town,
        flat_type=flat_type,
        street_name=street_name,
        flat_model=flat_model,
        storey_range=storey_range,
        floor_area=floor_area,
        lease_years=lease_years,
        metadata=model_metadata,
    )

    if validation_errors:
        for message in validation_errors:
            st.error(message)
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
                    "sale_year": sale_year,
                    "sale_month": MONTH_OPTIONS[sale_month_name],
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
                "input": prediction_input.iloc[0].to_dict(),
                "storey_range": storey_range,
                "sale_month_name": sale_month_name,
            }
        except Exception:
            st.error(
                "The estimate could not be generated. Please review the "
                "inputs and try again."
            )


st.divider()
st.markdown('<div class="section-label">Your estimate</div>', unsafe_allow_html=True)

prediction_result = st.session_state.get("prediction_result")
if prediction_result is None:
    st.markdown(
        """
        <div class="empty-state">
            <strong>Your result will appear here.</strong><br>
            Complete the required property details and request an estimate.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    predicted_price = prediction_result["price"]
    test_rmse = model_metadata["metrics"]["test_rmse"]
    lower_estimate = max(0, predicted_price - test_rmse)
    upper_estimate = predicted_price + test_rmse

    st.markdown(
        f"""
        <section class="result-card">
            <div class="result-label">Estimated resale price</div>
            <div class="result-price">{format_currency(predicted_price)}</div>
            <p class="result-note">
                A practical uncertainty guide using the model's test RMSE is
                {format_currency(lower_estimate)} to
                {format_currency(upper_estimate)}.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_one, metric_two, metric_three = st.columns(3)
    with metric_one:
        st.metric("Model test MAE", format_currency(model_metadata["metrics"]["test_mae"]))
    with metric_two:
        st.metric("Model test R²", f"{model_metadata['metrics']['test_r2']:.2%}")
    with metric_three:
        market_median = prediction_result["market_median"]
        if market_median is None:
            st.metric("Recent segment median", "Not available")
        else:
            st.metric(
                "Recent segment median",
                format_currency(market_median),
                delta=format_signed_currency(
                    predicted_price - market_median
                ),
                help=(
                    "Median transaction price for the selected town and flat "
                    f"type in {model_metadata['data_year_max']}."
                ),
            )

    if prediction_result["market_median"] is not None:
        comparison_data = pd.DataFrame(
            {
                "Price (S$)": [
                    predicted_price,
                    prediction_result["market_median"],
                ]
            },
            index=["Your estimate", "Recent segment median"],
        )
        st.bar_chart(comparison_data, color="#0b766d")

    with st.expander("Review the inputs used for this estimate"):
        input_values = prediction_result["input"]
        summary = pd.DataFrame(
            {
                "Property detail": [
                    "Town",
                    "Street",
                    "Flat type",
                    "Flat model",
                    "Floor area",
                    "Storey range",
                    "Remaining lease",
                    "Transaction period",
                ],
                "Selected value": [
                    input_values["town"],
                    input_values["street_name"],
                    input_values["flat_type"],
                    input_values["flat_model"],
                    f"{input_values['floor_area_sqm']:.0f} m²",
                    prediction_result["storey_range"],
                    f"{input_values['remaining_lease_years']:.2f} years",
                    (
                        f"{prediction_result['sale_month_name']} "
                        f"{int(input_values['sale_year'])}"
                    ),
                ],
            }
        )
        st.dataframe(summary, hide_index=True, width="stretch")

st.markdown(
    """
    <p class="small-print" style="margin-top: 1.2rem;">
        Use this estimate as an early reference only. Compare recent listings,
        transaction records and professional valuation advice before making a
        financial decision.
    </p>
    """,
    unsafe_allow_html=True,
)
