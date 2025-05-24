import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import truncnorm


st.set_page_config(page_title="OOIP Monte Carlo Simulation", layout="wide")
st.title("OOIP Monte Carlo Simulation")
st.write(
    "This app simulates the Original Oil in Place (OOIP) using Monte Carlo methods based on user-defined parameters. ")


def ooip(luas, ketebalan, sw, phi, ngr, bo, metric=True):
    so = 1 - sw
    if metric:
        # Convert luas from m² to acres (1 m² = 0.000247105 acres) and ketebalan from meters to feet (1 m = 3.28084 ft)
        ooip = 7758 * (luas * 0.000247105) * (ketebalan * 3.28084) * so * phi * ngr / bo
    else:
        ooip = 7758 * luas * ketebalan * so * phi * ngr / bo
    return ooip


st.sidebar.header("Input Parameters")
n_simulations = st.sidebar.number_input("Number of Simulations", min_value=1, max_value=10000000, value=1000, step=100)


st.sidebar.subheader("Area (m²)")
area_min = st.sidebar.number_input("Area Min Value", min_value=0.0, value=1000000.0, step=100000.0)
area_mean = st.sidebar.number_input("Area Mean", min_value=area_min, value=2000000.0, step=100000.0)
area_max = st.sidebar.number_input("Area Max Value", min_value=area_mean, value=3000000.0, step=100000.0)
area_std = st.sidebar.number_input("Area Std Dev", min_value=0.0, value=(area_max - area_min) / 4, step=10000.0)

st.sidebar.subheader("Thickness (m)")
thickness_min = st.sidebar.number_input("Thickness Min Value", min_value=0.0, value=10.0, step=1.0)
thickness_mean = st.sidebar.number_input("Thickness Mean", min_value=thickness_min, value=20.0, step=1.0)
thickness_max = st.sidebar.number_input("Thickness Max Value", min_value=thickness_mean, value=30.0, step=1.0)
thickness_std = st.sidebar.number_input("Thickness Std Dev", min_value=0.0, value=(thickness_max - thickness_min) / 4, step=0.5)

st.sidebar.subheader("Water Saturation (fraction)")
sw_min = st.sidebar.number_input("Sw Min Value", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
sw_mean = st.sidebar.number_input("Sw Mean", min_value=sw_min, max_value=1.0, value=0.2, step=0.01)
sw_max = st.sidebar.number_input("Sw Max Value", min_value=sw_mean, max_value=1.0, value=0.3, step=0.01)
sw_std = st.sidebar.number_input("Sw Std Dev", min_value=0.0, value=(sw_max - sw_min) / 4, step=0.01)

st.sidebar.subheader("Porosity (fraction)")
phi_min = st.sidebar.number_input("Porosity Min Value", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
phi_mean = st.sidebar.number_input("Porosity Mean", min_value=phi_min, max_value=1.0, value=0.2, step=0.01)
phi_max = st.sidebar.number_input("Porosity Max Value", min_value=phi_mean, max_value=1.0, value=0.3, step=0.01)
phi_std = st.sidebar.number_input("Porosity Std Dev", min_value=0.0, value=(phi_max - phi_min) / 4, step=0.01)

st.sidebar.subheader("Net-to-Gross Ratio (fraction)")
ngr_min = st.sidebar.number_input("NGR Min Value", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
ngr_mean = st.sidebar.number_input("NGR Mean", min_value=ngr_min, max_value=1.0, value=0.7, step=0.01)
ngr_max = st.sidebar.number_input("NGR Max Value", min_value=ngr_mean, max_value=1.0, value=0.8, step=0.01)
ngr_std = st.sidebar.number_input("NGR Std Dev", min_value=0.0, value=(ngr_max - ngr_min) / 4, step=0.01)

st.sidebar.subheader("Oil Formation Volume Factor (Bo)")
bo = st.sidebar.number_input("Bo Value", min_value=0.1, value=1.0, step=0.1)


if st.button("Run Monte Carlo Simulation"):
    # Generate random variables using truncated normal distribution
    def get_truncated_normal(mean, sd, low, high, size):
        a = (low - mean) / sd
        b = (high - mean) / sd
        return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

    luas = get_truncated_normal(area_mean, area_std, area_min, area_max, n_simulations).round()
    ketebalan = get_truncated_normal(thickness_mean, thickness_std, thickness_min, thickness_max, n_simulations).round()
    sw = get_truncated_normal(sw_mean, sw_std, sw_min, sw_max, n_simulations).round(2)
    phi = get_truncated_normal(phi_mean, phi_std, phi_min, phi_max, n_simulations).round(2)
    ngr = get_truncated_normal(ngr_mean, ngr_std, ngr_min, ngr_max, n_simulations).round(2)
    bo_array = np.full(n_simulations, bo)


    ooip_results = ooip(luas, ketebalan, sw, phi, ngr, bo_array)


    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean OOIP (STB)", f"{np.mean(ooip_results):,.2f}")
    col2.metric("P10 (90th percentile)", f"{np.percentile(ooip_results, 90):,.2f}")
    col3.metric("P50 (50th percentile)", f"{np.percentile(ooip_results, 50):,.2f}")
    col4.metric("P90 (10th percentile)", f"{np.percentile(ooip_results, 10):,.2f}")
    col5.metric("Std Dev", f"{np.std(ooip_results):,.2f}")


    fig = px.histogram(
        x=ooip_results,
        nbins=30,
        title="OOIP Monte Carlo Simulation Results",
        labels={"x": "OOIP (STB)", "y": "Frequency"},
        template="plotly_white"
    )
    fig.update_traces(marker=dict(line=dict(color="black", width=1)))
    fig.update_layout(showlegend=False, bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Results Table")
    results_df = pd.DataFrame({
        "Area (m²)": luas,
        "Thickness (m)": ketebalan,
        "Water Saturation": sw,
        "Porosity": phi,
        "Net-to-Gross": ngr,
        "Bo": bo_array,
        "OOIP (STB)": ooip_results
    })
    st.dataframe(results_df, use_container_width=True)
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="ooip_simulation_results.csv",
        mime="text/csv"
    )
