import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Predicción Mn EA - XGBoost", layout="wide")
st.title("🔬 Predicción de Mn EA (XGBoost)")

# Cargar modelos
@st.cache_resource
def load_model():
    modelo = pickle.load(open('modelo_xgb_mn.pkl', 'rb'))
    scaler = pickle.load(open('scaler_xgb_mn.pkl', 'rb'))
    return modelo, scaler

modelo, scaler = load_model()

features = ['T de EA (°C)', 'Zn Sol Pura (g/L)', 'Mn Sol Pura', 'Fe (mg/L)', 
            'Sb (mg/L)', 'Cu (mg/L)', 'Densidad (g/L)', 'Acidez (%)', 
            'Horas de depósito', 'Peso Depósito (kg)']

RANGO_MN_SOL_PURA = (5, 8)
RANGO_MN_EA = (12, 15)

st.sidebar.header("Parámetros Operativos")

with st.sidebar:
    t_ea = st.number_input("T de EA (°C)", 25.0, 45.0, 37.5)
    zn_sol = st.number_input("Zn Sol Pura (g/L)", 100.0, 200.0, 162.0)
    mn_sol = st.number_input("Mn Sol Pura (g/L)", 5.0, 9.0, 6.5)
    fe = st.number_input("Fe (mg/L)", 0.0, 10.0, 2.5)
    sb = st.number_input("Sb (mg/L)", 0.0, 0.1, 0.012, step=0.001, format="%.3f")
    cu = st.number_input("Cu (mg/L)", 0.0, 1.0, 0.14, step=0.001, format="%.3f")
    densidad = st.number_input("Densidad (g/L)", 1250.0, 1350.0, 1302.0)
    acidez = st.number_input("Acidez (%)", 150.0, 220.0, 188.0)
    horas = st.number_input("Horas de depósito", 20.0, 70.0, 44.0)
    peso = st.number_input("Peso Depósito (kg)", 40.0, 110.0, 73.0)

if st.button("🔮 Calcular Predicción", type="primary"):
    input_data = pd.DataFrame([[t_ea, zn_sol, mn_sol, fe, sb, cu, densidad, acidez, horas, peso]], 
                              columns=features)
    input_scaled = scaler.transform(input_data)
    mn_pred = modelo.predict(input_scaled)[0]

    st.success(f"**Mn EA Predicho: {mn_pred:.3f} g/L**")

    col1, col2 = st.columns(2)
    with col1:
        if RANGO_MN_SOL_PURA[0] <= mn_sol <= RANGO_MN_SOL_PURA[1]:
            st.info("✅ Mn Sol Pura DENTRO del rango (5-8 g/L)")
        else:
            st.warning("⚠️ Mn Sol Pura FUERA del rango")
    with col2:
        if RANGO_MN_EA[0] <= mn_pred <= RANGO_MN_EA[1]:
            st.info("✅ Mn EA DENTRO del rango objetivo (12-15 g/L)")
        else:
            st.warning("⚠️ Mn EA FUERA del rango objetivo")

st.caption("Modelo XGBoost entrenado para la tesis")
