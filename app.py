# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 02:58:07 2025

@author: Gabo San
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import BraytonRI as backend
import base64

def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Configuración de la página
tc = 'wide'
st.set_page_config(layout="wide")

# Encabezado institucional
logo_b64 = img_to_base64("logo_uam.png")

# Mostrar logo en la parte superior y centrado
cols = st.columns([1, 2, 1])      # columnas de ancho relativo 1:2:1
cols[1].image("logo_uam.png", width=300)  # la columna del medio muestra el logo

# Encabezado institucional
st.markdown("""
<div style="text-align:center; line-height:1.2;">
  <h2> UNIVERSIDAD AUTÓNOMA METROPOLITANA</h2>
  <p>Proyecto Terminal</p>
  <p><strong>Asesor:</strong> 🎓 Hernando Romero Paredes Rubio &nbsp;|&nbsp; <strong>Alumno:</strong> 🤖 Rolando Gabriel Garza Luna</p>
</div>
""", unsafe_allow_html=True)


st.title("🛠️ Simulación Ciclo Brayton + Cogeneración")

# — Sidebar: carga de bases de datos —
st.sidebar.header("🔄 Datos de Entrada")
mun_file = st.sidebar.file_uploader("Municipios (.xlsx)", type="xlsx")
tur_file = st.sidebar.file_uploader("Turbinas (.csv)", type="csv")

# — Sidebar: parámetros globales —
st.sidebar.header("⚙️ Parámetros Globales")
eta_caldera = st.sidebar.slider("η_caldera", 0.5, 1.0, 0.85, 0.01)
m_dot_water = st.sidebar.slider("ṁ_water (kg/s)", 0.1, 100.0, 2.0, 0.1)
U           = st.sidebar.slider("U (kW/m²·K)", 0.1, 10.0, 0.3, 0.1)
A           = st.sidebar.slider("A (m²)", 1.0, 50.0, 8.0, 1.0)
T_c_in      = st.sidebar.slider("T_C_IN (K)", 250.0, 350.0, 293.15, 1.0)
T_hot_out   = st.sidebar.slider("T_hot_out (K)", 350.0, 500.0, 373.15, 1.0)
T_c_out     = st.sidebar.slider("T_c_out (K)", 300.0, 370.0, 323.15, 1.0)
eta_gen     = st.sidebar.slider("η_gen", 0.5, 1.0, 0.95, 0.01)
PCI         = st.sidebar.slider("PCI (kJ/kg)", 10000, 60000, 50000, 1000)

# — Botón de simulación —
if st.sidebar.button("▶️ Simular"):
    if not mun_file or not tur_file:
        st.sidebar.error("Por favor sube ambos archivos antes de simular.")
    else:
        # 1) Leer datos
        df_mun = pd.read_excel(mun_file)
        df_mun["T1 (K)"]    = df_mun["Temperatura (°C)"] + 273.15
        df_mun["P1 (kPa)"]  = df_mun["Presión (bares)"] * 100.0
        df_tur = pd.read_csv(tur_file)

        # 2) Configurar globals del backend
        backend.eta_caldera = eta_caldera
        backend.M_DOT_WATER = m_dot_water
        backend.U_GLOBAL    = U
        backend.A_GLOBAL    = A
        backend.T_C_IN      = T_c_in
        backend.T_HOT_OUT   = T_hot_out
        backend.T_C_OUT     = T_c_out

        resultados_calc = []
        resultados_est  = []
        resultados_cog  = []
        total = len(df_mun) * len(df_tur)
        progress = st.sidebar.progress(0)
        idx = 0

        # 3) Simular
        for _, m in df_mun.iterrows():
            for _, t in df_tur.iterrows():
                est, calc, cog = backend.simular_ciclo(
                    m["T1 (K)"], m["P1 (kPa)"],
                    t["r_p"],    t["T3 (C)"],
                    t["eta_c"],  t["eta_t"],
                    P_ele       = t["Potencia (kW)"],
                    altitude_m  = m["Altitud (media)"],
                    eta_gen     = eta_gen,
                    PCI         = PCI,
                )
                # etiquetas y metadatos
                est.insert(0, "Municipio", m["Municipio"])
                est.insert(1, "Turbina"  , t["Turbina"])
                calc.update({
                    "Municipio": m["Municipio"],
                    "Turbina":   t["Turbina"],
                    "r_p":       t["r_p"],
                    "T3":        t["T3"],
                    "P_ele etiqueta (kW)": t["Potencia"]
                })
                cog.update({
                    "Municipio": m["Municipio"],
                    "Turbina":   t["Turbina"]
                })
                resultados_est.append(est)
                resultados_calc.append(calc)
                resultados_cog.append(cog)
                idx += 1
                progress.progress(idx/total)

        # 4) Construir DataFrames
        df_calc = pd.DataFrame(resultados_calc)
        df_est  = pd.concat(resultados_est, ignore_index=True)
        df_cog  = pd.DataFrame(resultados_cog)
        st.success("✅ Simulación completada!")

        # 5) Preprocesamiento para gráficas
        df_calc['Altitud (media)'] = df_calc['Municipio'].map(
            df_mun.set_index('Municipio')['Altitud (media)']
        )       
        df_merged = df_calc.merge(
            df_cog[['Municipio','Turbina','Q_rec (kW)']],
            on=['Municipio','Turbina']
        )

        # 6) Mostrar tablas
        with st.expander("📊 Tabla de Cálculos"):
            st.dataframe(df_calc)
        with st.expander("🔢 Tabla de Estados"):
            st.dataframe(df_est)
        with st.expander("♻️ Tabla de Cogeneración"):
            st.dataframe(df_cog)

        # 7) Gráficas en pestañas
        tabs = st.tabs([
            "Pot vs Eficiencia",
            "Qútil vs Qin",
            "Pot vs Qin",
            "SFC vs Pot",
            "η_global vs Altitud",
            "Q_rec vs Altitud",
            "3D Pot-Alt-SFC"  
        ])

        # 7.1 Potencia vs eficiencia de ciclo
        with tabs[0]:
            fig = px.scatter(
                df_calc,
                x='P_elec_corr (kW)',
                y='eta_ciclo (%)',
                color='Turbina',
                labels={
                    'P_elec_corr (kW)': 'Potencia corregida (kW)',
                    'eta_ciclo (%)': 'Eficiencia ciclo (%)'
                },
                title='Potencia vs Eficiencia de Ciclo'
            )
            st.plotly_chart(fig, use_container_width=True)

        # 7.2 Energía térmica útil vs calor suministrado
        with tabs[1]:
            fig = px.scatter(
                df_merged,
                x='Q_input (kW)',
                y='Q_rec (kW)',
                color='Turbina',
                labels={
                    'Q_input (kW)': 'Calor suministrado (kW)',
                    'Q_rec (kW)': 'Energía térmica útil (kW)'
                },
                title='Energía térmica útil vs Calor suministrado'
            )
            st.plotly_chart(fig, use_container_width=True)

        # 7.3 Potencia eléctrica corregida vs calor suministrado
        with tabs[2]:
            fig = px.scatter(
                df_merged,
                x='Q_input (kW)',
                y='P_elec_corr (kW)',
                color='Turbina',
                labels={
                    'Q_input (kW)': 'Calor suministrado (kW)',
                    'P_elec_corr (kW)': 'Potencia corregida (kW)'
                },
                title='Potencia eléctrica vs Calor suministrado'
            )
            st.plotly_chart(fig, use_container_width=True)

        # 7.4 SFC vs Potencia eléctrica
        with tabs[3]:
            fig = px.scatter(
                df_calc,
                x='P_elec_corr (kW)',
                y='SFC (kg/kWh)',
                color='Turbina',
                labels={
                    'P_elec_corr (kW)': 'Potencia corregida (kW)',
                    'SFC (kg/kWh)': 'SFC (kg/kWh)'
                },
                title='SFC vs Potencia eléctrica'
            )
            st.plotly_chart(fig, use_container_width=True)

        # 7.5 Eficiencia global vs Altitud
        with tabs[4]:
            fig = px.scatter(
                df_calc,
                x='Altitud (media)',
                y='eta_global (%)',
                color='Turbina',
                labels={'eta_global (%)':'Eficiencia global (%)'},
                title='Eficiencia global vs Altitud'
            )
            st.plotly_chart(fig, use_container_width=True)

        # 7.6 Calor recuperado vs Altitud
        with tabs[5]:
            fig = px.scatter(
                df_merged,
                x='Altitud (media)',
                y='Q_rec (kW)',
                color='Turbina',
                labels={'Q_rec (kW)':'Calor recuperado (kW)'},
                title='Calor recuperado vs Altitud'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 7.7 Gráfica 3D Potencia vs Altitud vs SFC
        with tabs[6]:
            fig3d = px.scatter_3d(
                df_calc,
                x='Altitud (media)',
                y='P_elec_corr (kW)',
                z='SFC (kg/kWh)',
                color='Turbina',
                hover_data=['Municipio'],
                labels={
                    'Altitud (media)': 'Altitud (m)',
                    'P_elec_corr (kW)': 'Potencia corregida (kW)',
                    'SFC (kg/kWh)': 'SFC (kg/kWh)'
                },
                title='Potencia vs Altitud vs SFC (3D)'
            )
            fig3d.update_layout(width=800, height=600)
            st.plotly_chart(fig3d, use_container_width=True)            

