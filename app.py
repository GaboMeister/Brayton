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
import Brayton as backend  

# Página
st.set_page_config(layout="wide")
logo_path = "logo_uam.png"

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

# Sidebar: inputs
st.sidebar.header("🔄 Datos de Entrada")
mun_file = st.sidebar.file_uploader("Municipios (.xlsx)", type="xlsx")
tur_file = st.sidebar.file_uploader("Turbinas (.csv)", type="csv")

st.sidebar.header("⚙️ Parámetros Globales")
eta_caldera = st.sidebar.slider("η_caldera", 0.5, 1.0, 0.85, 0.01)
m_dot_water = st.sidebar.slider("ṁ_water (kg/s)", 0.1, 100.0, 2.0, 0.1)
U           = st.sidebar.slider("U (kW/m²·K)", 0.1, 2.0, 0.3, 0.05)
A           = st.sidebar.slider("A (m²)", 1.0, 50.0, 8.0, 1.0)
T_c_in      = st.sidebar.slider("T_C_IN (K)", 250.0, 350.0, 293.15, 1.0)
eta_gen     = st.sidebar.slider("η_gen", 0.5, 1.0, 0.95, 0.01)
PCI         = st.sidebar.slider("PCI (kJ/kg)", 10000, 60000, 50000, 1000)

# Simulación

def run_simulation():
    if not mun_file or not tur_file:
        st.sidebar.error("Sube ambos archivos para simular.")
        return

    # Leer datos
    df_mun = pd.read_excel(mun_file)
    df_mun["T1 (K)"]   = df_mun["Temperatura (°C)"] + 273.15
    df_mun["P1 (kPa)"] = df_mun["Presión (bares)"] * 100.0
    df_tur = pd.read_csv(tur_file)

    # Configurar backend
    backend.eta_caldera = eta_caldera
    backend.M_DOT_WATER = m_dot_water
    backend.U_GLOBAL    = U
    backend.A_GLOBAL    = A
    backend.T_C_IN      = T_c_in

    # Flujo volumétrico diseño
    T1_ISO, P1_ISO = 288.15, 101.325
    rho_iso = backend.PropsSI("D","T",T1_ISO,"P",P1_ISO*1e3, backend.FLUIDO)
    df_tur["V_dot_design"] = df_tur["m_aire (kg/s)"] / rho_iso

    # Preparar
    resultados_est = []
    resultados_calc = []
    resultados_cog = []
    total = len(df_mun) * len(df_tur)
    progress = st.sidebar.progress(0)
    idx = 0

    # Loop
    for _, m in df_mun.iterrows():
        for _, t in df_tur.iterrows():
            est, calc, cog = backend.simular_ciclo(
                m["T1 (K)"], m["P1 (kPa)"],
                t["r_p"], t["T3 (C)"], t["eta_c"], t["eta_t"],
                V_dot_design=t["V_dot_design"],
                eta_gen=eta_gen, eta_caldera=eta_caldera, PCI=PCI
            )
            # Insertar metadatos
            est.insert(0, "Municipio", m["Municipio"])
            est.insert(1, "Turbina",   t["Turbina"])
            calc.update({"Municipio":m["Municipio"], "Turbina":t["Turbina"]})
            cog.update({"Municipio":m["Municipio"], "Turbina":t["Turbina"]})
            resultados_est.append(est)
            resultados_calc.append(calc)
            resultados_cog.append(cog)
            idx += 1
            progress.progress(idx/total)

    # DataFrames
    df_est = pd.concat(resultados_est, ignore_index=True)
    df_calc = pd.DataFrame(resultados_calc)
    df_cog = pd.DataFrame(resultados_cog)

    # Mapear altitud al DataFrame de cálculos si existe en df_mun
    if 'Altitud (media)' in df_mun.columns:
        df_calc['Altitud (media)'] = df_calc['Municipio'].map(
            df_mun.set_index('Municipio')['Altitud (media)']
        )

    # Merge para gráficos con Q_rec
    df_merge = df_calc.merge(
        df_cog[['Municipio','Turbina','Q_rec (kW)']],
        on=['Municipio','Turbina']
    )

    st.success("✅ Simulación completada!")

    # Mostrar tablas
    st.header("Resultados de Simulación")
    st.subheader("Estados termodinámicos")
    st.dataframe(df_est)
    st.subheader("Cálculos")
    st.dataframe(df_calc)
    st.subheader("Cogeneración")
    st.dataframe(df_cog)

    # Gráficas 2D y 3D
    tabs = st.tabs([
        "Pot vs Eficiencia",         
        "Recuperación vs Potencia",
        "Heat Rate vs Potencia",    
        "Potencia vs Altitud",      
        "3D Pot-Eficiencia-SFC",    
        "3D Recup-Pot-SFC",         
        "3D HR-Pot-SFC",            
        "3D Pot-Alt-SFC"
    ])

    # 2D: Potencia vs Eficiencia
    with tabs[0]:
        fig = px.scatter(
            df_calc, x='P_elec (kW)', y='eta_ciclo (%)',
            color='Turbina', hover_data=['Municipio'],
            title='Potencia vs Eficiencia de Ciclo'
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2D: Recuperación térmica vs Potencia
    with tabs[1]:
        fig = px.scatter(
            df_merge, x='P_elec (kW)', y='Q_rec (kW)',
            color='Turbina', hover_data=['Municipio'],
            title='Recuperación térmica vs Potencia'
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2D: Heat Rate vs Potencia
    with tabs[2]:
        fig = px.scatter(
            df_calc, x='P_elec (kW)', y='Heat rate (kJ/kWh)',
            color='Turbina', hover_data=['Municipio'],
            title='Heat Rate vs Potencia'
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2D: Potencia vs Altitud
    with tabs[3]:
        fig = px.scatter(
            df_calc, x='Altitud (media)', y='P_elec (kW)',
            color='Turbina', hover_data=['Municipio'],
            title='Potencia vs Altitud'
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3D: Potencia vs Eficiencia vs SFC
    with tabs[4]:
        fig3d = px.scatter_3d(
            df_calc, x='P_elec (kW)', y='eta_ciclo (%)', z='SFC (kg/kWh)',
            color='Turbina', hover_data=['Municipio'],
            title='3D: Potencia vs Eficiencia vs SFC'
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # 3D: Recuperación vs Potencia vs SFC
    with tabs[5]:
        fig3d = px.scatter_3d(
            df_merge, x='P_elec (kW)', y='Q_rec (kW)', z='SFC (kg/kWh)',
            color='Turbina', hover_data=['Municipio'],
            title='3D: Recuperación vs Potencia vs SFC'
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # 3D: Heat Rate vs Potencia vs SFC
    with tabs[6]:
        fig3d = px.scatter_3d(
            df_calc, x='P_elec (kW)', y='Heat rate (kJ/kWh)', z='SFC (kg/kWh)',
            color='Turbina', hover_data=['Municipio'],
            title='3D: Heat Rate vs Potencia vs SFC'
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # 3D: Potencia vs Altitud vs SFC
    with tabs[7]:
        fig3d = px.scatter_3d(
            df_calc, x='Altitud (media)', y='P_elec (kW)', z='SFC (kg/kWh)',
            color='Turbina', hover_data=['Municipio'],
            title='3D: Potencia vs Altitud vs SFC'
        )
        st.plotly_chart(fig3d, use_container_width=True)

# Botón
if st.sidebar.button("▶️ Simular"):
    run_simulation()

