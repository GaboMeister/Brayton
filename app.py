# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 02:58:07 2025

@author: Gabo San
"""
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import BraytonRI6 as backend  # motor RI6 con simular_batch

st.set_page_config(page_title="Brayton RI6 — Dashboard", layout="wide")

# ===== Sidebar: entradas y parámetros =====
st.sidebar.title("⚙️ Entradas y parámetros")

mun_file = st.sidebar.file_uploader("Municipios_D.xlsx", type=["xlsx"])
tur_file = st.sidebar.file_uploader("Base_de_datos_turbinas_de_gas.csv", type=["csv"])

with st.sidebar.expander("Intercambiador y lazo de agua", expanded=True):
    DTMIN_HX_C    = st.slider("ΔTmin HX (°C)", 5.0, 25.0, 10.0, 0.5)
    T7_user_C     = st.slider("T7 usuario (°C)", 120.0, 220.0, 180.0, 1.0)
    P7_bar        = st.slider("P7 (bar)", 5.0, 20.0, 10.0, 0.5)
    P8_bar        = st.slider("P8 (bar)", 1.0, 3.0, 2.0, 0.1)
    Delta_subcool = st.slider("Subenfriado retorno (°C)", 5.0, 30.0, 15.0, 1.0)

with st.sidebar.expander("Generación y combustible (opcional)", expanded=False):
    ETA_GEN  = st.slider("η generador", 0.80, 0.99, 0.95, 0.005)
    ETA_CALD = st.slider("η caldera",   0.70, 0.95, 0.85, 0.01)
    PCI      = st.number_input("PCI (kJ/kg)", 30000.0, 60000.0, 50000.0, 100.0)

run = st.sidebar.button("▶️ Simular")

st.title("Ciclo Brayton + Cogeneración — Dashboard (RI6)")
st.markdown(
    "Sube **municipios** y **turbinas**, ajusta parámetros y presiona **Simular**. "
    "La app corre el motor **RI6** en el servidor y te permite **descargar** el Excel de resultados."
)

# ===== Helpers cache =====
@st.cache_data(show_spinner=False)
def _read_inputs(mun_bytes, tur_bytes):
    df_mun = pd.read_excel(mun_bytes)
    df_tur = pd.read_csv(tur_bytes)
    return df_mun, df_tur

# ===== Ejecución =====
if run:
    if not mun_file or not tur_file:
        st.warning("Sube ambos archivos (municipios y turbinas) antes de simular.")
        st.stop()

    # Lectura
    df_mun, df_tur = _read_inputs(mun_file, tur_file)

    # Simulación
    with st.spinner("Calculando con RI6…"):
        df_est, df_calc, df_cog, dfv, inviables, top10, bottom10, excel_bytes = backend.simular_batch(
            df_mun, df_tur,
            DTMIN_HX_C=DTMIN_HX_C, T7_user_C=T7_user_C,
            P7_bar=P7_bar, P8_bar=P8_bar, Delta_subcool_C=Delta_subcool,
            ETA_GEN_=ETA_GEN, ETA_CALD_=ETA_CALD, PCI_=PCI,
            write_excel=True
        )

    st.success("¡Simulación completa!")

    # ===== Descargar Excel =====
    st.download_button(
        "⬇️ Descargar Excel de resultados",
        data=excel_bytes,
        file_name="Resultados_Ciclo_Brayton.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # ===== Tabs “core” =====
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "✅ Validaciones", "⚠️ Inviables", "📊 Cálculos", "🌡️ Estados",
        "🟢 Semáforo HX", "🌡️ T7_gap", "⚡ Eficiencias"
    ])

    with t1:
        st.subheader("Validaciones")
        st.dataframe(dfv, use_container_width=True, height=520)

    with t2:
        st.subheader("Inviables")
        st.dataframe(inviables, use_container_width=True, height=520)

    with t3:
        st.subheader("Cálculos")
        st.dataframe(df_calc, use_container_width=True, height=520)

    with t4:
        st.subheader("Estados termodinámicos")
        st.dataframe(df_est, use_container_width=True, height=520)

    with t5:
        st.subheader("Semáforo HX por municipio")
        sema = (df_calc.groupby(["Municipio", "Semaforo_HX"])
                        .size().reset_index(name="conteo"))
        if not sema.empty:
            fig = px.bar(
                sema, x="Municipio", y="conteo", color="Semaforo_HX",
                barmode="stack", title="OK / Límite usuario / Límite GC / Ambos"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin datos para graficar el semáforo.")

    with t6:
        st.subheader("Alcanzabilidad del setpoint (T7)")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_calc, x="T7_gap (K)", nbins=40,
                               title="Distribución de T7_gap (K)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                df_calc, x="T7_gap (K)", y="eta_cog_global (%)",
                color="Semaforo_HX", hover_data=["Municipio","Turbina"],
                title="T7_gap (K) vs eficiencia global"
            )
            st.plotly_chart(fig, use_container_width=True)

    with t7:
        st.subheader("Eficiencia vs Potencia eléctrica")
        fig = px.scatter(
            df_calc, x="P_elec (kW)", y="eta_cog_global (%)",
            color="Semaforo_HX", hover_data=["Municipio","Turbina"],
            title="η_cog_global (%) vs P_elec (kW)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===== PREPROCESO para gráficas “clásicas” del asesor (mapeo de columnas) =====
    # 1) Altitud (media) desde los municipios (si existe)
    if "Altitud (media)" in df_mun.columns:
        df_calc["Altitud (media)"] = df_calc["Municipio"].map(
            df_mun.set_index("Municipio")["Altitud (media)"]
        )
    else:
        df_calc["Altitud (media)"] = np.nan

    # 2) Renombrado a nombres “clásicos” del asesor
    df_asesor = df_calc.rename(columns={
        "P_elec (kW)":              "P_elec_corr (kW)",   # potencia "corregida"
        "eta_ciclo_electrico (%)":  "eta_ciclo (%)",
        "eta_cog_global (%)":       "eta_global (%)",
        "SFC_elec (kg/kWh)":        "SFC (kg/kWh)",
        "Q_gc (kW)":                "Q_input (kW)",       # calor de gases al HX
        "Q_user (kW)":              "Q_rec (kW)",         # calor útil al usuario
    })

    # ===== Pestañas “Clásico Asesor” (idénticas a la app anterior) =====
    tabs_clas = st.tabs([
        "Pot vs Eficiencia",
        "Qútil vs Qin",
        "Pot vs Qin",
        "SFC vs Pot",
        "η_global vs Altitud",
        "Q_rec vs Altitud",
        "3D Pot-Alt-SFC"
    ])

    # 1) Potencia vs eficiencia de ciclo
    with tabs_clas[0]:
        fig = px.scatter(
            df_asesor,
            x="P_elec_corr (kW)",
            y="eta_ciclo (%)",
            color="Turbina",
            labels={
                "P_elec_corr (kW)": "Potencia corregida (kW)",
                "eta_ciclo (%)": "Eficiencia ciclo (%)"
            },
            title="Potencia vs Eficiencia de Ciclo"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2) Energía térmica útil vs calor suministrado
    with tabs_clas[1]:
        fig = px.scatter(
            df_asesor,
            x="Q_input (kW)",
            y="Q_rec (kW)",
            color="Turbina",
            labels={
                "Q_input (kW)": "Calor suministrado (kW)",
                "Q_rec (kW)": "Energía térmica útil (kW)"
            },
            title="Energía térmica útil vs Calor suministrado"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3) Potencia eléctrica corregida vs calor suministrado
    with tabs_clas[2]:
        fig = px.scatter(
            df_asesor,
            x="Q_input (kW)",
            y="P_elec_corr (kW)",
            color="Turbina",
            labels={
                "Q_input (kW)": "Calor suministrado (kW)",
                "P_elec_corr (kW)": "Potencia corregida (kW)"
            },
            title="Potencia eléctrica vs Calor suministrado"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 4) SFC vs Potencia eléctrica
    with tabs_clas[3]:
        fig = px.scatter(
            df_asesor,
            x="P_elec_corr (kW)",
            y="SFC (kg/kWh)",
            color="Turbina",
            labels={
                "P_elec_corr (kW)": "Potencia corregida (kW)",
                "SFC (kg/kWh)": "SFC (kg/kWh)"
            },
            title="SFC vs Potencia eléctrica"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5) Eficiencia global vs Altitud
    with tabs_clas[4]:
        fig = px.scatter(
            df_asesor,
            x="Altitud (media)",
            y="eta_global (%)",
            color="Turbina",
            labels={"eta_global (%)": "Eficiencia global (%)"},
            title="Eficiencia global vs Altitud"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 6) Calor recuperado vs Altitud
    with tabs_clas[5]:
        fig = px.scatter(
            df_asesor,
            x="Altitud (media)",
            y="Q_rec (kW)",
            color="Turbina",
            labels={"Q_rec (kW)": "Calor recuperado (kW)"},
            title="Calor recuperado vs Altitud"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 7) Gráfica 3D Potencia vs Altitud vs SFC
    with tabs_clas[6]:
        fig3d = px.scatter_3d(
            df_asesor,
            x="Altitud (media)",
            y="P_elec_corr (kW)",
            z="SFC (kg/kWh)",
            color="Turbina",
            hover_data=["Municipio"],
            labels={
                "Altitud (media)": "Altitud (m)",
                "P_elec_corr (kW)": "Potencia corregida (kW)",
                "SFC (kg/kWh)": "SFC (kg/kWh)"
            },
            title="Potencia vs Altitud vs SFC (3D)"
        )
        fig3d.update_layout(width=900, height=650)
        st.plotly_chart(fig3d, use_container_width=True)

else:
    st.info("👈 Sube **Municipios_D.xlsx** y **Base_de_datos_turbinas_de_gas.csv**, ajusta parámetros y da clic en **Simular**.")

