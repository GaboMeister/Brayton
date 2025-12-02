# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 02:32:09 2025

@author: Gabo San
"""

# app.py
# Interfaz en Streamlit para correr BraytonRI8.py y visualizar resultados
# Pensada para usuarios que NO programan: suben sus bases, mueven sliders y listo.

import os
import glob
import json
import subprocess
import sys

import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -------------------------------------------------
st.set_page_config(
    page_title="Simulación Brayton + Cogeneración",
    layout="wide",
)

# Encabezado institucional
try:
    st.columns([1, 2, 1])[1].image("logo_uam.png", width=300)
except Exception:
    pass

st.markdown(
    """
<div style="text-align:center; line-height:1.2;">
  <h2>UNIVERSIDAD AUTÓNOMA METROPOLITANA</h2>
  <p>Proyecto Terminal</p>
  <p><strong>Asesor:</strong> 🎓 Hernando Romero Paredes Rubio &nbsp;|&nbsp;
     <strong>Alumno:</strong> 🤖 Rolando Gabriel Garza Luna</p>
</div>
""",
    unsafe_allow_html=True,
)

st.title("Simulación Ciclo Brayton + Cogeneración")


# -------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------
def escribir_config(
    PCI,
    eta_gen,
    eta_cc,
    modo_servicio,
    P_serv_bar,
    T_serv_C,
    P_ret_bar,
    m_dot_agua,
    UA_HRSG,
    T5_min_C,
    filename="config_brayton.json",
):
    """
    Crea/actualiza un archivo JSON con los parámetros de simulación.
    BraytonRI8.py puede leer este JSON para sobreescribir sus parámetros por defecto.
    """
    cfg = {
        "fluido": "Air",
        "PCI": PCI,
        "eta_gen": eta_gen,
        "eta_cc": eta_cc,
        "modo_servicio": modo_servicio,
        "P_serv_bar": P_serv_bar,
        "T_serv_C": T_serv_C,
        "P_ret_bar": P_ret_bar,
        "m_dot_agua": m_dot_agua,
        "UA_HRSG": UA_HRSG,
        "T5_min_C": T5_min_C,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def encontrar_ultimo_resultado(pattern="resultados_brayton*.xlsx"):
    """
    Devuelve la ruta del archivo resultados_brayton*.xlsx más reciente
    (por fecha de modificación). Si no existe, devuelve None.
    """
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


@st.cache_data
def cargar_resultados(path_excel: str, mtime: float):
    """
    Carga las hojas 'Estados' y 'Resultados' del archivo de Excel.
    mtime solo se usa para invalidar la caché si el archivo cambia.
    """
    xls = pd.ExcelFile(path_excel)
    df_estados = pd.read_excel(xls, sheet_name="Estados")
    df_resultados = pd.read_excel(xls, sheet_name="Resultados")
    return df_estados, df_resultados


def formatear_num(x, ndigits=2):
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "—"


# -------------------------------------------------
# SIDEBAR: CARGA DE BASES + PARÁMETROS + BOTÓN
# -------------------------------------------------
st.sidebar.header("📂 Bases de datos de entrada")

turbinas_file = st.sidebar.file_uploader(
    "Base de datos de turbinas (.csv)",
    type=["csv"],
    help="Sube el archivo CSV con las turbinas de gas.",
)

municipios_file = st.sidebar.file_uploader(
    "Base de datos de municipios (.xlsx / .xls)",
    type=["xlsx", "xls"],
    help="Sube el archivo Excel con los municipios/localidades.",
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros de simulación")

st.sidebar.markdown("**Parámetros del ciclo / combustible**")
PCI = st.sidebar.number_input(
    "PCI [kJ/kg]",
    min_value=10000.0,
    max_value=70000.0,
    value=55090.0,
    step=100.0,
)
eta_gen = st.sidebar.slider("Eficiencia del generador η_gen", 0.5, 1.0, 0.98, 0.01)
eta_cc = st.sidebar.slider(
    "Eficiencia de cámara de combustión η_cc", 0.5, 1.0, 0.95, 0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Parámetros de cogeneración / HRSG**")

modo_servicio = st.sidebar.selectbox(
    "Modo de servicio (solo informativo)", ["vapor", "agua"], index=0
)

P_serv_bar = st.sidebar.slider(
    "Presión de servicio P6 = P7 [bar abs]", 2.0, 40.0, 10.0, 0.5
)
T_serv_C = st.sidebar.slider(
    "Temperatura de servicio T7 [°C]", 80.0, 400.0, 180.0, 5.0
)
P_ret_bar = st.sidebar.slider(
    "Presión de retorno P8 [bar abs]", 1.0, 5.0, 2.0, 0.5
)
m_dot_agua = st.sidebar.slider(
    "Caudal de agua ṁ_agua [kg/s]", 0.5, 50.0, 5.0, 0.5
)
UA_HRSG = st.sidebar.slider(
    "UA_HRSG [kJ/s·K]", 100.0, 5000.0, 1500.0, 50.0
)
T5_min_C = st.sidebar.slider(
    "T5 mínima (gases salida HRSG) [°C]", 80.0, 200.0, 100.0, 5.0
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Ejecutar simulación")


# -------------------------------------------------
# EJECUTAR LA SIMULACIÓN SI SE PULSA EL BOTÓN
# -------------------------------------------------
if run_button:
    # 1) Validar que se subieron ambos archivos (o ya existan en la carpeta)
    base_turb_name = "Base_de_datos_turbinas_de_gas.csv"
    base_mpios_name = "Municipios_D.xlsx"

    if turbinas_file is not None:
        # Guardar el CSV subido con el nombre que espera BraytonRI8.py
        with open(base_turb_name, "wb") as f:
            f.write(turbinas_file.getbuffer())
    elif not os.path.exists(base_turb_name):
        st.error(
            "No se encontró la base de turbinas.\n\n"
            "Por favor, sube un archivo CSV con los datos de las turbinas."
        )
        st.stop()

    if municipios_file is not None:
        # Guardar el Excel subido con el nombre que espera BraytonRI8.py
        with open(base_mpios_name, "wb") as f:
            f.write(municipios_file.getbuffer())
    elif not os.path.exists(base_mpios_name):
        st.error(
            "No se encontró la base de municipios.\n\n"
            "Por favor, sube un archivo Excel con los datos de los municipios."
        )
        st.stop()

    # 2) Escribir archivo de configuración (para que BraytonRI8 use estos parámetros)
    escribir_config(
        PCI=PCI,
        eta_gen=eta_gen,
        eta_cc=eta_cc,
        modo_servicio=modo_servicio,
        P_serv_bar=P_serv_bar,
        T_serv_C=T_serv_C,
        P_ret_bar=P_ret_bar,
        m_dot_agua=m_dot_agua,
        UA_HRSG=UA_HRSG,
        T5_min_C=T5_min_C,
    )

    # 3) Ejecutar BraytonRI8.py con el mismo intérprete que está corriendo Streamlit
    with st.spinner("Ejecutando simulación Brayton + Cogeneración..."):
        try:
            result = subprocess.run(
                [sys.executable, "BraytonRI8.py"],
                capture_output=True,
                text=True,
                check=True,
            )
            st.success("Simulación completada correctamente.")
            if result.stdout:
                with st.expander("Ver salida de consola del modelo"):
                    st.text(result.stdout)
            if result.stderr:
                with st.expander("Ver mensajes de advertencia / error"):
                    st.text(result.stderr)
        except subprocess.CalledProcessError as e:
            st.error("Ocurrió un error al ejecutar BraytonRI8.py")
            st.code(e.stderr or str(e))
            st.stop()

# -------------------------------------------------
# CARGAR EL ÚLTIMO RESULTADO DISPONIBLE
# -------------------------------------------------
ultimo_archivo = encontrar_ultimo_resultado()

if not ultimo_archivo:
    st.warning(
        "Aún no hay archivos 'resultados_brayton*.xlsx' en la carpeta.\n\n"
        "Sube tus bases y ejecuta primero la simulación desde la barra lateral."
    )
    st.stop()

mtime = os.path.getmtime(ultimo_archivo)
st.info(f"Usando archivo de resultados: **{os.path.basename(ultimo_archivo)}**")

df_estados, df_resultados = cargar_resultados(ultimo_archivo, mtime)

# -------------------------------------------------
# DESCARGA DE HOJAS COMPLETAS
# -------------------------------------------------
st.subheader("💾 Descargar resultados completos")

col_d1, col_d2 = st.columns(2)

with col_d1:
    csv_est = df_estados.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar hoja 'Estados' (CSV)",
        data=csv_est,
        file_name="Estados_brayton.csv",
        mime="text/csv",
    )

with col_d2:
    csv_res = df_resultados.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar hoja 'Resultados' (CSV)",
        data=csv_res,
        file_name="Resultados_brayton.csv",
        mime="text/csv",
    )

st.markdown("---")

# -------------------------------------------------
# FILTROS: TURBINA Y MUNICIPIO PARA VER DETALLE
# -------------------------------------------------
st.subheader("🔍 Análisis detallado por turbina y municipio")

turbinas = sorted(df_resultados["Turbina"].unique())
t_col1, t_col2 = st.columns(2)
with t_col1:
    turbina_sel = st.selectbox("Turbina:", turbinas)

df_res_turb = df_resultados[df_resultados["Turbina"] == turbina_sel]

municipios = sorted(df_res_turb["Municipio"].unique())
with t_col2:
    municipio_sel = st.selectbox("Municipio:", municipios)

df_res_sel = df_res_turb[df_res_turb["Municipio"] == municipio_sel]
if df_res_sel.empty:
    st.warning("No hay resultados para esa combinación turbina/municipio.")
    st.stop()

fila = df_res_sel.iloc[0]

df_est_sel = df_estados[
    (df_estados["Turbina"] == turbina_sel)
    & (df_estados["Municipio"] == municipio_sel)
].sort_values("Estado")

# -------------------------------------------------
# PANEL DE MÉTRICAS LOCALES
# -------------------------------------------------
st.subheader("📊 Resumen del punto de operación seleccionado")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Potencia eléctrica neta [kW]",
    formatear_num(fila.get("P_elec [kW]"), 1),
)
col2.metric(
    "η ciclo [%]",
    formatear_num(fila.get("Eficiencia_ciclo [%]"), 1),
)
col3.metric(
    "η cogeneración [%]",
    formatear_num(fila.get("Eficiencia_cogeneracion [%]"), 1),
)
col4.metric(
    "Q_user [kW]",
    formatear_num(fila.get("Q_user [kW]"), 1),
)

col5, col6, col7, col8 = st.columns(4)
col5.metric(
    "Heat Rate real [kJ/kWh]",
    formatear_num(fila.get("HeatRate_real (kJ/kWh)"), 1),
)
col6.metric(
    "Altitud media [m]",
    formatear_num(fila.get("Altitud (media) [m]"), 1),
)
col7.metric(
    "T_municipio [°C]",
    formatear_num(fila.get("Temperatura_mpio [°C]"), 1),
)
col8.metric(
    "Derate potencia [-]",
    formatear_num(fila.get("Derate_P [-]"), 3),
)

# -------------------------------------------------
# ESTADOS TERMODINÁMICOS 1–8
# -------------------------------------------------
st.subheader("📋 Estados termodinámicos (1–8)")

if df_est_sel.empty:
    st.info(
        "Para esta combinación turbina/municipio solo se tienen estados 1–4 "
        "(no hubo cogeneración factible)."
    )
else:
    cols_orden = ["Estado", "T [K]", "P [Pa]", "h [kJ/kg]", "s [kJ/kg·K]"]
    cols_extra = [
        c
        for c in df_est_sel.columns
        if c not in cols_orden and c not in ["Turbina", "Municipio"]
    ]
    df_mostrar = df_est_sel[["Turbina", "Municipio"] + cols_orden + cols_extra]

    st.dataframe(
        df_mostrar.reset_index(drop=True),
        use_container_width=True,
        height=350,
    )

# -------------------------------------------------
# GRÁFICAS: TODAS LAS TURBINAS, TODOS LOS MUNICIPIOS
# -------------------------------------------------
st.subheader("📈 Gráficas globales (todas las turbinas, todos los municipios)")

tab1, tab2, tab3 = st.tabs(
    [
        "Potencia vs Altitud",
        "Heat Rate / eficiencias vs Temperatura ambiente",
        "Scatter 3D: P_elec vs HR vs Altitud",
    ]
)

with tab1:
    st.markdown("**Potencia eléctrica vs altitud (todas las turbinas)**")
    fig_p_alt = px.scatter(
        df_resultados,
        x="Altitud (media) [m]",
        y="P_elec [kW]",
        color="Turbina",
        hover_data=[
            "Municipio",
            "Eficiencia_ciclo [%]",
            "Eficiencia_cogeneracion [%]",
        ],
        labels={
            "Altitud (media) [m]": "Altitud [m]",
            "P_elec [kW]": "Potencia eléctrica [kW]",
        },
    )
    st.plotly_chart(fig_p_alt, use_container_width=True)

with tab2:
    st.markdown("**Heat Rate vs Temperatura ambiente**")
    fig_hr_T = px.scatter(
        df_resultados,
        x="Temperatura_mpio [°C]",
        y="HeatRate_real (kJ/kWh)",
        color="Turbina",
        hover_data=["Municipio", "Altitud (media) [m]"],
        labels={
            "Temperatura_mpio [°C]": "Temperatura ambiente [°C]",
            "HeatRate_real (kJ/kWh)": "Heat Rate real [kJ/kWh]",
        },
    )
    st.plotly_chart(fig_hr_T, use_container_width=True)

    st.markdown("**η ciclo vs Temperatura ambiente**")
    fig_eta_T = px.scatter(
        df_resultados,
        x="Temperatura_mpio [°C]",
        y="Eficiencia_ciclo [%]",
        color="Turbina",
        hover_data=["Municipio", "Eficiencia_cogeneracion [%]"],
        labels={"Temperatura_mpio [°C]": "Temperatura ambiente [°C]"},
    )
    st.plotly_chart(fig_eta_T, use_container_width=True)

with tab3:
    st.markdown("**Potencia eléctrica vs Heat Rate vs Altitud (3D)**")
    fig_3d = px.scatter_3d(
        df_resultados,
        x="Altitud (media) [m]",
        y="HeatRate_real (kJ/kWh)",
        z="P_elec [kW]",
        color="Turbina",
        hover_name="Turbina",
        hover_data=["Municipio", "Eficiencia_cogeneracion [%]"],
        labels={
            "Altitud (media) [m]": "Altitud [m]",
            "HeatRate_real (kJ/kWh)": "Heat Rate [kJ/kWh]",
            "P_elec [kW]": "Potencia [kW]",
        },
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# -------------------------------------------------
# ANÁLISIS ESTADÍSTICO GLOBAL
# -------------------------------------------------
st.subheader("📊 Análisis estadístico global")

tab_est1, tab_est2, tab_est3 = st.tabs(
    [
        "Resumen por turbina",
        "Matriz de correlación (heatmap)",
        "Histograma de η de cogeneración",
    ]
)

with tab_est1:
    st.markdown("### 📌 Estadísticos descriptivos por turbina")

    # Elegimos algunas columnas numéricas de interés
    cols_stats = [
        "P_elec [kW]",
        "Eficiencia_ciclo [%]",
        "Eficiencia_cogeneracion [%]",
        "HeatRate_real (kJ/kWh)",
    ]

    # Nos quedamos solo con las columnas que existan en el DataFrame
    cols_stats = [c for c in cols_stats if c in df_resultados.columns]

    if not cols_stats:
        st.info("No se encontraron columnas numéricas esperadas para el resumen.")
    else:
        df_stats = (
            df_resultados.groupby("Turbina")[cols_stats]
            .agg(["mean", "std", "min", "max"])
        )

        # Opcional: redondear para que se vea más bonito
        df_stats = df_stats.round(2)

        st.dataframe(df_stats, use_container_width=True)

        st.markdown(
            """
**Interpretación:**  
- Compara medias y desviaciones estándar entre turbinas para ver cuáles
  son más eficientes y cuáles tienen resultados más dispersos.  
"""
        )

with tab_est2:
    st.markdown("### 🔥 Matriz de correlación")

    # Seleccionamos algunas variables continuas relevantes
    cols_corr = [
        "Altitud (media) [m]",
        "Temperatura_mpio [°C]",
        "P_elec [kW]",
        "HeatRate_real (kJ/kWh)",
        "Eficiencia_ciclo [%]",
        "Eficiencia_cogeneracion [%]",
    ]
    cols_corr = [c for c in cols_corr if c in df_resultados.columns]

    if len(cols_corr) < 2:
        st.info("No hay suficientes variables numéricas para calcular correlaciones.")
    else:
        df_corr = df_resultados[cols_corr].corr()

        st.write("Correlaciones lineales entre variables seleccionadas:")
        st.dataframe(df_corr.round(2), use_container_width=True)

        # Heatmap con Plotly
        fig_corr = px.imshow(
            df_corr,
            text_auto=True,
            aspect="auto",
            labels=dict(color="Correlación"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown(
            """
**Interpretación:**  
- Valores cercanos a **+1** indican relación directa fuerte.  
- Valores cercanos a **−1** indican relación inversa fuerte.  
"""
        )

with tab_est3:
    st.markdown("### 📊 Histograma de eficiencia de cogeneración")

    col_hist1, col_hist2 = st.columns([2, 1])

    col_target = "Eficiencia_cogeneracion [%]"
    if col_target not in df_resultados.columns:
        st.info("No se encontró la columna 'Eficiencia_cogeneracion [%]' en los resultados.")
    else:
        with col_hist1:
            fig_hist = px.histogram(
                df_resultados,
                x=col_target,
                color="Turbina",
                nbins=30,
                labels={col_target: "Eficiencia de cogeneración [%]"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_hist2:
            st.markdown("#### Parámetros básicos")
            st.write(
                df_resultados[col_target].describe().round(2).to_frame(name=col_target)
            )

        st.markdown(
            """
**Interpretación:**  
- En qué rango se concentra la mayor parte de la eficiencia de cogeneración.  
- El color por turbina permite ver si hay modelos que tienden a operar
  sistemáticamente mejor que otros.
"""
        )

