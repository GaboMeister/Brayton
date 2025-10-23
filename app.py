# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 02:58:07 2025

@author: Gabo San
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import ri8_wrapper as w  
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

# -------------------------------------------------------------------
# Estado persistente
# -------------------------------------------------------------------
if "dfs" not in st.session_state:
    st.session_state.dfs = None  

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def safe_read_excel(path):
    try:
        return pd.read_excel(path)
    except Exception:
        return None

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def safe_range(series, pad_ratio=0.1, pad_abs=1.0):
    """Devuelve (min, max) seguro para sliders; si min==max agrega padding."""
    s = series.dropna()
    if s.empty:
        return (0.0, 1.0)
    mn, mx = float(s.min()), float(s.max())
    if mn == mx:
        pad = max(abs(mn) * pad_ratio, pad_abs)
        return (mn - pad, mx + pad)
    return (mn, mx)

# -------------------------------------------------------------------
# 1) Carga de datos (simple)
# -------------------------------------------------------------------
st.header("1) Datos de entrada")

col1, col2 = st.columns(2)
with col1:
    f_mun = st.file_uploader("Sube Municipios_D.xlsx", type=["xlsx"], key="mun")
    if f_mun is None:
        st.caption("Si no subes archivo, intenta **data/Municipios_D.xlsx** (si existe).")
        df_mun = safe_read_excel("data/Municipios_D.xlsx")
        if df_mun is None:
            df_mun = pd.DataFrame()
    else:
        df_mun = pd.read_excel(f_mun)

with col2:
    f_tur = st.file_uploader("Sube Base_de_datos_turbinas_de_gas.csv", type=["csv"], key="tur")
    if f_tur is None:
        st.caption("Si no subes archivo, intenta **data/Base_de_datos_turbinas_de_gas.csv** (si existe).")
        df_tur = safe_read_csv("data/Base_de_datos_turbinas_de_gas.csv")
        if df_tur is None:
            df_tur = pd.DataFrame()
    else:
        df_tur = pd.read_csv(f_tur)

ok_inputs = (not df_mun.empty) and (not df_tur.empty)

# -------------------------------------------------------------------
# 2) Parámetros (sliders)
# -------------------------------------------------------------------
st.header("2) Parámetros")

with st.form("form_params"):
    st.markdown("### HX & Agua")
    c1, c2, c3 = st.columns(3)
    with c1:
        U_GLOBAL = st.slider("U_GLOBAL (kW/m²·K)", 0.05, 2.0, 0.30, 0.01)
        A_GLOBAL = st.slider("A_GLOBAL (m²)", 1.0, 50.0, 8.0, 0.5)
        m_dot_w  = st.slider("m_dot_w (kg/s)", 0.1, 50.0, 2.0, 0.1)
    with c2:
        P7_BAR = st.slider("P7_BAR (bar)", 2.0, 60.0, 10.0, 0.5)
        P8_BAR = st.slider("P8_BAR (bar)", 0.1, 20.0, 2.0, 0.1)
        DELTA_SUBCOOL_C = st.slider("DELTA_SUBCOOL_C (°C)", 0.0, 40.0, 15.0, 1.0)
    with c3:
        T7_USER_C = st.slider("T7_USER_C (°C)", 60.0, 260.0, 180.0, 1.0)

    st.markdown("### Brayton / Combustible / Restricción")
    c4, c5, c6 = st.columns(3)
    with c4:
        ETA_GEN = st.slider("ETA_GEN (-)", 0.70, 1.00, 0.95, 0.01)
        ETA_CALDERA = st.slider("ETA_CALDERA (-)", 0.50, 1.00, 0.85, 0.01)
    with c5:
        PCI = st.slider("PCI (kJ/kg)", 20000, 60000, 50000, 500)
    with c6:
        T_GAS_OUT_MINC = st.slider("T_GAS_OUT_MINC (°C)", 80, 180, 120, 1)

    submitted = st.form_submit_button("Guardar parámetros")

if submitted:
    st.success("Parámetros listos ✅")

# -------------------------------------------------------------------
# 3) Barrido cartesiano completo
# -------------------------------------------------------------------
st.header("3) Barrido cartesiano completo")

with st.expander("Opciones del barrido", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        muni_sel_full = st.multiselect(
            "Filtrar Municipios (opcional)",
            sorted(df_mun["Municipio"].dropna().unique().tolist()) if ok_inputs else [],
        )
    with c2:
        turb_sel_full = st.multiselect(
            "Filtrar Turbinas (opcional)",
            sorted(df_tur["Turbina"].dropna().unique().tolist()) if ok_inputs else [],
        )
    mode_label = st.radio(
        "Modo de simulación",
        ["Full físico (ṁ volumétrico)", "Simplificado (target de placa)"],
        index=0, horizontal=False
    )
    mode_key = "mdot" if "Full físico" in mode_label else "target"


    use_cache = st.checkbox("Usar caché (más rápido, sin barra de progreso)", value=False)

btn_full = st.button("🧮 Ejecutar barrido cartesiano", disabled=not ok_inputs)

if btn_full:
    params = {
        "U_GLOBAL": U_GLOBAL, "A_GLOBAL": A_GLOBAL, "m_dot_w": m_dot_w,
        "P7_BAR": P7_BAR, "P8_BAR": P8_BAR, "DELTA_SUBCOOL_C": DELTA_SUBCOOL_C,
        "T7_USER_C": T7_USER_C, "ETA_GEN": ETA_GEN, "ETA_CALDERA": ETA_CALDERA,
        "PCI": PCI, "T_GAS_OUT_MINC": T_GAS_OUT_MINC,
    }

    if use_cache:
        # --------- caché sin barra de progreso ------------
        @st.cache_data(show_spinner=True)
        def _cached_run(m_df, t_df, params_, muni_, turb_, mode_):
            return w.run_cartesiano(
                m_df, t_df, params_,
                muni_sel=muni_ if muni_ else None,
                turb_sel=turb_ if turb_ else None,
                limit=None,
                progress_cb=None,
                mode=mode_,
            )
        df_est, df_calc, df_cog, df_valid = _cached_run(df_mun, df_tur, params, muni_sel_full, turb_sel_full, mode_key)
    else:
        # --------- ejecución con barra de progreso --------
        prog = st.progress(0, text="Preparando…")
        status = st.empty()

        def _cb(done, total):
            if total <= 0:
                prog.progress(0)
                status.write("0/0")
                return
            prog.progress(int(100 * done / total))
            status.write(f"{done}/{total}")

        with st.spinner("Corriendo barrido…"):
            df_est, df_calc, df_cog, df_valid = w.run_cartesiano(
                df_mun, df_tur, params,
                muni_sel=muni_sel_full if muni_sel_full else None,
                turb_sel=turb_sel_full if turb_sel_full else None,
                limit=None,
                progress_cb=_cb,
                mode=mode_key,
            )
        prog.progress(100, text="Listo ✔")
        status.write("Completado")

    # Persistir resultados del barrido (COMPLETOS)
    st.session_state.dfs = {
        "Estados": df_est, "Calculos": df_calc,
        "Cogeneracion": df_cog, "Validaciones": df_valid,
        "_params": params,
        "_scope": {"muni_sel": muni_sel_full, "turb_sel": turb_sel_full, "cached": use_cache, "mode": mode_key}
    }

# Mostrar resultados si existen (COMPLETOS)
if st.session_state.dfs:
    dfs_now = st.session_state.dfs
    st.success(f"Resultados del barrido: {len(dfs_now['Calculos'])} casos.")

    tab1, tab2, tab3, tab4 = st.tabs(["Estados", "Cálculos", "Cogeneración", "Validaciones"])
    with tab1: st.dataframe(dfs_now["Estados"], use_container_width=True)
    with tab2: st.dataframe(dfs_now["Calculos"], use_container_width=True)
    with tab3: st.dataframe(dfs_now["Cogeneracion"], use_container_width=True)
    with tab4: st.dataframe(dfs_now["Validaciones"], use_container_width=True)

    excel_bytes = w.make_excel_bytes(dfs_now)  # <-- Excel con TODAS las filas
    st.download_button(
        "⬇️ Descargar Excel (barrido completo)",
        data=excel_bytes,
        file_name="Resultados_Ciclo_Brayton_barrido.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# -------------------------------------------------------------------
# 4) Filtros y derivadas para gráficas
# -------------------------------------------------------------------
st.divider()
st.header("4) Filtros para gráficas")

dfs_state = st.session_state.dfs
if not dfs_state:
    st.info("No hay resultados para graficar. Ejecuta primero el barrido.")
else:
    df_calc = dfs_state["Calculos"].copy()
    df_cog  = dfs_state["Cogeneracion"].copy()
    params  = dfs_state.get("_params", {})
    PCI_val = float(params.get("PCI", 50000.0))

    # --------- Derivados ---------
    dfG = df_calc.copy()

    # Asegurar Altitud (m)
    if "Altitud (m)" not in dfG.columns:
        if "P1 (kPa)" in dfG.columns:
            P_Pa = dfG["P1 (kPa)"] * 1000.0
            dfG["Altitud (m)"] = 44330.0 * (1.0 - (P_Pa / 101325.0) ** 0.1903)
        else:
            dfG["Altitud (m)"] = float("nan")

    # Qin (kW)
    if "m_dot_fuel (kg/s)" in dfG.columns:
        dfG["Q_in (kW)"] = dfG["m_dot_fuel (kg/s)"] * PCI_val
    else:
        dfG["Q_in (kW)"] = float("nan")

    if "SFC_elec (kg/kWh)" not in dfG.columns:
        dfG["SFC_elec (kg/kWh)"] = float("nan")

    if "P_elec (kW)" not in dfG.columns and "P_net (kW)" in dfG.columns:
        dfG["P_elec (kW)"] = dfG["P_net (kW)"]

    # --------- Controles ---------
    with st.expander("Controles de visualización", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            pot_choice = st.radio(
                "Potencia a graficar",
                ["P_elec (kW)", "P_net (kW)"],
                index=0 if "P_elec (kW)" in dfG.columns else 1,
                horizontal=True,
            )

        with c2:
            eta_choice = st.radio(
                "Eficiencia",
                ["eta_cog_global (%)", "eta_ciclo_electrico (%)"],
                index=0 if "eta_cog_global (%)" in dfG.columns else 1,
                horizontal=True,
            )

        with c3:
            show_heatmap = st.checkbox("Mostrar heatmap (opcional)", value=False)

        c4, c5 = st.columns(2)
        with c4:
            muni_sel = st.multiselect("Municipios", sorted(dfG["Municipio"].dropna().unique().tolist()))
        with c5:
            turb_sel = st.multiselect("Turbinas", sorted(dfG["Turbina"].dropna().unique().tolist()))

        c6, c7 = st.columns(2)
        with c6:
            pmin, pmax = safe_range(dfG[pot_choice])
            pot_rng = st.slider(f"Rango {pot_choice}", pmin, pmax, (pmin, pmax))
        with c7:
            emin, emax = safe_range(dfG[eta_choice])
            eta_rng = st.slider(f"Rango {eta_choice}", emin, emax, (emin, emax))

        c8, c9 = st.columns(2)
        with c8:
            hmin, hmax = safe_range(dfG["Altitud (m)"])
            alt_rng = st.slider("Rango Altitud (m)", hmin, hmax, (hmin, hmax))
        with c9:
            smin, smax = safe_range(dfG["SFC_elec (kg/kWh)"])
            sfc_rng = st.slider("Rango SFC (kg/kWh)", smin, smax, (smin, smax))

        only_ok = st.checkbox("Solo casos OK (Temp_GC_OK y Temp_usuario_OK)", value=False)

    # --------- Filtro ---------
    mask = (
        dfG[pot_choice].between(*pot_rng)
        & dfG[eta_choice].between(*eta_rng)
        & dfG["Altitud (m)"].between(*alt_rng)
        & dfG["SFC_elec (kg/kWh)"].between(*sfc_rng)
    )
    if muni_sel:
        mask &= dfG["Municipio"].isin(muni_sel)
    if turb_sel:
        mask &= dfG["Turbina"].isin(turb_sel)
    if only_ok and {"Temp_GC_OK", "Temp_usuario_OK"}.issubset(dfG.columns):
        mask &= (dfG["Temp_GC_OK"].astype(bool) & dfG["Temp_usuario_OK"].astype(bool))

    dff = dfG.loc[mask].copy()
    st.caption(f"Filtrado activo → {len(dff)} filas")

    color_ok = "Semaforo_HX" if "Semaforo_HX" in dff.columns else None
    
    
    
    # ===== Colores consistentes por Turbina (en todas las gráficas) =====
    from itertools import cycle
    # paleta base (se repite si hay más turbinas)
    base_palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.Set2
        + px.colors.qualitative.Dark24
    )

    turbinas_unicas = sorted(dfG["Turbina"].dropna().unique().tolist())
    palette_cycle = cycle(base_palette)
    turb_color_map = {t: c for t, c in zip(turbinas_unicas, palette_cycle)}

    # helper para aplicar el mismo layout interactivo a todas las figs
    def tune(fig, title):
        fig.update_layout(
            title=title,
            legend=dict(
                title="Turbina",
                itemclick="toggle",            # clic = ocultar/mostrar
                itemdoubleclick="toggleothers" # doble clic = aislar
            )
        )
        return fig
    
    
    def hv(cols, df):
        return [c for c in cols if c in df.columns]

    common_hover = hv(
        ["Municipio","Turbina","Altitud (m)","eta_cog_global (%)","SFC_elec (kg/kWh)","Q_in (kW)","P_elec (kW)","P_net (kW)"],
        dff
    )    

    # -------------------------------------------------------------------
    # 5) Gráficas
    # -------------------------------------------------------------------
    st.header("5) Gráficas")

    # 1) Pot vs Eficiencia
    fig1 = px.scatter(
        dff, x=pot_choice, y=eta_choice,
        color="Turbina",
        color_discrete_map=turb_color_map,
        symbol="Semaforo_HX" if "Semaforo_HX" in dff.columns else None,
        hover_data=common_hover,
    )
    st.plotly_chart(tune(fig1, "Pot vs Eficiencia"), use_container_width=True)

    # 2) Qútil vs Qin
    if "Q_user (kW)" in dff.columns:
        fig2 = px.scatter(
            dff, x="Q_in (kW)", y="Q_user (kW)",
            color="Turbina",
            color_discrete_map=turb_color_map,
            symbol="Semaforo_HX" if "Semaforo_HX" in dff.columns else None,
            hover_data=common_hover,
        )
        st.plotly_chart(tune(fig2, "Qútil vs Qin"), use_container_width=True)

    # 3) Pot vs Qin
    fig3 = px.scatter(
        dff, x="Q_in (kW)", y=pot_choice,
        color="Turbina",
        color_discrete_map=turb_color_map,
        symbol="Semaforo_HX" if "Semaforo_HX" in dff.columns else None,
        hover_data=common_hover,
    )
    st.plotly_chart(tune(fig3, "Pot vs Qin"), use_container_width=True)

    # 4) SFC vs Pot
    fig4 = px.scatter(
        dff, x=pot_choice, y="SFC_elec (kg/kWh)",
        color="Turbina",
        color_discrete_map=turb_color_map,
        symbol="Semaforo_HX" if "Semaforo_HX" in dff.columns else None,
        hover_data=common_hover,
    )
    st.plotly_chart(tune(fig4, "SFC vs Pot"), use_container_width=True)

    # 5) η_global vs Altitud
    if "eta_cog_global (%)" in dff.columns:
        fig5 = px.scatter(
            dff, x="Altitud (m)", y="eta_cog_global (%)",
            color="Turbina",
            color_discrete_map=turb_color_map,
            symbol="Semaforo_HX" if "Semaforo_HX" in dff.columns else None,
            hover_data=common_hover,
        )
        st.plotly_chart(tune(fig5, "η_global vs Altitud"), use_container_width=True)

    # 6) Q_rec vs Altitud
    if "Q_user (kW)" in dff.columns:
        fig6 = px.scatter(
            dff, x="Altitud (m)", y="Q_user (kW)",
            color="Turbina",
            color_discrete_map=turb_color_map,
            symbol="Semaforo_HX" if "Semaforo_HX" in dff.columns else None,
            hover_data=common_hover,
        )
        st.plotly_chart(tune(fig6, "Q_rec vs Altitud"), use_container_width=True)

    # 7) 3D Pot–Alt–SFC
    fig7 = px.scatter_3d(
        dff, x=pot_choice, y="Altitud (m)", z="SFC_elec (kg/kWh)",
        color="Turbina",
        color_discrete_map=turb_color_map,
        symbol="Semaforo_HX" if "Semaforo_HX" in dff.columns else None,
        hover_name="Turbina",
        hover_data=common_hover,
    )
    st.plotly_chart(tune(fig7, "3D Pot–Alt–SFC (doble clic para aislar turbina)"), use_container_width=True)
    
         
            
    # ---- Gráfica especial: η_global vs Altitud (color por ambiente) ----
    st.subheader("Gráfica especial: η_global vs Altitud (color por ambiente)")

    # Añadir variables de ambiente al dff
    amb_cols = ["Municipio","Temperatura (°C)","Presión (bares)"]
    if all(c in df_mun.columns for c in amb_cols):
        dff_amb = dff.merge(df_mun[amb_cols], on="Municipio", how="left")
    else:
        dff_amb = dff.copy()
        if "Temperatura (°C)" not in dff_amb.columns: dff_amb["Temperatura (°C)"] = float("nan")
        if "Presión (bares)" not in dff_amb.columns:  dff_amb["Presión (bares)"]  = float("nan")

    color_var = st.radio("Variable de color", ["Temperatura (°C)","Presión (bares)"], index=0, horizontal=True)

    if "eta_cog_global (%)" in dff_amb.columns:
        fig_sp = px.scatter(
            dff_amb, x="Altitud (m)", y="eta_cog_global (%)",
            color=color_var, color_continuous_scale="Viridis",
            hover_data=hv(["Municipio","Turbina","Altitud (m)","eta_cog_global (%)","Temperatura (°C)","Presión (bares)","P_elec (kW)"], dff_amb)
        )
        fig_sp.update_layout(title="η_global vs Altitud (color = ambiente)")
        st.plotly_chart(fig_sp, use_container_width=True)
    else:
        st.info("No se encontró la columna 'eta_cog_global (%)' para esta gráfica.")

    # ---- 6) Heatmap opcional ----
    if show_heatmap:
        st.subheader("Heatmap (densidad) Pot vs Altitud – color = η_global (promedio en celda)")
        if not dff.empty and "eta_cog_global (%)" in dff.columns:
            hm = px.density_heatmap(
                dff, x=pot_choice, y="Altitud (m)", z="eta_cog_global (%)",
                nbinsx=25, nbinsy=25, histfunc="avg",
                title="Heatmap: promedio de η_global en celdas Pot×Alt"
            )
            st.plotly_chart(hm, use_container_width=True)
        else:
            st.info("No hay datos suficientes (o falta η_global) para construir el heatmap.")
