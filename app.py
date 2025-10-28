# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 02:32:09 2025

@author: Gabo San
"""

import io
import base64
import pandas as pd
import streamlit as st
import plotly.express as px
from CoolProp.CoolProp import PropsSI

import BraytonRI8 as w  # <--- motor único

# -------- utilidades UI --------
def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def safe_read_excel(path):
    try: return pd.read_excel(path)
    except Exception: return None

def safe_read_csv(path):
    try: return pd.read_csv(path)
    except Exception: return None

def safe_range(series, pad_ratio=0.1, pad_abs=1.0):
    s = series.dropna()
    if s.empty: return (0.0, 1.0)
    mn, mx = float(s.min()), float(s.max())
    if mn == mx:
        pad = max(abs(mn) * pad_ratio, pad_abs)
        return (mn - pad, mx + pad)
    return (mn, mx)

def make_excel_bytes(dfs: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        dfs.get("Estados", pd.DataFrame()).to_excel(xw, "Estados", index=False)
        dfs.get("Calculos", pd.DataFrame()).to_excel(xw, "Calculos", index=False)
        dfs.get("Cogeneracion", pd.DataFrame()).to_excel(xw, "Cogeneracion", index=False)
        dfs.get("Validaciones", pd.DataFrame()).to_excel(xw, "Validaciones", index=False)
    buf.seek(0)
    return buf.read()

# -------- página --------
st.set_page_config(layout="wide")

# logo/encabezado (igual que tu versión)
try:
    st.columns([1,2,1])[1].image("logo_uam.png", width=300)
except Exception:
    pass

st.markdown("""
<div style="text-align:center; line-height:1.2;">
  <h2> UNIVERSIDAD AUTÓNOMA METROPOLITANA</h2>
  <p>Proyecto Terminal</p>
  <p><strong>Asesor:</strong> 🎓 Hernando Romero Paredes Rubio &nbsp;|&nbsp;
     <strong>Alumno:</strong> 🤖 Rolando Gabriel Garza Luna</p>
</div>
""", unsafe_allow_html=True)

st.title("🛠️ Simulación Ciclo Brayton + Cogeneración (full-físico)")

# -------- estado persistente --------
if "dfs" not in st.session_state:
    st.session_state.dfs = None

# -------- 1) Entrada de datos --------
st.header("1) Datos de entrada")

col1, col2 = st.columns(2)
with col1:
    f_mun = st.file_uploader("Sube Municipios_D.xlsx", type=["xlsx"], key="mun")
    df_mun = pd.read_excel(f_mun) if f_mun else (safe_read_excel("data/Municipios_D.xlsx") or pd.DataFrame())
    if f_mun is None: st.caption("Si no subes archivo, intento data/Municipios_D.xlsx")
with col2:
    f_tur = st.file_uploader("Sube Base_de_datos_turbinas_de_gas.csv", type=["csv"], key="tur")
    df_tur = pd.read_csv(f_tur) if f_tur else (safe_read_csv("data/Base_de_datos_turbinas_de_gas.csv") or pd.DataFrame())
    if f_tur is None: st.caption("Si no subes archivo, intento data/Base_de_datos_turbinas_de_gas.csv")

ok_inputs = (not df_mun.empty) and (not df_tur.empty)

# -------- 2) Parámetros (sliders) --------
st.header("2) Parámetros")

with st.form("form_params"):
    st.markdown("### HX & Agua")
    c1, c2, c3 = st.columns(3)
    with c1:
        U_GLOBAL = st.slider("U_GLOBAL (kW/m²·K)", 0.05, 20.0, float(w.U_GLOBAL), 0.01)
        A_GLOBAL = st.slider("A_GLOBAL (m²)", 1.0, 50.0, float(w.A_GLOBAL), 0.5)
        m_dot_w  = st.slider("ṁ_w (kg/s)", 0.1, 50.0, float(w.m_dot_w), 0.1)
    with c2:
        P7_BAR = st.slider("P7 (bar)", 2.0, 60.0, float(w.P7_BAR), 0.5)
        P8_BAR = st.slider("P8 (bar)", 0.1, 20.0, float(w.P8_BAR), 0.1)
        DELTA_SUBCOOL_C = st.slider("Subenfriamiento Δ (°C)", 0.0, 40.0, float(w.DELTA_SUBCOOL_C), 1.0)
    with c3:
        T7_USER_C = st.slider("T7 usuario (°C)", 60.0, 260.0, float(w.T7_USER_C), 1.0)

    st.markdown("### Brayton / Combustible / Restricción")
    c4, c5, c6 = st.columns(3)
    with c4:
        ETA_GEN = st.slider("η_gen (-)", 0.70, 1.00, float(w.ETA_GEN), 0.01)
        ETA_CALDERA = st.slider("η_caldera (-)", 0.50, 1.00, float(w.ETA_CALDERA), 0.01)
    with c5:
        PCI = st.slider("PCI (kJ/kg)", 20000, 60000, int(w.PCI), 500)
    with c6:
        T_GAS_OUT_MINC = st.slider("T5 mínima (°C)", 80, 180, int(w.T_GAS_OUT_MINC), 1)

    submitted = st.form_submit_button("Guardar parámetros")

if submitted:
    # ← asignamos directamente a las globales del motor
    w.U_GLOBAL = float(U_GLOBAL)
    w.A_GLOBAL = float(A_GLOBAL)
    w.m_dot_w  = float(m_dot_w)
    w.P7_BAR   = float(P7_BAR)
    w.P8_BAR   = float(P8_BAR)
    w.DELTA_SUBCOOL_C = float(DELTA_SUBCOOL_C)
    w.T7_USER_C = float(T7_USER_C)
    w.ETA_GEN = float(ETA_GEN)
    w.ETA_CALDERA = float(ETA_CALDERA)
    w.PCI = float(PCI)
    w.T_GAS_OUT_MINC  = float(T_GAS_OUT_MINC)
    w.T_GAS_OUT_MINK  = w.T_GAS_OUT_MINC + 273.15
    st.success("Parámetros aplicados al motor ✅")

# -------- 3) Barrido cartesiano (solo full-físico) --------
st.header("3) Ejecutar barrido (full-físico)")

cbtn1, cbtn2 = st.columns([1,4])
run_btn = cbtn1.button("🧮 Ejecutar", disabled=not ok_inputs)

def _run_cartesiano_full(df_mun, df_tur):
    # compatibilidad encabezados municipio
    dm = df_mun.copy()
    if "Altitud (m)" not in dm.columns and "Altitud (media)" in dm.columns:
        dm["Altitud (m)"] = dm["Altitud (media)"]
    dm["T1 (K)"]   = dm["Temperatura (°C)"] + 273.15
    dm["P1 (kPa)"] = dm["Presión (bares)"] * 100.0

    dt = df_tur.copy()

    # producto cartesiano
    dm["_k"]=1; dt["_k"]=1
    cross = pd.merge(dm, dt, on="_k").drop(columns="_k")

    estados, calculos, cogener = [], [], []

    rho_ISO = PropsSI("D","T",288.15,"P",101325.0,"Air")
    for _, r in cross.iterrows():
        # derating por densidad local
        rho_loc = PropsSI("D","T", float(r["T1 (K)"]), "P", float(r["P1 (kPa)"])*1000.0, "Air")
        m_ISO   = float(r.get("m_aire (kg/s)", r.get("m_aire_kg_s", 0.0)))
        m_loc   = m_ISO * (rho_loc/rho_ISO)

        est, calc, cog = w.simular_ciclo_mdot(
            T1=float(r["T1 (K)"]),
            P1_kPa=float(r["P1 (kPa)"]),
            r_p=float(r["r_p"]),
            T3=float(r["T3 (C)"]) + 273.15,
            eta_c=float(r["eta_c"]),
            eta_t=float(r["eta_t"]),
            m_dot_gas=float(m_loc),
            p7_bar=float(w.P7_BAR),
            p8_bar=float(w.P8_BAR),
            delta_subcool_c=float(w.DELTA_SUBCOOL_C),
            t7_user_c=float(w.T7_USER_C),
        )

        # etiquetas
        est.insert(0, "Municipio", r["Municipio"]); est.insert(1, "Turbina", r["Turbina"])
        calc.update({
            "Municipio": r["Municipio"], "Turbina": r["Turbina"],
            "Altitud (m)": r.get("Altitud (m)", float("nan")),
            "rho_local (kg/m3)": rho_loc, "m_dot_ISO_design (kg/s)": m_ISO, "m_dot_local (kg/s)": m_loc
        })
        cog.update({"Municipio": r["Municipio"], "Turbina": r["Turbina"], "Altitud (m)": r.get("Altitud (m)", float("nan"))})

        estados.append(est); calculos.append(calc); cogener.append(cog)

    df_est = pd.concat(estados, ignore_index=True) if estados else pd.DataFrame()
    df_calc = pd.DataFrame(calculos)
    df_cog  = pd.DataFrame(cogener)

    # validaciones mínimas (las que usa tu motor)
    valid = df_cog[[
        "Municipio","Turbina","Altitud (m)","Temp_GC_OK","Temp_usuario_OK","Semaforo_HX"
    ]].copy() if not df_cog.empty else pd.DataFrame()

    return df_est, df_calc, df_cog, valid

if run_btn:
    with st.spinner("Corriendo…"):
        df_est, df_calc, df_cog, df_valid = _run_cartesiano_full(df_mun, df_tur)
    st.session_state.dfs = {
        "Estados": df_est, "Calculos": df_calc,
        "Cogeneracion": df_cog, "Validaciones": df_valid
    }
    st.success(f"Listo: {len(df_calc)} casos.")

# -------- 4) Mostrar tablas + descarga --------
dfs_now = st.session_state.dfs
if dfs_now:
    tab1, tab2, tab3, tab4 = st.tabs(["Estados", "Cálculos", "Cogeneración", "Validaciones"])
    with tab1: st.dataframe(dfs_now["Estados"], use_container_width=True)
    with tab2: st.dataframe(dfs_now["Calculos"], use_container_width=True)
    with tab3: st.dataframe(dfs_now["Cogeneracion"], use_container_width=True)
    with tab4: st.dataframe(dfs_now["Validaciones"], use_container_width=True)

    st.download_button(
        "⬇️ Descargar Excel",
        data=make_excel_bytes(dfs_now),
        file_name="Resultados_Ciclo_Brayton.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# -------- 5) Filtros + gráficas (igual que antes) --------
st.divider()
st.header("4) Filtros para gráficas")

if not dfs_now:
    st.info("No hay resultados. Ejecuta primero el barrido.")
else:
    df_calc = dfs_now["Calculos"].copy()
    df_cog  = dfs_now["Cogeneracion"].copy()

    # derivados útiles
    PCI_val = float(getattr(w, "PCI", 50000.0))
    dfG = df_calc.copy()
    if "Altitud (m)" not in dfG.columns:
        if "P1 (kPa)" in dfG.columns:
            P_Pa = dfG["P1 (kPa)"]*1000.0
            dfG["Altitud (m)"] = 44330.0 * (1.0 - (P_Pa/101325.0)**0.1903)
        else:
            dfG["Altitud (m)"] = float("nan")
    if "m_dot_fuel (kg/s)" in dfG.columns:
        dfG["Q_in (kW)"] = dfG["m_dot_fuel (kg/s)"] * PCI_val
    else:
        dfG["Q_in (kW)"] = float("nan")
    if "P_elec (kW)" not in dfG.columns and "P_net (kW)" in dfG.columns:
        dfG["P_elec (kW)"] = dfG["P_net (kW)"]

    # controles
    with st.expander("Controles de visualización", expanded=True):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            pot_choice = st.radio("Potencia a graficar", ["P_elec (kW)","P_net (kW)"], index=0, horizontal=True)
        with c2:
            eta_choice = st.radio("Eficiencia", ["eta_cog_global (%)","eta_ciclo_electrico (%)"], index=0, horizontal=True)
        with c3:
            show_heatmap = st.checkbox("Heatmap", value=False)

        c4, c5 = st.columns(2)
        with c4:
            muni_sel = st.multiselect("Municipios", sorted(dfG["Municipio"].dropna().unique().tolist()))
        with c5:
            turb_sel = st.multiselect("Turbinas", sorted(dfG["Turbina"].dropna().unique().tolist()))

        c6, c7 = st.columns(2)
        with c6:
            pmin, pmax = safe_range(dfG[pot_choice]); pot_rng = st.slider(f"Rango {pot_choice}", pmin, pmax, (pmin, pmax))
        with c7:
            emin, emax = safe_range(dfG[eta_choice]); eta_rng = st.slider(f"Rango {eta_choice}", emin, emax, (emin, emax))

        c8, c9 = st.columns(2)
        with c8:
            hmin, hmax = safe_range(dfG["Altitud (m)"]); alt_rng = st.slider("Rango Altitud (m)", hmin, hmax, (hmin, hmax))
        with c9:
            smin, smax = safe_range(dfG.get("SFC_elec (kg/kWh)", pd.Series([0,1]))); sfc_rng = st.slider("Rango SFC (kg/kWh)", smin, smax, (smin, smax))

        only_ok = st.checkbox("Solo casos OK (Temp_GC_OK y Temp_usuario_OK)", value=False)

    # filtro
    mask = (
        dfG[pot_choice].between(*pot_rng)
        & dfG[eta_choice].between(*eta_rng)
        & dfG["Altitud (m)"].between(*alt_rng)
    )
    if "SFC_elec (kg/kWh)" in dfG.columns:
        mask &= dfG["SFC_elec (kg/kWh)"].between(*sfc_rng)
    if muni_sel: mask &= dfG["Municipio"].isin(muni_sel)
    if turb_sel: mask &= dfG["Turbina"].isin(turb_sel)
    if only_ok and {"Temp_GC_OK","Temp_usuario_OK"}.issubset(dfG.columns):
        mask &= (dfG["Temp_GC_OK"].astype(bool) & dfG["Temp_usuario_OK"].astype(bool))

    dff = dfG.loc[mask].copy()
    st.caption(f"Filtrado activo → {len(dff)} filas")

    # paleta estable por turbina
    base_palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set2 + px.colors.qualitative.Dark24
    turbinas = sorted(dfG["Turbina"].dropna().unique().tolist())
    turb_color_map = {t: base_palette[i % len(base_palette)] for i, t in enumerate(turbinas)}
    sym = "Semaforo_HX" if "Semaforo_HX" in dff.columns else None

    # gráficas
    def tune(fig, title):
        fig.update_layout(title=title, legend=dict(title="Turbina", itemclick="toggle", itemdoubleclick="toggleothers"))
        return fig

    hv_cols = [c for c in ["Municipio","Turbina","Altitud (m)","eta_cog_global (%)","SFC_elec (kg/kWh)","Q_in (kW)","P_elec (kW)","P_net (kW)"] if c in dff.columns]

    st.header("5) Gráficas")
    st.plotly_chart(tune(px.scatter(dff, x=pot_choice, y=eta_choice, color="Turbina",
                                    color_discrete_map=turb_color_map, symbol=sym, hover_data=hv_cols),
                         "Pot vs Eficiencia"), use_container_width=True)

    if "Q_user (kW)" in dff.columns:
        st.plotly_chart(tune(px.scatter(dff, x="Q_in (kW)", y="Q_user (kW)", color="Turbina",
                                        color_discrete_map=turb_color_map, symbol=sym, hover_data=hv_cols),
                             "Q útil vs Q_in"), use_container_width=True)

    st.plotly_chart(tune(px.scatter(dff, x="Q_in (kW)", y=pot_choice, color="Turbina",
                                    color_discrete_map=turb_color_map, symbol=sym, hover_data=hv_cols),
                         "Pot vs Q_in"), use_container_width=True)

    if "SFC_elec (kg/kWh)" in dff.columns:
        st.plotly_chart(tune(px.scatter(dff, x=pot_choice, y="SFC_elec (kg/kWh)", color="Turbina",
                                        color_discrete_map=turb_color_map, symbol=sym, hover_data=hv_cols),
                             "SFC vs Pot"), use_container_width=True)

    if "eta_cog_global (%)" in dff.columns:
        st.plotly_chart(tune(px.scatter(dff, x="Altitud (m)", y="eta_cog_global (%)", color="Turbina",
                                        color_discrete_map=turb_color_map, symbol=sym, hover_data=hv_cols),
                             "η_global vs Altitud"), use_container_width=True)

    if "Q_user (kW)" in dff.columns:
        st.plotly_chart(tune(px.scatter(dff, x="Altitud (m)", y="Q_user (kW)", color="Turbina",
                                        color_discrete_map=turb_color_map, symbol=sym, hover_data=hv_cols),
                             "Q_rec vs Altitud"), use_container_width=True)

    # 3D opcional
    try:
        st.plotly_chart(tune(px.scatter_3d(dff, x=pot_choice, y="Altitud (m)", z="SFC_elec (kg/kWh)",
                                           color="Turbina", color_discrete_map=turb_color_map,
                                           symbol=sym, hover_name="Turbina", hover_data=hv_cols),
                             "3D Pot–Alt–SFC"), use_container_width=True)
    except Exception:
        pass

    # Heatmap opcional
    if show_heatmap and "eta_cog_global (%)" in dff.columns:
        hm = px.density_heatmap(dff, x=pot_choice, y="Altitud (m)", z="eta_cog_global (%)",
                                nbinsx=25, nbinsy=25, histfunc="avg",
                                title="Heatmap: promedio de η_global en celdas Pot×Alt")
        st.plotly_chart(hm, use_container_width=True)
