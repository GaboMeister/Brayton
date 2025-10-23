# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 01:58:31 2025

@author: Gabo San
"""

# ri8_wrapper.py
# Adaptador entre la UI (Streamlit) y el motor BraytonRI8.
# Versión: smoke-0.2 (ejecución 1×1 + Excel en memoria)

from typing import Dict, Tuple, Optional
import io
import pandas as pd
import BraytonRI8 as ri8  # mismo directorio que app.py

# Columnas esperadas por BraytonRI8 (según tu script)
EXPECTED_MUN = {"Municipio", "Temperatura (°C)", "Presión (bares)"}
EXPECTED_TUR = {"Turbina", "r_p", "T3 (C)", "eta_c", "eta_t", "Potencia (kW)"}

R_AIR = 287.0         # J/(kg·K)
RHO_ISO = 1.225       # kg/m3  (ISA 15°C, 1 atm)
T_ISO = 288.15        # K
P_ISO_kPa = 101.325   # kPa


def expected_columns() -> Dict[str, set]:
    return {
        "Municipios_D.xlsx": set(EXPECTED_MUN),
        "Base_de_datos_turbinas_de_gas.csv": set(EXPECTED_TUR),
    }


def validate_inputs(df_mun: pd.DataFrame, df_tur: pd.DataFrame) -> Tuple[Dict[str, list], bool]:
    mun_cols = set(df_mun.columns)
    tur_cols = set(df_tur.columns)

    missing_mun = sorted(list(EXPECTED_MUN - mun_cols))
    extra_mun   = sorted(list(mun_cols - EXPECTED_MUN))
    missing_tur = sorted(list(EXPECTED_TUR - tur_cols))
    extra_tur   = sorted(list(tur_cols - EXPECTED_TUR))

    report = {
        "Municipios_D.xlsx": {"faltantes": missing_mun, "sobrantes": extra_mun},
        "Base_de_datos_turbinas_de_gas.csv": {"faltantes": missing_tur, "sobrantes": extra_tur},
    }
    ok = (len(missing_mun) == 0) and (len(missing_tur) == 0)
    return report, ok


def params_preview_dict(
    *, U_GLOBAL: float, A_GLOBAL: float, m_dot_w: float,
    P7_BAR: float, P8_BAR: float, DELTA_SUBCOOL_C: float, T7_USER_C: float,
    ETA_GEN: float, ETA_CALDERA: float, PCI: float, T_GAS_OUT_MINC: float,
) -> Dict[str, float]:
    return {
        "U_GLOBAL (kW/m²·K)": U_GLOBAL, "A_GLOBAL (m²)": A_GLOBAL, "m_dot_w (kg/s)": m_dot_w,
        "P7_BAR (bar)": P7_BAR, "P8_BAR (bar)": P8_BAR, "DELTA_SUBCOOL_C (°C)": DELTA_SUBCOOL_C,
        "T7_USER_C (°C)": T7_USER_C, "ETA_GEN (-)": ETA_GEN, "ETA_CALDERA (-)": ETA_CALDERA,
        "PCI (kJ/kg)": PCI, "T_GAS_OUT_MINC (°C)": T_GAS_OUT_MINC,
    }


# ---------- Ejecución real (1×1 o más) ----------
def set_globals_from_params(params: Dict[str, float]) -> None:
    """Escribe sliders en los globals del motor BraytonRI8."""
    ri8.U_GLOBAL        = float(params["U_GLOBAL"])
    ri8.A_GLOBAL        = float(params["A_GLOBAL"])
    ri8.m_dot_w         = float(params["m_dot_w"])
    ri8.P7_BAR          = float(params["P7_BAR"])
    ri8.P8_BAR          = float(params["P8_BAR"])
    ri8.DELTA_SUBCOOL_C = float(params["DELTA_SUBCOOL_C"])
    ri8.T7_USER_C       = float(params["T7_USER_C"])
    ri8.ETA_GEN         = float(params["ETA_GEN"])
    ri8.ETA_CALDERA     = float(params["ETA_CALDERA"])
    ri8.PCI             = float(params["PCI"])
    ri8.T_GAS_OUT_MINC  = float(params["T_GAS_OUT_MINC"])
    ri8.T_GAS_OUT_MINK  = ri8.T_GAS_OUT_MINC + 273.15  # importante para validación


def _build_validaciones(df_calc: pd.DataFrame, df_cog: pd.DataFrame) -> pd.DataFrame:
    """Replica la lógica de 'Validaciones' del main() para cualquier número de filas."""
    valid_left = df_calc[[
        "Municipio","Turbina","T4 (K)","T5_gas_out (K)","Q_gc (kW)","Q_user (kW)","eta_cog_global (%)"
    ]].copy()

    valid = valid_left.merge(
        df_cog[[
            "Municipio","Turbina",
            "T7_obj (K)","T7_real (K)","Tsat7 (K)",
            "Temp_GC_OK","Temp_usuario_OK","Semaforo_HX",
            "Q_pre (kW)","Q_plateau_NTU (kW)"
        ]],
        on=["Municipio","Turbina"], how="left"
    )

    valid["T5_OK (>=120C)"] = valid["T5_gas_out (K)"] >= (120.0 + 273.15)
    valid["T7_OK"]          = valid["T7_real (K)"]    >= (valid["T7_obj (K)"] - 1e-6)
    valid["Q_avail_NTU (kW)"] = valid["Q_pre (kW)"].fillna(0) + valid["Q_plateau_NTU (kW)"].fillna(0)
    valid["T7_gap (K)"]       = (valid["T7_obj (K)"] - valid["T7_real (K)"]).clip(lower=0)
    valid["T5_margin (K)"]    = valid["T5_gas_out (K)"] - (120.0 + 273.15)

    epsK = 0.5
    def _regimen(t7, ts):
        if pd.isna(t7) or pd.isna(ts): return "N/A"
        if t7 < ts - epsK: return "Subenfriado"
        elif abs(t7 - ts) <= epsK: return "Ebullición (plateau)"
        else: return "Sobrecalentado"

    valid["Regimen_HX"] = [_regimen(t7, ts) for t7, ts in zip(valid["T7_real (K)"], valid["Tsat7 (K)"])]
    return valid


def run_cartesiano(
    df_mun: pd.DataFrame,
    df_tur: pd.DataFrame,
    params: Dict[str, float],
    *,
    muni_sel: Optional[list] = None,
    turb_sel: Optional[list] = None,
    limit: Optional[int] = None,
    progress_cb: Optional[callable] = None,
    mode: str = "mdot",  # <--- "mdot" (full físico) o "target" (placa)
):
    """
    Barrido Municipio × Turbina.
    mode="mdot":   caudal volumétrico fijo (full físico) usando simular_ciclo_mdot(...)
    mode="target": objetivo de potencia de placa en cada caso (simular_ciclo(...))
    Devuelve: df_est, df_calc, df_cog, df_valid
    """
    set_globals_from_params(params)

    dfm = df_mun.copy()
    dft = df_tur.copy()

    if "Altitud (m)" not in dfm.columns and "Altitud (media)" in dfm.columns:
        dfm["Altitud (m)"] = dfm["Altitud (media)"]

    dfm["T1 (K)"]   = dfm["Temperatura (°C)"] + 273.15
    dfm["P1 (kPa)"] = dfm["Presión (bares)"] * 100.0  # o 1e2

    if muni_sel:
        dfm = dfm[dfm["Municipio"].isin(muni_sel)]
    if turb_sel:
        dft = dft[dft["Turbina"].isin(turb_sel)]

    dfm["_k"] = 1; dft["_k"] = 1
    cross = pd.merge(dfm, dft, on="_k").drop(columns="_k")

    if limit is not None and limit > 0:
        cross = cross.head(limit)

    total = len(cross)
    done = 0

    estados_list, calc_list, cog_list = [], [], []

    cache_Wele_ISO = {}  # solo para mode="mdot"

    for r in cross.to_dict("records"):
        P_name = r["Potencia (kW)"]

        if mode == "target":
            # ---------- MODO SIMPLIFICADO: objetivo de placa ----------
            est, calc, cog = ri8.simular_ciclo(
                T1=r["T1 (K)"], P1_kPa=r["P1 (kPa)"],
                r_p=r["r_p"], T3=r["T3 (C)"], eta_c=r["eta_c"], eta_t=r["eta_t"],
                P_ele=P_name,
                p7_bar=ri8.P7_BAR, p8_bar=ri8.P8_BAR,
                delta_subcool_c=ri8.DELTA_SUBCOOL_C, t7_user_c=ri8.T7_USER_C
            )
            # etiquetas
            calc.update({
                "Municipio": r["Municipio"], "Turbina": r["Turbina"],
                "Altitud (m)": r.get("Altitud (m)", float("nan")),
                "Potencia etiqueta (kW)": P_name,
                "rho_local (kg/m3)": (r["P1 (kPa)"]*1000.0)/(287.0*r["T1 (K)"]),
                "m_dot_ISO_design (kg/s)": float("nan"),
                "m_dot_local (kg/s)": calc.get("m_dot_gas (kg/s)", float("nan")),
                "P_elec_capped?": False,
            })

        else:
            # ---------- MODO FULL FÍSICO: caudal volumétrico fijo ----------
            P1_Pa = r["P1 (kPa)"] * 1000.0
            T1_K  = r["T1 (K)"]
            rho_l = P1_Pa / (R_AIR * T1_K)

            key_t = (r["Turbina"], r["r_p"], r["T3 (C)"], r["eta_c"], r["eta_t"])
            if key_t not in cache_Wele_ISO:
                _, calc_iso, _ = ri8.simular_ciclo(
                    T1=T_ISO, P1_kPa=P_ISO_kPa,
                    r_p=r["r_p"], T3=r["T3 (C)"], eta_c=r["eta_c"], eta_t=r["eta_t"],
                    P_ele=P_name,
                    p7_bar=ri8.P7_BAR, p8_bar=ri8.P8_BAR,
                    delta_subcool_c=ri8.DELTA_SUBCOOL_C, t7_user_c=ri8.T7_USER_C
                )
                mdot_iso_dummy = calc_iso.get("m_dot_gas (kg/s)", None)
                W_ele_ISO = (P_name / mdot_iso_dummy) if mdot_iso_dummy and mdot_iso_dummy>0 else 0.0
                cache_Wele_ISO[key_t] = W_ele_ISO
            else:
                W_ele_ISO = cache_Wele_ISO[key_t]

            m_dot_ISO_design = (P_name / W_ele_ISO) if W_ele_ISO>0 else 0.0
            m_dot_local = m_dot_ISO_design * (rho_l / RHO_ISO)

            # Derating extra empírico sobre caudal (opcional)
            alpha = 0.0  # 0.8–1.1 si te lo piden; por defecto apagado
            if alpha != 0.0:
                m_dot_local *= (rho_l / RHO_ISO) ** alpha

            # Corrida física con ese caudal
            est, calc, cog = ri8.simular_ciclo_mdot(
                T1=T1_K, P1_kPa=r["P1 (kPa)"],
                r_p=r["r_p"], T3=r["T3 (C)"], eta_c=r["eta_c"], eta_t=r["eta_t"],
                m_dot_gas=m_dot_local,
                p7_bar=ri8.P7_BAR, p8_bar=ri8.P8_BAR,
                delta_subcool_c=ri8.DELTA_SUBCOOL_C, t7_user_c=ri8.T7_USER_C
            )

            # Clipping por placa/mínimo con recálculo
            P_min  = 0.10 * P_name
            P_out  = float(calc.get("P_elec (kW)", 0.0))
            W_ele_local = float(calc.get("W_ele (kJ/kg)", 0.0))

            need_rerun = False
            if W_ele_local > 0:
                if P_out > P_name:
                    m_dot_cap = P_name / W_ele_local; need_rerun = True
                elif P_out < P_min:
                    m_dot_cap = P_min / W_ele_local; need_rerun = True

            if need_rerun:
                est, calc, cog = ri8.simular_ciclo_mdot(
                    T1=T1_K, P1_kPa=r["P1 (kPa)"],
                    r_p=r["r_p"], T3=r["T3 (C)"], eta_c=r["eta_c"], eta_t=r["eta_t"],
                    m_dot_gas=m_dot_cap,
                    p7_bar=ri8.P7_BAR, p8_bar=ri8.P8_BAR,
                    delta_subcool_c=ri8.DELTA_SUBCOOL_C, t7_user_c=ri8.T7_USER_C
                )
                P_out = float(calc.get("P_elec (kW)", 0.0))

            calc.update({
                "Municipio": r["Municipio"], "Turbina": r["Turbina"],
                "Altitud (m)": r.get("Altitud (m)", float("nan")),
                "Potencia etiqueta (kW)": P_name,
                "rho_local (kg/m3)": rho_l,
                "m_dot_ISO_design (kg/s)": m_dot_ISO_design,
                "m_dot_local (kg/s)": m_dot_local,
                "P_elec_capped?": (P_out >= P_name-1e-6 or P_out <= P_min+1e-6),
            })

        # Insertar etiquetas en estados / acumular
        est.insert(0, "Municipio", r["Municipio"])
        est.insert(1, "Turbina",   r["Turbina"])
        estados_list.append(est)
        cog.update({"Municipio": r["Municipio"], "Turbina": r["Turbina"], "Altitud (m)": r.get("Altitud (m)", float("nan"))})
        calc_list.append(calc); cog_list.append(cog)

        done += 1
        if progress_cb:
            progress_cb(done, total)

    df_est  = pd.concat(estados_list, ignore_index=True) if estados_list else pd.DataFrame()
    df_calc = pd.DataFrame(calc_list)
    df_cog  = pd.DataFrame(cog_list)
    df_valid = _build_validaciones(df_calc, df_cog) if not df_calc.empty else pd.DataFrame()
    return df_est, df_calc, df_cog, df_valid


def run(
    df_mun: pd.DataFrame,
    df_tur: pd.DataFrame,
    params: Dict[str, float],
    *,
    sample: Optional[int] = 1,
):
    """
    Compatibilidad 1×1 / N×1: delega al barrido con limit=sample (sin progreso).
    """
    return run_cartesiano(
        df_mun, df_tur, params,
        muni_sel=None, turb_sel=None,
        limit=sample if sample and sample > 0 else None,
        progress_cb=None,
    )


def make_excel_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
    """Crea un Excel en memoria con hojas: Estados, Calculos, Cogeneracion, Validaciones."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        dfs.get("Estados", pd.DataFrame()).to_excel(w,  sheet_name="Estados",       index=False)
        dfs.get("Calculos", pd.DataFrame()).to_excel(w, sheet_name="Calculos",      index=False)
        dfs.get("Cogeneracion", pd.DataFrame()).to_excel(w, sheet_name="Cogeneracion", index=False)
        dfs.get("Validaciones", pd.DataFrame()).to_excel(w, sheet_name="Validaciones", index=False)
    buf.seek(0)
    return buf.read()
