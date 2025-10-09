# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:18:39 2025

@author: Gabo San
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
from io import BytesIO

# =======================
# 1) CoolProp con caché
# =======================
from functools import lru_cache
from CoolProp.CoolProp import PropsSI as _PropsSI

# Granularidad de redondeo (ajustable)
_QSTEP = {"T": 0.1, "P": 100.0, "H": 10.0, "S": 1e-4, "D": 1e-3, "Q": 1e-3}
def _q(name: str, x):
    if not isinstance(x, (int, float)):
        return x
    q = _QSTEP.get(name, 1e-6)
    return round(float(x) / q) * q

@lru_cache(maxsize=200_000)
def _props_cached(output, n1, v1, n2, v2, fluid):
    return float(_PropsSI(output, n1, _q(n1, v1), n2, _q(n2, v2), fluid))

def Props(output, name1, val1, name2, val2, fluid):
    """Wrapper de PropsSI con caché + redondeo (misma interfaz)."""
    return _props_cached(output, name1, val1, name2, val2, fluid)

# =======================
# 2) Parámetros globales
# =======================
FLUIDO_GAS = "Air"
FLUIDO_W   = "Water"

# Intercambiador (HX) y lazo de agua
ETA_UT         = 0.85          # (-) fracción utilizable del calor de gases
T_GAS_OUT_MINK = 393.15        # K (límite por corrosión = 120 °C)
DTMIN_HX_C     = 10.0          # °C (pinch mínimo del intercambiador)

# Entrega al usuario
P7_BAR          = 10.0         # bar
T7_USER_C       = 180.0        # °C (setpoint del usuario)
P8_BAR          = 2.0          # bar (retorno)
DELTA_SUBCOOL_C = 15.0         # °C (retorno = Tsat(P8) - 15 °C)

# Generador / caldera / combustible
ETA_GEN  = 0.95
ETA_CALD = 0.85
PCI      = 50_000.0            # kJ/kg

# =======================
# 3) Núcleo: SOLO propiedades (lo caro)
# =======================
def simular_ciclo_core(
    T1: float, P1_kPa: float, r_p: float, T3: float,
    eta_c: float, eta_t: float, *,
    eta_gen: float = ETA_GEN, eta_cald: float = ETA_CALD,
    eta_ut: float = ETA_UT,
    t_gas_out_min: float = T_GAS_OUT_MINK,   # K
    dtmin_hx_c: float = DTMIN_HX_C,          # °C (≡ K en diferencias)
    p7_bar: float = P7_BAR, t7_user_c: float = T7_USER_C,
    p8_bar: float = P8_BAR, delta_subcool_c: float = DELTA_SUBCOOL_C
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Devuelve:
      - DataFrame 'estados' (1–8) con T,h,s,P
      - dict 'core' con magnitudes por kg (Q_in, W_net, W_ele, Q_sum), h4,h5,h6,h7,h8,
        T4,T5,T6,T7, eta_ut, dtmin, banderas Temp_GC_OK/Temp_usuario_OK, márgenes dT_hot/cold_approach
        y alcanzabilidad: T7_obj, T7_max, T7_gap, Semaforo_HX.
    NO calcula flujos; eso se vectoriza en main().
    """
    # --- Presiones ---
    P1 = P1_kPa * 1e3
    P2 = P3 = P1 * r_p
    P4 = P1

    # --- Estado 1 ---
    h1 = Props("H", "T", T1, "P", P1, FLUIDO_GAS) / 1000
    s1 = Props("S", "T", T1, "P", P1, FLUIDO_GAS) / 1000

    # 1->2 Compresor (ηc)
    s1_J = Props("S", "T", T1, "P", P1, FLUIDO_GAS)
    T2s  = Props("T", "S", s1_J, "P", P2, FLUIDO_GAS)
    T2   = T1 + (T2s - T1) / eta_c
    h2   = Props("H", "T", T2, "P", P2, FLUIDO_GAS) / 1000
    s2   = Props("S", "T", T2, "P", P2, FLUIDO_GAS) / 1000

    # 2->3 Cámara de combustión
    h3 = Props("H", "T", T3, "P", P3, FLUIDO_GAS) / 1000
    s3 = Props("S", "T", T3, "P", P3, FLUIDO_GAS) / 1000

    # 3->4 Turbina (ηt)
    s3_J = Props("S", "T", T3, "P", P3, FLUIDO_GAS)
    T4s  = Props("T", "S", s3_J, "P", P4, FLUIDO_GAS)
    T4   = T3 - eta_t * (T3 - T4s)
    h4   = Props("H", "T", T4, "P", P4, FLUIDO_GAS) / 1000
    s4   = Props("S", "T", T4, "P", P4, FLUIDO_GAS) / 1000

    # Balances por kg
    Q_in, Q_out = h3 - h2, h4 - h1
    W_comp, W_turb = h2 - h1, h3 - h4
    W_net = W_turb - W_comp
    W_ele = W_net * eta_gen
    Q_sum = Q_in / eta_cald if eta_cald else np.nan

    # --- Lado agua 8 -> 6 -> 7 ---
    P8 = p8_bar * 1e5
    Tsat_P8 = Props("T", "P", P8, "Q", 0, FLUIDO_W)
    T8  = max(Tsat_P8 - delta_subcool_c, 273.15 + 5.0)
    h8  = Props("H", "T", T8, "P", P8, FLUIDO_W) / 1000
    s8  = Props("S", "T", T8, "P", P8, FLUIDO_W) / 1000

    P6  = p7_bar * 1e5
    rho8 = Props("D", "T", T8, "P", P8, FLUIDO_W)
    v8   = 1.0 / rho8
    dh_pump = v8 * (P6 - P8) / 1000.0
    h6  = h8 + dh_pump
    T6  = Props("T", "H", h6 * 1000.0, "P", P6, FLUIDO_W)
    s6  = Props("S", "H", h6 * 1000.0, "P", P6, FLUIDO_W) / 1000

    P7  = P6
    T7_target = t7_user_c + 273.15
    Tsat_P7   = Props("T", "P", P7, "Q", 0, FLUIDO_W)
    eps_K     = 0.5
    if abs(T7_target - Tsat_P7) <= eps_K:   # vapor saturado
        T7 = Tsat_P7
        h7 = Props("H", "P", P7, "Q", 1, FLUIDO_W) / 1000
        s7 = Props("S", "P", P7, "Q", 1, FLUIDO_W) / 1000
    else:                                   # subenfriado / sobrecalentado
        T7 = T7_target
        h7 = Props("H", "T", T7, "P", P7, FLUIDO_W) / 1000
        s7 = Props("S", "T", T7, "P", P7, FLUIDO_W) / 1000

    # --- Intercambiador: Pinch y salida de gases (T5) ---
    DTminK = dtmin_hx_c  # (°C ≡ K para diferencias)
    T5_cold_pinch = T6 + DTminK
    T5 = max(t_gas_out_min, T5_cold_pinch)   # respeta corrosión y pinch frío

    h5 = Props("H", "T", T5, "P", P4, FLUIDO_GAS) / 1000
    s5 = Props("S", "T", T5, "P", P4, FLUIDO_GAS) / 1000

    # Márgenes de acercamiento y banderas
    dT_hot_approach  = T4 - T7                 # extremo caliente (turbina vs usuario)
    dT_cold_approach = T5 - T6                 # extremo frío (gases vs retorno-bomba)
    ok_gc_temp       = (T5 >= t_gas_out_min)   # debería ser True por construcción
    ok_user_temp     = (dT_hot_approach >= DTminK)

    # Métrica de alcanzabilidad del setpoint
    T7_obj = T7                                  # lo pedido/definido
    T7_max = T4 - DTminK                         # máximo físico alcanzable con pinch
    T7_gap = T7_obj - T7_max                     # >0 => no se alcanza el setpoint

    # Semáforo resumido
    if (not ok_gc_temp) and (not ok_user_temp):
        semaforo = "Ambos"
    elif not ok_gc_temp:
        semaforo = "Límite GC"
    elif not ok_user_temp:
        semaforo = "Límite usuario"
    else:
        semaforo = "OK"

    status = "OK" if (np.isfinite(W_ele) and W_ele > 0) else "INV_Wnet<=0"

    # Estados para exportar
    estados = pd.DataFrame([
        {"Estado":1,"Fluido":FLUIDO_GAS,"P (kPa)":P1_kPa,     "T (K)":T1,"h (kJ/kg)":round(h1,3),"s (kJ/kg·K)":round(s1,5)},
        {"Estado":2,"Fluido":FLUIDO_GAS,"P (kPa)":P1_kPa*r_p, "T (K)":T2,"h (kJ/kg)":round(h2,3),"s (kJ/kg·K)":round(s2,5)},
        {"Estado":3,"Fluido":FLUIDO_GAS,"P (kPa)":P1_kPa*r_p, "T (K)":T3,"h (kJ/kg)":round(h3,3),"s (kJ/kg·K)":round(s3,5)},
        {"Estado":4,"Fluido":FLUIDO_GAS,"P (kPa)":P1_kPa,     "T (K)":T4,"h (kJ/kg)":round(h4,3),"s (kJ/kg·K)":round(s4,5)},
        {"Estado":5,"Fluido":FLUIDO_GAS,"P (kPa)":P1_kPa,     "T (K)":T5,"h (kJ/kg)":round(h5,3),"s (kJ/kg·K)":round(s5,5)},
        {"Estado":6,"Fluido":FLUIDO_W,  "P (kPa)":P6/1e3,     "T (K)":T6,"h (kJ/kg)":round(h6,3),"s (kJ/kg·K)":round(s6,5)},
        {"Estado":7,"Fluido":FLUIDO_W,  "P (kPa)":P7/1e3,     "T (K)":T7,"h (kJ/kg)":round(h7,3),"s (kJ/kg·K)":round(s7,5)},
        {"Estado":8,"Fluido":FLUIDO_W,  "P (kPa)":P8/1e3,     "T (K)":T8,"h (kJ/kg)":round(h8,3),"s (kJ/kg·K)":round(s8,5)},
    ])

    core = {
        # por kg
        "Q_in (kJ/kg)": Q_in, "Q_out (kJ/kg)": Q_out,
        "W_comp (kJ/kg)": W_comp, "W_turb (kJ/kg)": W_turb,
        "W_net (kJ/kg)": W_net, "W_ele (kJ/kg)": W_ele,
        "Q_sum (kJ/kg)": Q_sum,
        # estados clave para flujos
        "h4": h4, "h5": h5, "h6": h6, "h7": h7, "h8": h8,
        "T4 (K)": T4, "T5_gas_out (K)": T5, "T6 (K)": T6, "T7 (K)": T7,
        # HX realism
        "eta_HX_utilizacion (-)": eta_ut,
        "DTmin_HX_C": dtmin_hx_c,
        "dT_hot_approach (K)": dT_hot_approach,
        "dT_cold_approach (K)": dT_cold_approach,
        "ok_gc_temp": ok_gc_temp,
        "ok_user_temp": ok_user_temp,
        # Alcanzabilidad usuario
        "T7_obj (K)": T7_obj,
        "T7_max (K)": T7_max,
        "T7_gap (K)": T7_gap,
        "Semaforo_HX": semaforo,
        # estado general del punto
        "status": status,
    }
    return estados, core

def simular_batch(
    df_mun: pd.DataFrame,
    df_tur: pd.DataFrame,
    *,
    DTMIN_HX_C: float = DTMIN_HX_C,
    T7_user_C: float = T7_USER_C,
    P7_bar: float = P7_BAR,
    P8_bar: float = P8_BAR,
    Delta_subcool_C: float = DELTA_SUBCOOL_C,
    ETA_GEN_: float = ETA_GEN,
    ETA_CALD_: float = ETA_CALD,
    PCI_: float = PCI,
    write_excel: bool = True
):
    
    # --- Entradas defensivas/derivadas ---
    df_mun = df_mun.copy()
    if "T1 (K)" not in df_mun.columns:
        df_mun["T1 (K)"] = df_mun["Temperatura (°C)"] + 273.15
    if "P1 (kPa)" not in df_mun.columns:
        df_mun["P1 (kPa)"] = df_mun["Presión (bares)"] * 100

    df_tur = df_tur.copy()

    # Producto cartesiano
    df_mun["_k"] = 1
    df_tur["_k"] = 1
    df_cross = pd.merge(df_mun, df_tur, on="_k").drop(columns="_k")

    def _val(row, name, default):
        try:
            v = row.get(name, np.nan)
        except AttributeError:
            v = np.nan
        return float(v) if pd.notna(v) else default

    # --- Bucle SOLO propiedades (motor) ---
    estados_list, core_list = [], []
    for _, row in df_cross.iterrows():
        est, core = simular_ciclo_core(
            T1=row["T1 (K)"], P1_kPa=row["P1 (kPa)"], r_p=row["r_p"],
            T3=row["T3 (C)"], eta_c=row["eta_c"], eta_t=row["eta_t"],
            eta_gen=ETA_GEN_, eta_cald=ETA_CALD_,
            eta_ut=_val(row, "eta_ut", ETA_UT),
            t_gas_out_min=(_val(row, "T_gas_out_min_C", 120.0) + 273.15),
            dtmin_hx_c=_val(row, "DTmin_HX_C", DTMIN_HX_C),
            p7_bar=_val(row, "P7_bar", P7_bar),
            t7_user_c=_val(row, "T7_user_C", T7_user_C),
            p8_bar=_val(row, "P8_bar", P8_bar),
            delta_subcool_c=_val(row, "Delta_subcool_C", Delta_subcool_C),
        )
        # Etiquetas
        est.insert(0, "Municipio", row["Municipio"])
        est.insert(1, "Turbina", row["Turbina"])
        estados_list.append(est)

        core.update({
            "Municipio": row["Municipio"],
            "Turbina": row["Turbina"],
            "Potencia etiqueta (kW)": row["Potencia (kW)"],
        })
        core_list.append(core)

    df_est  = pd.concat(estados_list, ignore_index=True)
    df_core = pd.DataFrame(core_list)

    # --- Vectorizado: flujos, potencias, eficiencias ---
    mask_ok   = (df_core["status"] == "OK") & np.isfinite(df_core["W_ele (kJ/kg)"]) & (df_core["W_ele (kJ/kg)"] > 0)
    m_dot_gas = np.where(mask_ok, df_core["Potencia etiqueta (kW)"] / df_core["W_ele (kJ/kg)"], np.nan)
    P_net     = m_dot_gas * df_core["W_net (kJ/kg)"]
    P_elec    = df_core["Potencia etiqueta (kW)"]
    m_dot_fuel = (m_dot_gas * df_core["Q_sum (kJ/kg)"]) / PCI_

    delta_h_45 = np.maximum(df_core["h4"] - df_core["h5"], 0.0)
    Q_gc  = m_dot_gas * delta_h_45
    Q_vap = df_core["eta_HX_utilizacion (-)"] * Q_gc

    delta_h_76 = np.maximum(df_core["h7"] - df_core["h6"], 1e-6)
    m_dot_vap  = Q_vap / delta_h_76
    Q_user     = m_dot_vap * (df_core["h7"] - df_core["h8"])

    eta_electrica = np.where((m_dot_fuel > 0) & np.isfinite(m_dot_fuel),
                             P_elec / (m_dot_fuel * PCI_) * 100, np.nan)
    eta_cog = np.where((m_dot_fuel > 0) & np.isfinite(m_dot_fuel),
                       (P_elec + Q_user) / (m_dot_fuel * PCI_) * 100, np.nan)

    df_calc = pd.DataFrame({
        "Municipio": df_core["Municipio"],
        "Turbina":   df_core["Turbina"],
        "Q_in (kJ/kg)":  df_core["Q_in (kJ/kg)"],
        "Q_out (kJ/kg)": df_core["Q_out (kJ/kg)"],
        "W_comp (kJ/kg)": df_core["W_comp (kJ/kg)"],
        "W_turb (kJ/kg)": df_core["W_turb (kJ/kg)"],
        "W_net (kJ/kg)":  df_core["W_net (kJ/kg)"],
        "W_ele (kJ/kg)":  df_core["W_ele (kJ/kg)"],
        "Q_sum (kJ/kg)":  df_core["Q_sum (kJ/kg)"],
        "eta_ciclo_electrico (%)": np.where(
            np.isfinite(df_core["Q_sum (kJ/kg)"]) & (np.abs(df_core["Q_sum (kJ/kg)"]) > 1e-12),
            df_core["W_ele (kJ/kg)"] / df_core["Q_sum (kJ/kg)"] * 100, np.nan),

        "m_dot_gas (kg/s)": m_dot_gas,
        "P_net (kW)":       P_net,
        "P_elec (kW)":      P_elec,
        "m_dot_fuel (kg/s)": m_dot_fuel,
        "SFC_elec (kg/kWh)": np.where(P_elec > 0, (m_dot_fuel / P_elec) * 3600, np.nan),
        "Heat rate_elec (kJ/kWh)": np.where(P_elec > 0, (m_dot_fuel * PCI_) / P_elec, np.nan),
        "eta_electrica (%)": eta_electrica,

        "T4 (K)": df_core["T4 (K)"],
        "T5_gas_out (K)": df_core["T5_gas_out (K)"],
        "T6 (K)": df_core["T6 (K)"],
        "T7 (K)": df_core["T7 (K)"],
        "dT_hot_approach (K)":  df_core["dT_hot_approach (K)"],
        "dT_cold_approach (K)": df_core["dT_cold_approach (K)"],
        "Temp_GC_OK": df_core["ok_gc_temp"],
        "Temp_usuario_OK": df_core["ok_user_temp"],
        "DTmin_HX_C": df_core["DTmin_HX_C"],
        "T7_obj (K)": df_core["T7_obj (K)"],
        "T7_max (K)": df_core["T7_max (K)"],
        "T7_gap (K)": df_core["T7_gap (K)"],
        "Semaforo_HX": df_core["Semaforo_HX"],

        "Q_gc (kW)": Q_gc,
        "Q_vap (kW)": Q_vap,
        "m_dot_vap (kg/s)": m_dot_vap,
        "Q_user (kW)": Q_user,

        "eta_cog_global (%)": eta_cog,
        "eta_HX_utilizacion (-)": df_core["eta_HX_utilizacion (-)"],
        "status": df_core["status"],
        "Potencia etiqueta (kW)": P_elec,
    })

    df_cog = pd.DataFrame({
        "Municipio": df_core["Municipio"],
        "Turbina":   df_core["Turbina"],
        "P7 (bar)":  P7_bar,
        "T7_user (°C)": T7_user_C,
        "P8 (bar)":  P8_bar,
        "Delta_subcool (°C)": Delta_subcool_C,
        "DTmin_HX_C": df_core["DTmin_HX_C"],
        "T_gas_out_min (°C)": T_GAS_OUT_MINK - 273.15,
    })

    # --- Validaciones derivadas (sin volver a llamar CoolProp) ---
    h_pivot = df_est.pivot_table(index=["Municipio", "Turbina"], columns="Estado",
                                 values="h (kJ/kg)", aggfunc="first").rename(
        columns={4: "h4", 5: "h5", 6: "h6", 7: "h7", 8: "h8"}
    )
    T_pivot = df_est.pivot_table(index=["Municipio", "Turbina"], columns="Estado",
                                 values="T (K)", aggfunc="first").rename(
        columns={4: "T4_state", 5: "T5_state", 6: "T6_state", 7: "T7_state"}
    )
    dfv = (df_calc
           .merge(h_pivot, left_on=["Municipio", "Turbina"], right_index=True, how="left")
           .merge(T_pivot,   left_on=["Municipio", "Turbina"], right_index=True, how="left"))

    def _rel_err(calc, ref):
        calc = np.asarray(calc, dtype=float); ref = np.asarray(ref, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.where(np.abs(ref) > 1e-12, ref, np.nan)
            return np.abs((calc - ref) / denom)

    dfv["Q_gc_calc"]       = dfv["m_dot_gas (kg/s)"] * (dfv["h4"] - dfv["h5"])
    dfv["Q_vap_calc"]      = dfv["eta_HX_utilizacion (-)"] * dfv["Q_gc (kW)"]
    dfv["m_dot_vap_calc"]  = dfv["Q_vap (kW)"] / (dfv["h7"] - dfv["h6"])
    dfv["Q_user_calc"]     = dfv["m_dot_vap (kg/s)"] * (dfv["h7"] - dfv["h8"])
    dfv["eta_cog_calc"]    = (dfv["P_elec (kW)"] + dfv["Q_user (kW)"]) / (dfv["m_dot_fuel (kg/s)"] * PCI_) * 100

    dfv["err_Q_gc_rel"]      = _rel_err(dfv["Q_gc_calc"],  dfv["Q_gc (kW)"])
    dfv["err_Q_vap_rel"]     = _rel_err(dfv["Q_vap_calc"], dfv["Q_vap (kW)"])
    dfv["err_m_dot_vap_rel"] = _rel_err(dfv["m_dot_vap_calc"], dfv["m_dot_vap (kg/s)"])
    dfv["err_Q_user_rel"]    = _rel_err(dfv["Q_user_calc"], dfv["Q_user (kW)"])
    dfv["err_eta_cog_rel"]   = _rel_err(dfv["eta_cog_calc"], dfv["eta_cog_global (%)"])

    dfv["T5_violation"]   = dfv["T5_gas_out (K)"] < T_GAS_OUT_MINK
    dfv["inviable_auto"]  = (dfv["W_net (kJ/kg)"] <= 0) | (dfv["m_dot_gas (kg/s)"] <= 0) | (dfv["Q_vap (kW)"] < 0)
    dfv["Temp_GC_OK"]     = dfv["Temp_GC_OK"].astype(bool)
    dfv["Temp_usuario_OK"]= dfv["Temp_usuario_OK"].astype(bool)

    inviables = dfv[(dfv["status"].astype(str).str.startswith("INV")) | (dfv["inviable_auto"])][[
        "Municipio","Turbina","status","W_net (kJ/kg)","m_dot_gas (kg/s)",
        "Q_vap (kW)","Q_user (kW)","eta_cog_global (%)",
        "Temp_GC_OK","Temp_usuario_OK","Semaforo_HX"
    ]].sort_values(["Turbina","Municipio"]).reset_index(drop=True)

    mask_valid = ~(dfv["status"].astype(str).str.startswith("INV")) & ~(dfv["inviable_auto"]) & \
                 dfv["Temp_GC_OK"] & dfv["Temp_usuario_OK"] & pd.notna(dfv["eta_cog_global (%)"])
    top10 = dfv[mask_valid].sort_values("eta_cog_global (%)", ascending=False).head(10)[[
        "Municipio","Turbina","eta_cog_global (%)","P_elec (kW)","Q_user (kW)","m_dot_gas (kg/s)"
    ]].reset_index(drop=True)
    bottom10 = dfv[mask_valid].sort_values("eta_cog_global (%)", ascending=True).head(10)[[
        "Municipio","Turbina","eta_cog_global (%)","P_elec (kW)","Q_user (kW)","m_dot_gas (kg/s)"
    ]].reset_index(drop=True)

    # --- Excel en memoria opcional ---
    excel_bytes = None
    if write_excel:
        bio = BytesIO()
        with pd.ExcelWriter(bio) as w:
            df_est.to_excel(w, sheet_name="Estados", index=False)
            df_calc.to_excel(w, sheet_name="Calculos", index=False)
            df_cog.to_excel(w,  sheet_name="Cogeneracion", index=False)
            dfv[[
                "Municipio","Turbina","status","DTmin_HX_C",
                "T4 (K)","T7_obj (K)","T7_max (K)","T7 (K)","dT_hot_approach (K)","Temp_usuario_OK","T7_gap (K)","Semaforo_HX",
                "T6 (K)","T5_gas_out (K)","dT_cold_approach (K)","Temp_GC_OK","T5_violation",
                "Q_gc (kW)","Q_vap (kW)","m_dot_vap (kg/s)","Q_user (kW)",
                "err_Q_gc_rel","err_Q_vap_rel","err_m_dot_vap_rel","err_Q_user_rel","err_eta_cog_rel"
            ]].to_excel(w, sheet_name="validaciones", index=False)
            inviables.to_excel(w, sheet_name="inviables", index=False)
            top10.to_excel(w, sheet_name="top10", index=False)
            bottom10.to_excel(w, sheet_name="bottom10", index=False)
        excel_bytes = bio.getvalue()

    return df_est, df_calc, df_cog, dfv, inviables, top10, bottom10, excel_bytes


# =======================
# 4) Main vectorizado
# =======================
def main():
    # Asegurar cwd
    try:
        os.chdir(Path(__file__).resolve().parent)
    except Exception:
        pass

    # --- Datos de entrada ---
    df_mun = pd.read_excel("Municipios_D.xlsx")
    df_mun["T1 (K)"]   = df_mun["Temperatura (°C)"] + 273.15
    df_mun["P1 (kPa)"] = df_mun["Presión (bares)"] * 100

    df_tur = pd.read_csv("Base_de_datos_turbinas_de_gas.csv")

    # Producto cartesiano
    df_mun["_k"]=1; df_tur["_k"]=1
    df_cross = pd.merge(df_mun, df_tur, on="_k").drop(columns="_k")

    # Helper overrides por fila
    def _val(row, name, default):
        try:
            v = row.get(name, np.nan)
        except AttributeError:
            v = np.nan
        return float(v) if pd.notna(v) else default

    estados_list, core_list = [], []

    # ======= BUCLE (solo propiedades) =======
    for _, row in df_cross.iterrows():
        est, core = simular_ciclo_core(
            T1=row["T1 (K)"], P1_kPa=row["P1 (kPa)"], r_p=row["r_p"],
            T3=row["T3 (C)"], eta_c=row["eta_c"], eta_t=row["eta_t"],
            eta_gen=ETA_GEN, eta_cald=ETA_CALD,
            eta_ut=_val(row,"eta_ut",ETA_UT),
            t_gas_out_min=(_val(row,"T_gas_out_min_C",120.0)+273.15),
            dtmin_hx_c=_val(row, "DTmin_HX_C", DTMIN_HX_C),
            p7_bar=_val(row,"P7_bar",P7_BAR),
            t7_user_c=_val(row,"T7_user_C",T7_USER_C),
            p8_bar=_val(row,"P8_bar",P8_BAR),
            delta_subcool_c=_val(row,"Delta_subcool_C",DELTA_SUBCOOL_C),
        )

        # Etiquetas en estados
        est.insert(0,"Municipio",row["Municipio"])
        est.insert(1,"Turbina",  row["Turbina"])
        estados_list.append(est)

        core.update({
            "Municipio": row["Municipio"],
            "Turbina":   row["Turbina"],
            "Potencia etiqueta (kW)": row["Potencia (kW)"],
        })
        core_list.append(core)

    df_est  = pd.concat(estados_list, ignore_index=True)
    df_core = pd.DataFrame(core_list)

    # ======= VECTORIZADO: flujos y balances =======
    mask_ok = (df_core["status"]=="OK") & np.isfinite(df_core["W_ele (kJ/kg)"]) & (df_core["W_ele (kJ/kg)"]>0)

    # Flujo de gases requerido para la potencia de etiqueta
    m_dot_gas = np.where(mask_ok,
                         df_core["Potencia etiqueta (kW)"] / df_core["W_ele (kJ/kg)"],
                         np.nan)

    P_net  = m_dot_gas * df_core["W_net (kJ/kg)"]
    P_elec = df_core["Potencia etiqueta (kW)"]  # reporta la etiqueta

    # Combustible
    m_dot_fuel = (m_dot_gas * df_core["Q_sum (kJ/kg)"]) / PCI

    # Intercambiador y agua (con T5 realista ya calculado en core)
    delta_h_45 = np.maximum(df_core["h4"] - df_core["h5"], 0.0)
    Q_gc  = m_dot_gas * delta_h_45
    Q_vap = df_core["eta_HX_utilizacion (-)"] * Q_gc
    delta_h_76 = np.maximum(df_core["h7"] - df_core["h6"], 1e-6)
    m_dot_vap = Q_vap / delta_h_76
    Q_user = m_dot_vap * (df_core["h7"] - df_core["h8"])

    # Eficiencias
    eta_electrica = np.where((m_dot_fuel>0) & np.isfinite(m_dot_fuel),
                             P_elec / (m_dot_fuel * PCI) * 100, np.nan)
    eta_cog = np.where((m_dot_fuel>0) & np.isfinite(m_dot_fuel),
                       (P_elec + Q_user) / (m_dot_fuel * PCI) * 100, np.nan)

    # ======= Tablas de salida =======
    df_calc = pd.DataFrame({
        "Municipio": df_core["Municipio"],
        "Turbina":   df_core["Turbina"],

        # Brayton por kg
        "Q_in (kJ/kg)":  df_core["Q_in (kJ/kg)"],
        "Q_out (kJ/kg)": df_core["Q_out (kJ/kg)"],
        "W_comp (kJ/kg)":df_core["W_comp (kJ/kg)"],
        "W_turb (kJ/kg)":df_core["W_turb (kJ/kg)"],
        "W_net (kJ/kg)": df_core["W_net (kJ/kg)"],
        "W_ele (kJ/kg)": df_core["W_ele (kJ/kg)"],
        "Q_sum (kJ/kg)": df_core["Q_sum (kJ/kg)"],
        "eta_ciclo_electrico (%)": np.where(
            np.isfinite(df_core["Q_sum (kJ/kg)"]) & (np.abs(df_core["Q_sum (kJ/kg)"])>1e-12),
            df_core["W_ele (kJ/kg)"]/df_core["Q_sum (kJ/kg)"]*100, np.nan),

        # Potencias y combustible
        "m_dot_gas (kg/s)": m_dot_gas,
        "P_net (kW)":       P_net,
        "P_elec (kW)":      P_elec,           # etiqueta
        "m_dot_fuel (kg/s)":m_dot_fuel,
        "SFC_elec (kg/kWh)": np.where(P_elec>0, (m_dot_fuel / P_elec)*3600, np.nan),
        "Heat rate_elec (kJ/kWh)": np.where(P_elec>0, (m_dot_fuel*PCI)/P_elec, np.nan),
        "eta_electrica (%)": eta_electrica,

        # Intercambiador / usuario (realista + alcanzabilidad)
        "T4 (K)": df_core["T4 (K)"],
        "T5_gas_out (K)": df_core["T5_gas_out (K)"],
        "T6 (K)": df_core["T6 (K)"],
        "T7 (K)": df_core["T7 (K)"],
        "dT_hot_approach (K)": df_core["dT_hot_approach (K)"],
        "dT_cold_approach (K)": df_core["dT_cold_approach (K)"],
        "Temp_GC_OK": df_core["ok_gc_temp"],
        "Temp_usuario_OK": df_core["ok_user_temp"],
        "DTmin_HX_C": df_core["DTmin_HX_C"],
        "T7_obj (K)": df_core["T7_obj (K)"],
        "T7_max (K)": df_core["T7_max (K)"],
        "T7_gap (K)": df_core["T7_gap (K)"],
        "Semaforo_HX": df_core["Semaforo_HX"],

        "Q_gc (kW)": Q_gc,
        "Q_vap (kW)": Q_vap,
        "m_dot_vap (kg/s)": m_dot_vap,
        "Q_user (kW)": Q_user,

        # Global
        "eta_cog_global (%)": eta_cog,
        "eta_HX_utilizacion (-)": df_core["eta_HX_utilizacion (-)"],

        # Estado
        "status": df_core["status"],
        "Potencia etiqueta (kW)": P_elec,
    })

    df_cog = pd.DataFrame({
        "Municipio": df_core["Municipio"],
        "Turbina":   df_core["Turbina"],
        "P7 (bar)":  P7_BAR,      # si parametrizas por fila, agrega al core y úsalo
        "T7_user (°C)": T7_USER_C,
        "P8 (bar)":  P8_BAR,
        "Delta_subcool (°C)": DELTA_SUBCOOL_C,
        "DTmin_HX_C": df_core["DTmin_HX_C"],
        "T_gas_out_min (°C)": T_GAS_OUT_MINK - 273.15,
    })

    # === Validación y hojas extra ===
    PCI_CONST = PCI; MIN_T5_K = T_GAS_OUT_MINK
    def _rel_err(calc, ref):
        calc = np.asarray(calc, dtype=float); ref = np.asarray(ref, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.where(np.abs(ref) > 1e-12, ref, np.nan)
            return np.abs((calc - ref) / denom)

    # Re-cálculos para verificar con 'Estados'
    h_pivot = df_est.pivot_table(index=["Municipio","Turbina"], columns="Estado",
                                 values="h (kJ/kg)", aggfunc="first").rename(
        columns={4:"h4",5:"h5",6:"h6",7:"h7",8:"h8"}
    )
    T_pivot = df_est.pivot_table(index=["Municipio","Turbina"], columns="Estado",
                                 values="T (K)", aggfunc="first").rename(
        columns={4:"T4_state",5:"T5_state",6:"T6_state",7:"T7_state"}
    )

    dfv = (df_calc
           .merge(h_pivot, left_on=["Municipio","Turbina"], right_index=True, how="left")
           .merge(T_pivot,   left_on=["Municipio","Turbina"], right_index=True, how="left"))

    dfv["Q_gc_calc"]       = dfv["m_dot_gas (kg/s)"] * (dfv["h4"] - dfv["h5"])
    dfv["Q_vap_calc"]      = dfv["eta_HX_utilizacion (-)"] * dfv["Q_gc (kW)"]
    dfv["m_dot_vap_calc"]  = dfv["Q_vap (kW)"] / (dfv["h7"] - dfv["h6"])
    dfv["Q_user_calc"]     = dfv["m_dot_vap (kg/s)"] * (dfv["h7"] - dfv["h8"])
    dfv["eta_cog_calc"]    = (dfv["P_elec (kW)"] + dfv["Q_user (kW)"]) / (dfv["m_dot_fuel (kg/s)"]*PCI_CONST) * 100

    dfv["err_Q_gc_rel"]      = _rel_err(dfv["Q_gc_calc"], dfv["Q_gc (kW)"])
    dfv["err_Q_vap_rel"]     = _rel_err(dfv["Q_vap_calc"], dfv["Q_vap (kW)"])
    dfv["err_m_dot_vap_rel"] = _rel_err(dfv["m_dot_vap_calc"], dfv["m_dot_vap (kg/s)"])
    dfv["err_Q_user_rel"]    = _rel_err(dfv["Q_user_calc"], dfv["Q_user (kW)"])
    dfv["err_eta_cog_rel"]   = _rel_err(dfv["eta_cog_calc"], dfv["eta_cog_global (%)"])

    # Flags (incluye viabilidad básica y límites térmicos)
    dfv["T5_violation"]   = dfv["T5_gas_out (K)"] < MIN_T5_K
    dfv["inviable_auto"]  = (dfv["W_net (kJ/kg)"]<=0) | (dfv["m_dot_gas (kg/s)"]<=0) | (dfv["Q_vap (kW)"]<0)
    dfv["Temp_GC_OK"]     = dfv["Temp_GC_OK"].astype(bool)
    dfv["Temp_usuario_OK"]= dfv["Temp_usuario_OK"].astype(bool)

    # Resumen
#    resumen = pd.DataFrame({
#        "total_filas":[len(dfv)],
#        "violaciones_T5":[int(dfv["T5_violation"].sum())],
#        "inviables_por_status":[int((dfv["status"].astype(str).str.startswith("INV")).sum())],
#        "inviables_auto":[int(dfv["inviable_auto"].sum())],
#        "no_valid_Temp_GC":[int((~dfv["Temp_GC_OK"]).sum())],
#        "no_valid_Temp_usuario":[int((~dfv["Temp_usuario_OK"]).sum())],
#        "casos_limitados_usuario":[int((dfv["T7_gap (K)"] > 0).sum())],
#        "casos_ok_usuario":[int((dfv["T7_gap (K)"] <= 0).sum())],
#        "semaforo_OK":[int((dfv["Semaforo_HX"]=="OK").sum())],
#        "semaforo_Limite_usuario":[int((dfv["Semaforo_HX"]=="Límite usuario").sum())],
#        "semaforo_Limite_GC":[int((dfv["Semaforo_HX"]=="Límite GC").sum())],
#        "semaforo_Ambos":[int((dfv["Semaforo_HX"]=="Ambos").sum())],
#        "max_error_Qgc":[np.nanmax(dfv["err_Q_gc_rel"])],
#        "max_error_Qvap":[np.nanmax(dfv["err_Q_vap_rel"])],
#        "max_error_mdotvap":[np.nanmax(dfv["err_m_dot_vap_rel"])],
#        "max_error_Quser":[np.nanmax(dfv["err_Q_user_rel"])],
#        "max_error_eta_cog":[np.nanmax(dfv["err_eta_cog_rel"])],
#    })

    # Validaciones (con márgenes y banderas y alcanzabilidad)
    valid_cols = [
        "Municipio","Turbina","status",
        "DTmin_HX_C",
        "T4 (K)","T7_obj (K)","T7_max (K)","T7 (K)","dT_hot_approach (K)","Temp_usuario_OK","T7_gap (K)","Semaforo_HX",
        "T6 (K)","T5_gas_out (K)","dT_cold_approach (K)","Temp_GC_OK","T5_violation",
        "Q_gc (kW)","Q_vap (kW)","m_dot_vap (kg/s)","Q_user (kW)",
        "err_Q_gc_rel","err_Q_vap_rel","err_m_dot_vap_rel","err_Q_user_rel","err_eta_cog_rel"
    ]
    validaciones = dfv[valid_cols].copy()

    # Inviables (por status o por chequeo auto)
    inviables = dfv[(dfv["status"].astype(str).str.startswith("INV")) | (dfv["inviable_auto"])][[
        "Municipio","Turbina","status","W_net (kJ/kg)","m_dot_gas (kg/s)",
        "Q_vap (kW)","Q_user (kW)","eta_cog_global (%)",
        "Temp_GC_OK","Temp_usuario_OK","Semaforo_HX"
    ]].sort_values(["Turbina","Municipio"]).reset_index(drop=True)

    # Top/Bottom por eficiencia (válidos)
    mask_valid = ~(dfv["status"].astype(str).str.startswith("INV")) & ~(dfv["inviable_auto"]) & \
                 dfv["Temp_GC_OK"] & dfv["Temp_usuario_OK"] & pd.notna(dfv["eta_cog_global (%)"])
    top10 = dfv[mask_valid].sort_values("eta_cog_global (%)", ascending=False).head(10)[[
        "Municipio","Turbina","eta_cog_global (%)","P_elec (kW)","Q_user (kW)","m_dot_gas (kg/s)"
    ]].reset_index(drop=True)
    bottom10 = dfv[mask_valid].sort_values("eta_cog_global (%)", ascending=True).head(10)[[
        "Municipio","Turbina","eta_cog_global (%)","P_elec (kW)","Q_user (kW)","m_dot_gas (kg/s)"
    ]].reset_index(drop=True)

    # --- Guardado ---
    base, idx = "Resultados_Ciclo_Brayton", 1
    fname = f"{base}.xlsx"
    while Path(fname).exists():
        fname = f"{base}_{idx:02d}.xlsx"; idx += 1

    with pd.ExcelWriter(fname) as w:
        df_est.to_excel(w, sheet_name="Estados", index=False)
        df_calc.to_excel(w, sheet_name="Calculos", index=False)
        df_cog.to_excel(w,  sheet_name="Cogeneracion", index=False)
#       resumen.to_excel(w, sheet_name="resumen", index=False)
        validaciones.to_excel(w, sheet_name="validaciones", index=False)
        inviables.to_excel(w, sheet_name="inviables", index=False)
        top10.to_excel(w, sheet_name="top10", index=False)
        bottom10.to_excel(w, sheet_name="bottom10", index=False)

    print(f"Simulación completa. Resultados guardados en '{fname}'")

if __name__ == "__main__":
    main() 

    
