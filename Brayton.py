# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:18:39 2025

@author: Gabo San
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

# -----------------------------------------------------------------------------
# Parámetros globales
# -----------------------------------------------------------------------------
FLUIDO = "Air"

# Cogeneración – parámetros de diseño
U_GLOBAL     = 0.9     # kW/(m²·K)
A_GLOBAL     = 25.0    # m²
M_DOT_WATER  = 25.0    # kg/s (agua)
CP_WATER     = 4.18    # kJ/(kg·K)
T_C_IN       = 293.15  # K (20 °C agua)

# ------------------------------------------------------------------
# Helper termodinámico: entalpía (kJ/kg) y entropía (kJ/kg·K)
# ------------------------------------------------------------------
def estado_PT(T: float, P: float) -> Tuple[float, float]:
    h = PropsSI("H", "T", T, "P", P, FLUIDO) / 1000
    s = PropsSI("S", "T", T, "P", P, FLUIDO) / 1000
    return h, s

# ------------------------------------------------------------------
# Helper para reordenar columnas
# ------------------------------------------------------------------
def reorder(df: pd.DataFrame, front_cols: list[str]) -> pd.DataFrame:
    cols = df.columns.tolist()
    for c in reversed(front_cols):
        if c in cols:
            cols.remove(c)
            cols.insert(0, c)
    return df[cols]

# -----------------------------------------------------------------------------
# Función de simulación del ciclo Brayton + cogeneración
# -----------------------------------------------------------------------------
def simular_ciclo(
    T1: float,
    P1_kPa: float,
    r_p: float,
    T3: float,
    eta_c: float,
    eta_t: float,
    V_dot_design: float,    # m³/s de aire de diseño
    *,
    eta_gen: float     = 0.95,
    eta_caldera: float = 0.85,
    PCI: float         = 50_000.0,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Devuelve (estados, cálculos, cogeneración) usando V_dot_design + densidad local."""

    # --- Temperatura entrada turbina a Kelvin ---
    T3_K = T3 + 273.15

    # --- Presiones absolutas (Pa) ---
    P1 = P1_kPa * 1e3
    P2 = P3 = P1 * r_p
    P4 = P1

    # --- Estados termodinámicos con helper ---
    h1, s1 = estado_PT(T1, P1)
    s1_J   = PropsSI("S", "T", T1, "P", P1, FLUIDO)
    T2s    = PropsSI("T", "S", s1_J, "P", P2, FLUIDO)
    T2     = T1 + (T2s - T1) / eta_c
    h2, s2 = estado_PT(T2, P2)

    h3, s3 = estado_PT(T3_K, P3)
    s3_J   = PropsSI("S", "T", T3_K, "P", P3, FLUIDO)
    T4s    = PropsSI("T", "S", s3_J, "P", P4, FLUIDO)
    T4     = T3_K - eta_t * (T3_K - T4s)
    h4, s4 = estado_PT(T4, P4)

    # --- Balances del ciclo (por kg de gas) ---
    Q_in     = h3 - h2
    W_comp   = h2 - h1
    W_turb   = h3 - h4
    W_net    = W_turb - W_comp
    W_ele    = W_net * eta_gen

    # --- Caudal másico real según altitud ---
    rho_amb   = PropsSI("D", "T", T1, "P", P1, FLUIDO)
    m_dot_gas = V_dot_design * rho_amb

    # --- Fuel por kg de gas y potencias ---
    Q_fuel_spec = Q_in / eta_caldera if eta_caldera else 0.0
    P_net       = m_dot_gas * W_net
    P_elec      = m_dot_gas * W_ele
    m_dot_fuel  = (m_dot_gas * Q_fuel_spec) / PCI if PCI else 0.0

    # --- Indicadores de combustible ---
    SFC        = (m_dot_fuel / P_elec * 3600) if P_elec else None
    heat_rate  = SFC * PCI if SFC is not None else None
    eta_global = (P_elec / (m_dot_fuel * PCI) * 100) if (m_dot_fuel and PCI) else None

    # --- Cogeneración ε-NTU y temperaturas dinámicas ---
    cp_gas    = PropsSI("Cpmass", "T", T4, "P", P4, FLUIDO) / 1000
    C_hot     = m_dot_gas * cp_gas
    C_cold    = M_DOT_WATER * CP_WATER
    C_min     = min(C_hot, C_cold)
    C_max     = max(C_hot, C_cold)
    NTU       = (U_GLOBAL * A_GLOBAL) / C_min if C_min else 0.0
    Cr        = C_min / C_max         if C_max else 0.0
    eps       = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr))) if C_min else 0.0
    Q_rec     = eps * C_min * (T4 - T_C_IN)

    T_cold_out = T_C_IN + Q_rec / C_cold
    T_hot_out  = T4      - Q_rec / C_hot

    q_rec_spec = Q_rec / m_dot_gas if m_dot_gas else None
    eta_cog    = (W_ele + q_rec_spec) / Q_fuel_spec * 100 if Q_fuel_spec else None
    
    # --- Estados del intercambiador: 5 gas salida, 6 agua entrada, 7 agua salida ---
    #  P5 = P1_kPa,       T5 = T_hot_out
    #  P6 = 10 bares,     T6 = T_C_IN
    #  P7 = 10 bares,     T7 = T_cold_out
    
    P6_kPa = 10 * 100.0      # 10 bar = 1000 kPa
    P6_Pa  = P6_kPa * 1e3    # 1 000 000 Pa
    
    # Estado 5 (gases a la salida del intercambiador)
    h5, s5 = estado_PT(T_hot_out, P1)  
    
    # Estado 6 (agua a la entrada, 10 bar)
    h6, s6 = estado_PT(T_C_IN,  P6_Pa)
    
    # Estado 7 (agua a la salida, 10 bar)
    h7, s7 = estado_PT(T_cold_out, P6_Pa)

    estados = pd.DataFrame([
        {"Estado": 1, "P (kPa)": P1_kPa,      "T (K)": T1,   "h (kJ/kg)": round(h1,3), "s (kJ/kg·K)": round(s1,5)},
        {"Estado": 2, "P (kPa)": P1_kPa * r_p, "T (K)": T2,   "h (kJ/kg)": round(h2,3), "s (kJ/kg·K)": round(s2,5)},
        {"Estado": 3, "P (kPa)": P1_kPa * r_p, "T (K)": T3_K, "h (kJ/kg)": round(h3,3), "s (kJ/kg·K)": round(s3,5)},
        {"Estado": 4, "P (kPa)": P1_kPa,      "T (K)": T4,   "h (kJ/kg)": round(h4,3), "s (kJ/kg·K)": round(s4,5)},
        {"Estado": 5, "P (kPa)": P1_kPa,  "T (K)": round(T_hot_out,2), "h (kJ/kg)": round(h5,3), "s (kJ/kg·K)": round(s5,5)},
        {"Estado": 6, "P (kPa)": P6_kPa,  "T (K)": T_C_IN,           "h (kJ/kg)": round(h6,3), "s (kJ/kg·K)": round(s6,5)},
        {"Estado": 7, "P (kPa)": P6_kPa,  "T (K)": round(T_cold_out,2),"h (kJ/kg)": round(h7,3), "s (kJ/kg·K)": round(s7,5)},
    ])

    calculos = {
        "Q_in (kJ/kg)":      Q_in,
        "W_net (kJ/kg)":     W_net,
        "W_ele (kJ/kg)":     W_ele,
        "Q_fuel_spec (kJ/kg)": Q_fuel_spec,
        "eta_ciclo (%)":     (W_ele / Q_fuel_spec * 100) if Q_fuel_spec else None,
        "m_dot_gas (kg/s)":  m_dot_gas,
        "P_net (kW)":        P_net,
        "P_elec (kW)":       P_elec,
        "m_dot_fuel (kg/s)": m_dot_fuel,
        "SFC (kg/kWh)":      SFC,
        "Heat rate (kJ/kWh)": heat_rate,
        "eta_global (%)":    eta_global,
    }

    cog = {
        "T_cold_out (°C)":  round(T_cold_out - 273.15, 2),
        "T_hot_out (°C)":   round(T_hot_out  - 273.15, 2),
        "Q_rec (kW)":       round(Q_rec, 2),
        "q_rec_spec (kJ/kg)": q_rec_spec,
        "eta_cog (%)":      eta_cog,
    }


    return estados, calculos, cog

# -----------------------------------------------------------------------------  
# Main vectorizado  
# -----------------------------------------------------------------------------  
def main():  
    df_mun = pd.read_excel("Municipios_D.xlsx")  
    df_mun["T1 (K)"]   = df_mun["Temperatura (°C)"] + 273.15  
    df_mun["P1 (kPa)"] = df_mun["Presión (bares)"] * 100  

    df_tur = pd.read_csv("Base_de_datos_turbinas_de_gas.csv")  
    T1_ISO, P1_ISO = 288.15, 101.325  
    rho_iso = PropsSI("D", "T", T1_ISO, "P", P1_ISO*1e3, FLUIDO)  
    df_tur["V_dot_design"] = df_tur["m_aire (kg/s)"] / rho_iso  

    df_cross = df_mun.merge(df_tur, how="cross")  

    def aplicar_fila(row):  
        _, calc, cog = simular_ciclo(  
            T1            = row["T1 (K)"],  
            P1_kPa        = row["P1 (kPa)"],  
            r_p           = row["r_p"],  
            T3            = row["T3 (C)"],  
            eta_c         = row["eta_c"],  
            eta_t         = row["eta_t"],  
            V_dot_design  = row["V_dot_design"],  
        )  
        calc.update({  
            "Municipio": row["Municipio"],  
            "Turbina":   row["Turbina"],  
            "Potencia etiqueta (kW)": row["Potencia (kW)"],  
        })  
        cog.update({  
            "Municipio": row["Municipio"],  
            "Turbina":   row["Turbina"],  
        })  
        return pd.Series({**calc, **cog})  

    df_all = df_cross.apply(aplicar_fila, axis=1)  

    metadata = ["Municipio", "Turbina"]  
    cog_only = ["T_cold_out (°C)", "T_hot_out (°C)", "Q_rec (kW)", "q_rec_spec (kJ/kg)", "eta_cog (%)"]  

    df_cog  = df_all[metadata + cog_only]  
    df_calc = df_all.drop(columns=cog_only)  

    df_calc = reorder(df_calc, metadata)  
    df_cog  = reorder(df_cog,  metadata)  

    estados_list = []  
    for _, row in df_cross.iterrows():  
        est, _, _ = simular_ciclo(  
            T1           = row["T1 (K)"],  
            P1_kPa       = row["P1 (kPa)"],  
            r_p          = row["r_p"],  
            T3           = row["T3 (C)"],  
            eta_c        = row["eta_c"],  
            eta_t        = row["eta_t"],  
            V_dot_design = row["V_dot_design"],  
        )  
        est.insert(0, "Municipio", row["Municipio"])  
        est.insert(1, "Turbina",   row["Turbina"])  
        estados_list.append(est)  
    df_est = pd.concat(estados_list, ignore_index=True)  

    base, idx = "Resultados_Ciclo_Brayton", 1  
    fname = f"{base}.xlsx"  
    while os.path.exists(fname):  
        fname = f"{base}_{idx:02d}.xlsx"  
        idx += 1  

    with pd.ExcelWriter(fname) as writer:  
        df_est.to_excel(writer, sheet_name="Estados", index=False)  
        df_calc.to_excel(writer, sheet_name="Calculos", index=False)  
        df_cog.to_excel(writer, sheet_name="Cogeneracion", index=False)  

    print(f"Simulación completa. Resultados guardados en '{fname}'")  

if __name__ == "__main__":  
    main()  
    