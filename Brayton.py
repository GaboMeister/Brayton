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
U_GLOBAL     = 0.3     # kW/(m²·K)
A_GLOBAL     = 8.0     # m²
M_DOT_WATER  = 2.0     # kg/s (agua)
CP_WATER     = 4.18    # kJ/(kg·K)
T_HOT_OUT    = 373.15  # K (100 °C gases a chimenea)
T_C_IN       = 293.15  # K (20 °C agua)
T_C_OUT      = 323.15  # K (50 °C agua salida)

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
    P_ele: float,                 # kW (potencia de etiqueta)
    *,
    eta_gen: float     = 0.95,
    eta_caldera: float = 0.85,
    PCI: float         = 50_000.0,  # kJ/kg
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Devuelve (DataFrame estados, dict cálculos, dict cogeneración)."""

    # --- Estados termodinámicos ---
    P1 = P1_kPa * 1e3
    P2 = P3 = P1 * r_p
    P4 = P1

    h1 = PropsSI("H", "T", T1, "P", P1, FLUIDO) / 1000
    s1 = PropsSI("S", "T", T1, "P", P1, FLUIDO) / 1000

    s1_J = PropsSI("S", "T", T1, "P", P1, FLUIDO)
    T2s  = PropsSI("T", "S", s1_J, "P", P2, FLUIDO)
    T2   = T1 + (T2s - T1) / eta_c
    h2   = PropsSI("H", "T", T2, "P", P2, FLUIDO) / 1000
    s2   = PropsSI("S", "T", T2, "P", P2, FLUIDO) / 1000

    h3 = PropsSI("H", "T", T3, "P", P3, FLUIDO) / 1000
    s3 = PropsSI("S", "T", T3, "P", P3, FLUIDO) / 1000

    s3_J = PropsSI("S", "T", T3, "P", P3, FLUIDO)
    T4s  = PropsSI("T", "S", s3_J, "P", P4, FLUIDO)
    T4   = T3 - eta_t * (T3 - T4s)
    h4   = PropsSI("H", "T", T4, "P", P4, FLUIDO) / 1000
    s4   = PropsSI("S", "T", T4, "P", P4, FLUIDO) / 1000

    # --- Balances del ciclo ---
    Q_in, Q_out = h3 - h2, h4 - h1
    W_comp, W_turb = h2 - h1, h3 - h4
    W_net = W_turb - W_comp
    W_ele = W_net * eta_gen

    # Flujo de gas para cumplir P_ele exactamente
    m_dot_gas = P_ele / W_ele if W_ele else 0.0
    Q_sum     = Q_in / eta_caldera if eta_caldera else 0.0

    P_net      = m_dot_gas * W_net
    P_elec     = eta_gen * P_net
    m_dot_fuel = (m_dot_gas * Q_sum) / PCI if PCI else 0.0

    # Indicadores de combustible
    SFC        = (m_dot_fuel / P_elec * 3600) if P_elec else None
    heat_rate  = (m_dot_fuel * PCI) / P_elec     if P_elec else None
    eta_global = (P_elec / (m_dot_fuel * PCI) * 100) if (m_dot_fuel and PCI) else None

    # --- Cogeneración ε-NTU ---
    cp_gas = PropsSI("Cpmass", "T", T4, "P", P4, FLUIDO) / 1000  # kJ/kg·K
    C_hot  = m_dot_gas * cp_gas
    C_cold = M_DOT_WATER * CP_WATER
    C_min, C_max = min(C_hot, C_cold), max(C_hot, C_cold)
    NTU    = (U_GLOBAL * A_GLOBAL) / C_min if C_min else 0.0
    Cr     = C_min / C_max          if C_max else 0.0
    eps    = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr))) if C_min else 0.0

    Q_rec      = eps * C_min * (T4 - T_C_IN)
    q_rec_spec = Q_rec / m_dot_gas if m_dot_gas else None
    eta_cog    = (W_ele + q_rec_spec) / Q_sum * 100 if Q_sum else None

    # --- DataFrames y diccionarios ---
    estados = pd.DataFrame([
        {"Estado": 1, "P (kPa)": P1_kPa,         "T (K)": T1, "h (kJ/kg)": round(h1, 3), "s (kJ/kg·K)": round(s1, 5)},
        {"Estado": 2, "P (kPa)": P1_kPa * r_p,    "T (K)": T2, "h (kJ/kg)": round(h2, 3), "s": round(s2, 5)},
        {"Estado": 3, "P (kPa)": P1_kPa * r_p,    "T (K)": T3, "h (kJ/kg)": round(h3, 3), "s": round(s3, 5)},
        {"Estado": 4, "P (kPa)": P1_kPa,         "T (K)": T4, "h (kJ/kg)": round(h4, 3), "s": round(s4, 5)},
    ])

    calculos = {
        "Q_in (kJ/kg)": Q_in, "Q_out (kJ/kg)": Q_out,
        "W_comp (kJ/kg)": W_comp, "W_turb (kJ/kg)": W_turb,
        "W_net (kJ/kg)": W_net, "W_ele (kJ/kg)": W_ele,
        "Q_sum (kJ/kg)": Q_sum, "eta_ciclo (%)": (W_ele / Q_sum * 100) if Q_sum else None,
        "m_dot_gas (kg/s)": m_dot_gas, "P_net (kW)": P_net, "P_elec (kW)": P_elec,
        "m_dot_fuel (kg/s)": m_dot_fuel, "SFC (kg/kWh)": SFC,
        "Heat rate (kJ/kWh)": heat_rate, "eta_global (%)": eta_global,
    }

    cog = {
        "T_hot_in (K)": T4, "T_hot_out (K)": T_HOT_OUT,
        "T_c_in (K)": T_C_IN, "T_c_out (K)": T_C_OUT,
        "DeltaT1 (K)": T4 - T_C_OUT, "DeltaT2 (K)": T_HOT_OUT - T_C_IN,
        "U (kW/m2K)": U_GLOBAL, "A (m2)": A_GLOBAL,
        "m_dot_water (kg/s)": M_DOT_WATER,
        "C_min (kW/K)": C_min, "C_max (kW/K)": C_max,
        "Cr": Cr, "NTU": NTU, "epsilon": eps,
        "Q_rec (kW)": Q_rec, "q_rec_spec (kJ/kg)": q_rec_spec,
        "eta_cog (%)": eta_cog,
    }

    return estados, calculos, cog


# -----------------------------------------------------------------------------
# Main vectorizado
# -----------------------------------------------------------------------------
def main():
    # Carga de datos
    df_mun = pd.read_excel("Municipios_D.xlsx")
    df_mun["T1 (K)"] = df_mun["Temperatura (°C)"] + 273.15
    df_mun["P1 (kPa)"] = df_mun["Presión (bares)"] * 100

    df_tur = pd.read_csv("Base_de_datos_turbinas_de_gas.csv")

    # Producto cartesiano municipios × turbinas
    df_mun["_key"] = 1
    df_tur["_key"] = 1
    df_cross = pd.merge(df_mun, df_tur, on="_key").drop(columns="_key")

    # Aplicar simulación para cálculos y cogeneración
    def aplicar_fila(row):
        _, calc, cog = simular_ciclo(
            T1     = row["T1 (K)"],
            P1_kPa = row["P1 (kPa)"],
            r_p    = row["r_p"],
            T3     = row["T3 (C)"],
            eta_c  = row["eta_c"],
            eta_t  = row["eta_t"],
            P_ele  = row["Potencia (kW)"],
        )
        # Etiquetas
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

    # Separar resultados
    cog_cols = [
        "T_hot_in (K)", "T_hot_out (K)", "T_c_in (K)", "T_c_out (K)",
        "DeltaT1 (K)", "DeltaT2 (K)",
        "U (kW/m2K)", "A (m2)", "m_dot_water (kg/s)",
        "C_min (kW/K)", "C_max (kW/K)", "Cr", "NTU", "epsilon",
        "Q_rec (kW)", "q_rec_spec (kJ/kg)", "eta_cog (%)",
        "Municipio", "Turbina"
    ]
    df_cog  = df_all[cog_cols]
    df_calc = df_all.drop(columns=cog_cols)

    # Estados: usar bucle simple para concatenarlos (4 filas por simulación)
    estados_list = []
    for _, row in df_cross.iterrows():
        est, _, _ = simular_ciclo(
            T1     = row["T1 (K)"],
            P1_kPa = row["P1 (kPa)"],
            r_p    = row["r_p"],
            T3     = row["T3 (C)"],
            eta_c  = row["eta_c"],
            eta_t  = row["eta_t"],
            P_ele  = row["Potencia (kW)"],
        )
        est.insert(0, "Municipio", row["Municipio"])
        est.insert(1, "Turbina",   row["Turbina"])
        estados_list.append(est)
    df_est = pd.concat(estados_list, ignore_index=True)

    # Exportar a Excel
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
