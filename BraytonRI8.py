# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:18:39 2025

@author: Gabo San
"""

import os
from io import BytesIO
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# --- CoolProp con cache ---
from CoolProp.CoolProp import PropsSI as _CP_PropsSI
from functools import lru_cache

def _q(x, nd=6):
    """Cuantiza un float para estabilizar la clave de caché sin afectar resultados."""
    return float(round(float(x), nd))

@lru_cache(maxsize=500_000)
def _PropsSI_cached(out, n1, v1, n2, v2, fluid):
    return _CP_PropsSI(out, n1, float(v1), n2, float(v2), fluid)

def PropsSI(out, n1, v1, n2, v2, fluid):
    """Drop-in replacement de PropsSI que pasa por caché."""
    return _PropsSI_cached(out, n1, _q(v1), n2, _q(v2), fluid)

# Helpers convenientes (en J/kg, K, Pa)
def H_TP(T, P, fluid):   return PropsSI("H", "T", T, "P", P, fluid)
def S_TP(T, P, fluid):   return PropsSI("S", "T", T, "P", P, fluid)
def T_HP(H, P, fluid):   return PropsSI("T", "H", H, "P", P, fluid)
def Cp_TP(T, P, fluid):  return PropsSI("Cpmass", "T", T, "P", P, fluid)
def D_TP(T, P, fluid):   return PropsSI("D", "T", T, "P", P, fluid)

# =============================================================================
# Parámetros globales
# =============================================================================
FLUIDO_GAS = "Air"
FLUIDO_W   = "Water"

# Brayton
ETA_GEN      = 0.95
ETA_CALDERA  = 0.85
PCI          = 50_000.0  # kJ/kg

# HX (UA global)
U_GLOBAL   = 0.30     # kW/(m2·K)
A_GLOBAL   = 8.0      # m2
m_dot_w    = 2.0      # kg/s  (caudal de agua de diseño, visible/ajustable)

# Lazo agua (defaults; pueden venir en CSV si quieres)
P7_BAR         = 10.0
P8_BAR         = 2.0
DELTA_SUBCOOL_C= 15.0   # T8 = Tsat(P8) - DELTA_SUBCOOL_C
T7_USER_C      = 180.0  # objetivo setpoint

# Restricción de corrosión en salida gases
T_GAS_OUT_MINC = 120.0
T_GAS_OUT_MINK = T_GAS_OUT_MINC + 273.15

# =============================================================================
# Utilidades agua: estados 6-7-8
# =============================================================================
def agua_estado7(P7: float, T7_target: float):
    """(T7, h7, s7, Tsat7) a P7; maneja sub/sat/sobrecalentado con umbral."""
    Tsat = PropsSI("T", "P", P7, "Q", 0, FLUIDO_W)
    epsC = 0.5  # margen para "casi saturado"
    if abs(T7_target - Tsat) <= epsC:  # vapor saturado
        h7 = PropsSI("H","P",P7,"Q",1,FLUIDO_W)/1000.0
        s7 = PropsSI("S","P",P7,"Q",1,FLUIDO_W)/1000.0
        T7 = Tsat
    elif T7_target < Tsat:             # líquido subenfriado
        T7 = T7_target
        h7 = PropsSI("H","T",T7,"P",P7,FLUIDO_W)/1000.0
        s7 = PropsSI("S","T",T7,"P",P7,FLUIDO_W)/1000.0
    else:                               # vapor sobrecalentado
        T7 = T7_target
        h7 = PropsSI("H","T",T7,"P",P7,FLUIDO_W)/1000.0
        s7 = PropsSI("S","T",T7,"P",P7,FLUIDO_W)/1000.0
    return T7, h7, s7, Tsat

def agua_estado8(P8: float, delta_subcool_C: float):
    """Estado 8 (entregado por usuario): T8 = Tsat(P8) - delta_subcool_C."""
    Tsat8 = PropsSI("T","P",P8,"Q",0,FLUIDO_W)
    T8    = Tsat8 - delta_subcool_C
    h8    = PropsSI("H","T",T8,"P",P8,FLUIDO_W)/1000.0
    s8    = PropsSI("S","T",T8,"P",P8,FLUIDO_W)/1000.0
    rho8  = D_TP(T8, P8, FLUIDO_W)   # kg/m3
    v8    = 1.0/rho8                 # m3/kg
    return T8, h8, s8, v8

def agua_bomba_a6(P6: float, P8: float, h8: float, v8: float):
    """Bomba (8->6) incompresible: h6 = h8 + v*ΔP."""
    dh_pump = v8*(P6-P8)/1000.0  # kJ/kg
    h6 = h8 + dh_pump
    T6 = T_HP(h6*1000.0, P6, FLUIDO_W)
    s6 = PropsSI("S","H",h6*1000.0,"P",P6,FLUIDO_W)/1000.0
    return T6, h6, s6, dh_pump

# =============================================================================
# Brayton puro (con reuso de estados)
# =============================================================================
def brayton_estados(T1: float, P1_kPa: float, r_p: float, T3: float,
                    eta_c: float, eta_t: float):
    P1 = P1_kPa*1e3
    P2 = P3 = P1*r_p
    P4 = P1

    # Estado 1
    h1J = H_TP(T1, P1, FLUIDO_GAS); h1 = h1J/1000.0
    s1J = S_TP(T1, P1, FLUIDO_GAS); s1 = s1J/1000.0

    # Compresión 1->2 (isentropía con s1J)
    T2s = PropsSI("T", "S", s1J, "P", P2, FLUIDO_GAS)
    T2  = T1 + (T2s - T1)/eta_c
    h2J = H_TP(T2, P2, FLUIDO_GAS); h2 = h2J/1000.0
    s2J = S_TP(T2, P2, FLUIDO_GAS); s2 = s2J/1000.0

    # Cámara de combustión 2->3 (fijado por T3,P3)
    h3J = H_TP(T3, P3, FLUIDO_GAS); h3 = h3J/1000.0
    s3J = S_TP(T3, P3, FLUIDO_GAS); s3 = s3J/1000.0

    # Expansión 3->4 (isentropía con s3J)
    T4s = PropsSI("T", "S", s3J, "P", P4, FLUIDO_GAS)
    T4  = T3 - eta_t*(T3 - T4s)
    h4J = H_TP(T4, P4, FLUIDO_GAS); h4 = h4J/1000.0
    s4J = S_TP(T4, P4, FLUIDO_GAS); s4 = s4J/1000.0

    # Balances específicos (kJ/kg)
    Q_in   = h3 - h2
    Q_out  = h4 - h1
    W_comp = h2 - h1
    W_turb = h3 - h4
    W_net  = W_turb - W_comp
    W_ele  = W_net*ETA_GEN
    Q_sum  = Q_in/ETA_CALDERA

    return (P1,P2,P3,P4), (T2,T4), (h1,h2,h3,h4), (s1,s2,s3,s4), (Q_in,Q_out,W_comp,W_turb,W_net,W_ele,Q_sum), T4

# =============================================================================
# HX ε–NTU con lazo agua 6–7–8 (aprox con cambio de fase)
# =============================================================================
def hx_ntu_lazo_agua(m_dot_gas: float, T4: float, P4_gas_Pa: float,
                     P7_bar: float, T7_user_C: float,
                     P8_bar: float, delta_subcool_C: float):
    """
    Devuelve dict con:
      - estados 8,6,7 (real y objetivo), caudales y Q_pre/plateau
      - T5/h5 del gas por cierre energético exacto
    """
    P7 = P7_bar*1e5
    P8 = P8_bar*1e5

    # Estados agua 8 y 6
    T8, h8, s8, v8 = agua_estado8(P8, delta_subcool_C)
    T6, h6, s6, dh_pump = agua_bomba_a6(P6=P7, P8=P8, h8=h8, v8=v8)

    # Estado 7 objetivo (según setpoint)
    T7_target = T7_user_C + 273.15
    T7_obj, h7_obj, s7_obj, Tsat7 = agua_estado7(P7, T7_target)

    # Demanda entálpica por kg de agua
    delta_h_67_req = max(h7_obj - h6, 0.0)  # kJ/kg

    # --- Lado gas: cp en T4 (aprox) y capacidad caliente ---
    cp_gas = Cp_TP(T4, P4_gas_Pa, FLUIDO_GAS)/1000.0  # kJ/kgK
    C_hot  = max(m_dot_gas*cp_gas, 1e-9)       # kW/K

    # Plateau (ebullición) y precalentamiento
    Tplateau = max(T6, Tsat7)
    cp_liq   = Cp_TP(max(T6, Tsat7 - 1.0), P7, FLUIDO_W)/1000.0
    q_pre    = max(0.0, cp_liq*(Tplateau - T6))    # kJ/kg
    Q_pre    = m_dot_w * q_pre                     # kW

    NTU  = (U_GLOBAL*A_GLOBAL)/C_hot
    eps  = 1.0 - np.exp(-NTU)
    Q_plateau_NTU = eps*C_hot*(T4 - Tplateau)      # kW (limite por lado gas + UA)

    Q_req_total = m_dot_w * delta_h_67_req         # kW
    Q_avail_NTU = Q_pre + max(Q_plateau_NTU, 0.0)

    Q_to_water  = min(Q_req_total, Q_avail_NTU)    # kW transferidos realmente

    # h7_real y T7_real
    q_spec_real = Q_to_water/m_dot_w if m_dot_w>0 else 0.0
    h7_real = h6 + q_spec_real
    T7_real = T_HP(h7_real*1000.0, P7, FLUIDO_W)
    s7_real = PropsSI("S","H",h7_real*1000.0,"P",P7,FLUIDO_W)/1000.0

    # Salida del gas: cerrar por energía (h5 primero), luego T5 desde (H,P)
    h4_local_kJkg = H_TP(T4, P4_gas_Pa, FLUIDO_GAS)/1000.0
    if m_dot_gas > 0:
        h5 = h4_local_kJkg - Q_to_water / m_dot_gas  # kJ/kg (balance exacto)
        T5 = T_HP(h5*1000.0, P4_gas_Pa, FLUIDO_GAS)
    else:
        h5 = h4_local_kJkg
        T5 = T4

    return {
        "P7":P7,"P8":P8,
        "T8":T8,"h8":h8,"s8":s8,
        "T6":T6,"h6":h6,"s6":s6,"dh_pump":dh_pump,
        "T7_obj":T7_obj,"h7_obj":h7_obj,"s7_obj":s7_obj,"Tsat7":Tsat7,
        "T7_real":T7_real,"h7_real":h7_real,"s7_real":s7_real,
        "m_dot_w":m_dot_w,
        "Q_pre (kW)":Q_pre,
        "Q_plateau_NTU (kW)":Q_plateau_NTU,
        "Q_req_total (kW)":Q_req_total,
        "Q_to_water (kW)":Q_to_water,
        "T5":T5,"h5":h5,
        "cp_gas":cp_gas,"NTU":NTU,"eps":eps,
        "h4_used": h4_local_kJkg
    }

# =============================================================================
# Simulación ciclo completo
# =============================================================================
def simular_ciclo(
    T1: float, P1_kPa: float, r_p: float, T3: float, eta_c: float, eta_t: float,
    P_ele: float,
    *,
    eta_gen: float = ETA_GEN, eta_caldera: float = ETA_CALDERA, PCI_: float = PCI,
    p7_bar: float = P7_BAR, p8_bar: float = P8_BAR, delta_subcool_c: float = DELTA_SUBCOOL_C,
    t7_user_c: float = T7_USER_C
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:

    # --- Brayton puro ---
    (P1,P2,P3,P4), (T2,T4), (h1,h2,h3,h4), (s1,s2,s3,s4), \
        (Q_in,Q_out,W_comp,W_turb,W_net,W_ele,Q_sum), T4 = brayton_estados(
            T1,P1_kPa,r_p,T3,eta_c,eta_t
        )

    # Flujo de gas para cumplir P_ele
    m_dot_gas = P_ele / W_ele if W_ele>0 else 0.0
    P_net     = m_dot_gas * W_net
    P_elec    = eta_gen * P_net
    m_dot_fuel= (m_dot_gas * Q_sum)/PCI_ if PCI_>0 else 0.0

    SFC       = (m_dot_fuel / P_elec * 3600) if P_elec>0 else np.nan
    heat_rate = (m_dot_fuel*PCI_) / P_elec  if P_elec>0 else np.nan
    eta_ciclo = (W_ele/Q_sum*100) if Q_sum else np.nan

    # --- HX + lazo agua 6-7-8 ---
    hx = hx_ntu_lazo_agua(m_dot_gas, T4, P4,
                          P7_bar=p7_bar, T7_user_C=t7_user_c,
                          P8_bar=p8_bar, delta_subcool_C=delta_subcool_c)

    T5 = hx["T5"]; h5 = hx["h5"]
    # Específicos lado gas y usuario (usa h4_used para cierre)
    Q_gc_spec = max(hx.get("h4_used", h4) - h5, 0.0)  # kJ/kg_gas
    Q_gc      = m_dot_gas * Q_gc_spec                 # kW
    Q_user    = hx["Q_to_water (kW)"]                 # kW
    eta_cog   = (P_elec + Q_user)/(m_dot_fuel*PCI_)*100 if (m_dot_fuel>0) else np.nan

    # --- Validaciones ---
    ok_gc_temp   = bool(T5 >= T_GAS_OUT_MINK)
    ok_user_temp = bool(hx["T7_real"] >= hx["T7_obj"] - 1e-6)
    if ok_gc_temp and ok_user_temp:
        semaf = "OK"
    elif (not ok_gc_temp) and ok_user_temp:
        semaf = "Límite GC"
    elif ok_gc_temp and (not ok_user_temp):
        semaf = "Límite usuario"
    else:
        semaf = "Ambos"

    # --- Estados DataFrame ---
    estados = pd.DataFrame([
        {"Estado":1, "P (kPa)":P1/1e3, "T (K)":T1, "h (kJ/kg)":h1, "s (kJ/kg·K)":s1},
        {"Estado":2, "P (kPa)":P2/1e3, "T (K)":T2, "h (kJ/kg)":h2, "s (kJ/kg·K)":s2},
        {"Estado":3, "P (kPa)":P3/1e3, "T (K)":T3, "h (kJ/kg)":h3, "s (kJ/kg·K)":s3},
        {"Estado":4, "P (kPa)":P4/1e3, "T (K)":T4, "h (kJ/kg)":h4, "s (kJ/kg·K)":s4},
        {"Estado":5, "P (kPa)":P4/1e3, "T (K)":T5, "h (kJ/kg)":h5, "s (kJ/kg·K)":
            PropsSI("S","T",T5,"P",P4,FLUIDO_GAS)/1000.0},
        {"Estado":6, "P (kPa)":p7_bar*100, "T (K)":hx["T6"], "h (kJ/kg)":hx["h6"], "s (kJ/kg·K)":hx["s6"]},
        {"Estado":7, "P (kPa)":p7_bar*100, "T (K)":hx["T7_real"], "h (kJ/kg)":hx["h7_real"], "s (kJ/kg·K)":hx["s7_real"]},
        {"Estado":8, "P (kPa)":p8_bar*100, "T (K)":hx["T8"], "h (kJ/kg)":hx["h8"], "s (kJ/kg·K)":hx["s8"]},
    ])

    # --- Cálculos (Brayton + usuario) ---
    calculos = {
        "Q_in (kJ/kg)":Q_in, "Q_out (kJ/kg)":Q_out,
        "W_comp (kJ/kg)":W_comp, "W_turb (kJ/kg)":W_turb,
        "W_net (kJ/kg)":W_net, "W_ele (kJ/kg)":W_ele,
        "Q_sum (kJ/kg)":Q_sum, "eta_ciclo_electrico (%)":eta_ciclo,
        "m_dot_gas (kg/s)":m_dot_gas, "P_net (kW)":P_net, "P_elec (kW)":P_elec,
        "m_dot_fuel (kg/s)":m_dot_fuel, "SFC_elec (kg/kWh)":SFC,
        "Heat rate_elec (kJ/kWh)":heat_rate,
        "T4 (K)":T4, "T5_gas_out (K)":T5,
        "Q_gc (kW)":Q_gc, "Q_user (kW)":Q_user,
        "eta_cog_global (%)":eta_cog,
    }

    # --- Cogeneración / HX y validaciones ---
    cog = {
        "U (kW/m2K)":U_GLOBAL, "A (m2)":A_GLOBAL,
        "cp_gas (kJ/kgK)":hx["cp_gas"], "NTU":hx["NTU"], "epsilon":hx["eps"],
        "m_dot_w (kg/s)":hx["m_dot_w"],
        "Q_pre (kW)":hx["Q_pre (kW)"],
        "Q_plateau_NTU (kW)":hx["Q_plateau_NTU (kW)"],
        "Q_req_total (kW)":hx["Q_req_total (kW)"],
        "Q_to_water (kW)":hx["Q_to_water (kW)"],
        "T7_obj (K)":hx["T7_obj"], "T7_real (K)":hx["T7_real"], "Tsat7 (K)":hx["Tsat7"],
        "Temp_GC_OK":ok_gc_temp, "Temp_usuario_OK":ok_user_temp,
        "Semaforo_HX":semaf
    }

    return estados, calculos, cog

def simular_ciclo_mdot(
    T1: float, P1_kPa: float, r_p: float, T3: float, eta_c: float, eta_t: float,
    m_dot_gas: float,
    *,
    eta_gen: float = ETA_GEN, eta_caldera: float = ETA_CALDERA, PCI_: float = PCI,
    p7_bar: float = P7_BAR, p8_bar: float = P8_BAR, delta_subcool_c: float = DELTA_SUBCOOL_C,
    t7_user_c: float = T7_USER_C
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Variante de simular_ciclo que usa caudal másico de gas como entrada.
    Devuelve los mismos tres objetos (estados, calculos, cog).
    """
    # --- Brayton por kg ---
    (P1,P2,P3,P4), (T2,T4), (h1,h2,h3,h4), (s1,s2,s3,s4), \
        (Q_in,Q_out,W_comp,W_turb,W_net,W_ele,Q_sum), T4 = brayton_estados(
            T1,P1_kPa,r_p,T3,eta_c,eta_t
        )

    # --- Escalado por caudal ---
    P_net     = m_dot_gas * W_net
    P_elec    = eta_gen * P_net
    m_dot_fuel= (m_dot_gas * Q_sum)/PCI_ if PCI_>0 else 0.0

    SFC       = (m_dot_fuel / P_elec * 3600) if P_elec>0 else np.nan
    heat_rate = (m_dot_fuel*PCI_) / P_elec  if P_elec>0 else np.nan
    eta_ciclo = (W_ele/Q_sum*100) if Q_sum else np.nan

    # --- HX + lazo agua 6-7-8 ---
    hx = hx_ntu_lazo_agua(m_dot_gas, T4, P4,
                          P7_bar=p7_bar, T7_user_C=t7_user_c,
                          P8_bar=p8_bar, delta_subcool_C=delta_subcool_c)

    T5 = hx["T5"]; h5 = hx["h5"]

    Q_gc_spec = max(hx.get("h4_used", h4) - h5, 0.0)
    Q_gc      = m_dot_gas * Q_gc_spec
    Q_user    = hx["Q_to_water (kW)"]
    eta_cog   = (P_elec + Q_user)/(m_dot_fuel*PCI_)*100 if (m_dot_fuel>0) else np.nan

    ok_gc_temp   = bool(T5 >= T_GAS_OUT_MINK)
    ok_user_temp = bool(hx["T7_real"] >= hx["T7_obj"] - 1e-6)
    if ok_gc_temp and ok_user_temp:            semaf = "OK"
    elif (not ok_gc_temp) and ok_user_temp:    semaf = "Límite GC"
    elif ok_gc_temp and (not ok_user_temp):    semaf = "Límite usuario"
    else:                                      semaf = "Ambos"

    estados = pd.DataFrame([
        {"Estado":1, "P (kPa)":P1/1e3, "T (K)":T1, "h (kJ/kg)":h1, "s (kJ/kg·K)":s1},
        {"Estado":2, "P (kPa)":P2/1e3, "T (K)":T2, "h (kJ/kg)":h2, "s (kJ/kg·K)":s2},
        {"Estado":3, "P (kPa)":P3/1e3, "T (K)":T3, "h (kJ/kg)":h3, "s (kJ/kg·K)":s3},
        {"Estado":4, "P (kPa)":P4/1e3, "T (K)":T4, "h (kJ/kg)":h4, "s (kJ/kg·K)":
            PropsSI("S","T",T4,"P",P4,FLUIDO_GAS)/1000.0},
        {"Estado":5, "P (kPa)":P4/1e3, "T (K)":T5, "h (kJ/kg)":h5, "s (kJ/kg·K)":
            PropsSI("S","T",T5,"P",P4,FLUIDO_GAS)/1000.0},
        {"Estado":6, "P (kPa)":p7_bar*100, "T (K)":hx["T6"], "h (kJ/kg)":hx["h6"], "s (kJ/kg·K)":hx["s6"]},
        {"Estado":7, "P (kPa)":p7_bar*100, "T (K)":hx["T7_real"], "h (kJ/kg)":hx["h7_real"], "s (kJ/kg·K)":hx["s7_real"]},
        {"Estado":8, "P (kPa)":p8_bar*100, "T (K)":hx["T8"], "h (kJ/kg)":hx["h8"], "s (kJ/kg·K)":hx["s8"]},
    ])

    calculos = {
        "Q_in (kJ/kg)":Q_in, "Q_out (kJ/kg)":Q_out,
        "W_comp (kJ/kg)":W_comp, "W_turb (kJ/kg)":W_turb,
        "W_net (kJ/kg)":W_net, "W_ele (kJ/kg)":W_ele,
        "Q_sum (kJ/kg)":Q_sum, "eta_ciclo_electrico (%)":eta_ciclo,
        "m_dot_gas (kg/s)":m_dot_gas, "P_net (kW)":P_net, "P_elec (kW)":P_elec,
        "m_dot_fuel (kg/s)":m_dot_fuel, "SFC_elec (kg/kWh)":SFC,
        "Heat rate_elec (kJ/kWh)":heat_rate,
        "T4 (K)":T4, "T5_gas_out (K)":T5,
        "Q_gc (kW)":Q_gc, "Q_user (kW)":Q_user,
        "eta_cog_global (%)":eta_cog,
    }

    cog = {
        "U (kW/m2K)":U_GLOBAL, "A (m2)":A_GLOBAL,
        "cp_gas (kJ/kgK)":hx["cp_gas"], "NTU":hx["NTU"], "epsilon":hx["eps"],
        "m_dot_w (kg/s)":hx["m_dot_w"],
        "Q_pre (kW)":hx["Q_pre (kW)"],
        "Q_plateau_NTU (kW)":hx["Q_plateau_NTU (kW)"],
        "Q_req_total (kW)":hx["Q_req_total (kW)"],
        "Q_to_water (kW)":hx["Q_to_water (kW)"],
        "T7_obj (K)":hx["T7_obj"], "T7_real (K)":hx["T7_real"], "Tsat7 (K)":hx["Tsat7"],
        "Temp_GC_OK":ok_gc_temp, "Temp_usuario_OK":ok_user_temp,
        "Semaforo_HX":semaf
    }

    return estados, calculos, cog


# =============================================================================
# Main vectorizado + Excel con Validaciones
# =============================================================================
def main():
    # (opcional) limpiar cache entre corridas largas:
    # _PropsSI_cached.cache_clear()

    # Entradas
    df_mun = pd.read_excel("Municipios_D.xlsx")
    df_mun["T1 (K)"]   = df_mun["Temperatura (°C)"] + 273.15
    df_mun["P1 (kPa)"] = df_mun["Presión (bares)"] * 100
    df_tur = pd.read_csv("Base_de_datos_turbinas_de_gas.csv")

    # Producto cartesiano
    df_mun["_k"]=1; df_tur["_k"]=1
    df_cross = pd.merge(df_mun, df_tur, on="_k").drop(columns="_k")

    # --- Nombres "seguros" para itertuples (sin espacios/paréntesis) ---
    colmap = {
        "T1 (K)":       "T1_K",
        "P1 (kPa)":     "P1_kPa",
        "T3 (C)":       "T3_C",
        "Potencia (kW)":"Potencia_kW",
    }
    df_cross = df_cross.rename(columns=colmap)

    estados_list, calc_list, cog_list = [], [], []
    for row in df_cross.itertuples(index=False):
        est, calc, cog = simular_ciclo(
            T1=row.T1_K,
            P1_kPa=row.P1_kPa,
            r_p=row.r_p,
            T3=row.T3_C,
            eta_c=row.eta_c,
            eta_t=row.eta_t,
            P_ele=row.Potencia_kW,
            p7_bar=P7_BAR, p8_bar=P8_BAR, delta_subcool_c=DELTA_SUBCOOL_C,
            t7_user_c=T7_USER_C
        )
        # Etiquetas
        est.insert(0, "Municipio", row.Municipio)
        est.insert(1, "Turbina",   row.Turbina)
        estados_list.append(est)

        calc.update({
            "Municipio": row.Municipio,
            "Turbina":   row.Turbina,
            "Potencia etiqueta (kW)": row.Potencia_kW,
        })
        cog.update({
            "Municipio": row.Municipio,
            "Turbina":   row.Turbina,
        })
        calc_list.append(calc); cog_list.append(cog)


    df_est  = pd.concat(estados_list, ignore_index=True)
    df_calc = pd.DataFrame(calc_list)
    df_cog  = pd.DataFrame(cog_list)

    # ---------------- Validaciones (compactas) ----------------
    valid_left = df_calc[[
        "Municipio","Turbina",
        "T4 (K)","T5_gas_out (K)","Q_gc (kW)","Q_user (kW)","eta_cog_global (%)"
    ]]

    valid = valid_left.merge(
        df_cog[[
            "Municipio","Turbina",
            "T7_obj (K)","T7_real (K)","Tsat7 (K)",
            "Temp_GC_OK","Temp_usuario_OK","Semaforo_HX",
            "Q_pre (kW)","Q_plateau_NTU (kW)"
        ]],
        on=["Municipio","Turbina"],
        how="left"
    )

    # Indicadores legibles con tolerancia
    valid["T5_OK (>=120C)"] = valid["T5_gas_out (K)"] >= (120.0 + 273.15)
    valid["T7_OK"]          = valid["T7_real (K)"]    >= (valid["T7_obj (K)"] - 1e-6)

    # Desglose energético y brechas
    valid["Q_avail_NTU (kW)"] = valid["Q_pre (kW)"].fillna(0) + valid["Q_plateau_NTU (kW)"].fillna(0)
    valid["T7_gap (K)"]       = (valid["T7_obj (K)"] - valid["T7_real (K)"]).clip(lower=0)
    valid["T5_margin (K)"]    = valid["T5_gas_out (K)"] - (120.0 + 273.15)

    # Clasificación de régimen térmico en la salida (estado 7)
    epsK = 0.5  # tolerancia ~0.5 K
    def _regimen(t7, tsat):
        if pd.isna(t7) or pd.isna(tsat):
            return "N/A"
        if t7 < tsat - epsK:
            return "Subenfriado"
        elif abs(t7 - tsat) <= epsK:
            return "Ebullición (plateau)"
        else:
            return "Sobrecalentado"

    valid["Regimen_HX"] = [ _regimen(t7, ts) for t7, ts in zip(valid["T7_real (K)"], valid["Tsat7 (K)"]) ]

    # -------- Exportar a Excel --------
    base, idx = "Resultados_Ciclo_Brayton", 1
    fname = f"{base}.xlsx"
    while os.path.exists(fname):
        fname = f"{base}_{idx:02d}.xlsx"; idx += 1

    with pd.ExcelWriter(fname) as w:
        df_est.to_excel(w,  sheet_name="Estados",       index=False)
        df_calc.to_excel(w, sheet_name="Calculos",       index=False)
        df_cog.to_excel(w,  sheet_name="Cogeneracion",   index=False)
        valid.to_excel(w,   sheet_name="Validaciones",   index=False)

    print(f"Simulación completa. Resultados guardados en '{fname}'")

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()

