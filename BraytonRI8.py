# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:54:28 2025

@author: Gabo San
"""

from CoolProp.CoolProp import PropsSI
import pandas as pd
import os
import math
import json

# Parámetros "globales" (comunes a todas las combinaciones)
fluido = "Air"
PCI = 55090        # Poder Calorífico Inferior del combustible (kJ/kg)
eta_gen = 0.98     # Eficiencia del generador
eta_cc = 0.95      # Eficiencia de la cámara de combustión

# -----------------------------
# PARÁMETROS PARA COGENERACIÓN / HRSG
# -----------------------------
# Servicio térmico (los podrá cambiar el usuario que "sabe"):
modo_servicio = "vapor"   # "agua" o "vapor" (solo informativo)
P_serv_bar = 10.0         # [bar abs] presión de servicio (estados 6 y 7)
T_serv_C = 180.0          # [°C] temperatura de servicio (estado 7)
P_ret_bar = 2.0           # [bar abs] presión de retorno (estado 8)

# Caudal de agua de servicio (puedes ajustar a tu caso)
m_dot_agua = 5.0          # [kg/s] flujo másico de agua en el lazo 6-7-8

# Datos del intercambiador de calor (HRSG) para método NTU
UA_HRSG = 1500.0          # [kJ/s·K] ~ kW/K, AJUSTAR según tu diseño
T5_min_C = 100.0          # [°C] temperatura mínima permitida de gases a la salida (estado 5)

# Parámetros para cálculo de densidad del aire
R_air = 287.0          # [J/kg·K] constante de gas para el aire
T_ref = 288.15         # [K] ~ 15 °C, condición de referencia tipo ISO
P_ref = 1.013e5        # [Pa] ~ 1.013 bar
rho_ref = P_ref / (R_air * T_ref)  # [kg/m³] densidad de referencia

# -----------------------------
# LECTURA OPCIONAL DE CONFIGURACIÓN EXTERNA
# -----------------------------
CONFIG_FILE = "config_brayton.json"

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Parámetros globales
        fluido = cfg.get("fluido", fluido)
        PCI = cfg.get("PCI", PCI)
        eta_gen = cfg.get("eta_gen", eta_gen)
        eta_cc = cfg.get("eta_cc", eta_cc)

        # Parámetros de cogeneración
        modo_servicio = cfg.get("modo_servicio", modo_servicio)
        P_serv_bar = cfg.get("P_serv_bar", P_serv_bar)
        T_serv_C = cfg.get("T_serv_C", T_serv_C)
        P_ret_bar = cfg.get("P_ret_bar", P_ret_bar)
        m_dot_agua = cfg.get("m_dot_agua", m_dot_agua)
        UA_HRSG = cfg.get("UA_HRSG", UA_HRSG)
        T5_min_C = cfg.get("T5_min_C", T5_min_C)

    except Exception as e:
        print(f"Advertencia: no se pudo leer {CONFIG_FILE}: {e}")
        print("Se usarán los parámetros por defecto definidos en el script.")

def efectividad_contra(NTU, C_r):
    """
    Efectividad de un intercambiador contracorriente usando el método NTU.
    NTU = UA/C_min ; C_r = C_min/C_max.
    """
    if NTU <= 0.0:
        return 0.0
    # C_r -> 0 (flujo caliente muy grande vs frío)
    if C_r < 1e-6:
        return 1.0 - math.exp(-NTU)
    # C_r ~ 1 (C_min ~= C_max)
    if abs(1.0 - C_r) < 1e-6:
        return NTU / (1.0 + NTU)
    # Caso general
    return (1.0 - math.exp(-NTU * (1.0 - C_r))) / (1.0 - C_r * math.exp(-NTU * (1.0 - C_r)))


# Leer bases de datos
turbinas_df = pd.read_csv("Base_de_datos_turbinas_de_gas.csv")
municipios_df = pd.read_excel("Municipios_D.xlsx")

# Listas para acumular resultados
filas_estados = []
filas_resultados = []

# Recorremos todas las turbinas x todos los municipios (producto cruzado)
for _, turb in turbinas_df.iterrows():
    nombre_turbina = turb["Turbina"]
    P_ele = turb["Potencia (kW)"]        # Potencia de placa de la turbina [kW]
    T3_C = turb["T3 (C)"]               # [°C]
    r_p = turb["r_p"]
    eta_c = turb["eta_c"]
    eta_t = turb["eta_t"]
    m_aire_nom = turb["m_aire (kg/s)"]   # Flujo másico nominal de aire [kg/s]
    CTU = turb["CTU (kJ/kWh)"]          # Consumo térmico unitario [kJ/kWh]
    T3_max = T3_C + 273.15             # [K] temperatura máxima admisible a la entrada de turbina
    
    # Flujo de combustible nominal a condiciones de placa (ISO)
    if P_ele > 0 and CTU > 0:
        Qdot_comb_nom = P_ele * CTU / 3600.0   # [kJ/s] potencia térmica nominal del combustible
        m_dot_fuel_nom = Qdot_comb_nom / PCI   # [kg/s] flujo másico nominal de combustible
    else:
        Qdot_comb_nom = 0.0
        m_dot_fuel_nom = 0.0 

    for _, loc in municipios_df.iterrows():
        nombre_mpio = loc["Municipio"]
        T1_C = loc["Temperatura (°C)"]  # [°C]
        P1_bar = loc["Presión (bares)"] # [bar]

        # Condiciones de entrada del aire según municipio
        T1 = T1_C + 273.15              # [K]
        P1 = P1_bar * 1e5               # [Pa] (1 bar = 1e5 Pa)
        
        # Densidad del aire en el sitio (usamos P y T locales)
        # Flujo másico de gases de combustión ≈ flujo másico de aire
        rho_site = P1 / (R_air * T1)    # [kg/m³]
        m_dot_gas_real = m_aire_nom * (rho_site / rho_ref)  # [kg/s]

        # Presiones en el ciclo
        P2 = P1 * r_p                   # [Pa]
        P3 = P2                         # Calentamiento a presión constante
        P4 = P1                         # Expansión hasta la presión inicial

        # -----------------------------
        # CÁLCULO DE ESTADOS
        # -----------------------------
        # Estado 1
        h1_Jkg = PropsSI("H", "T", T1, "P", P1, fluido)    # [J/kg]
        s1_JkgK = PropsSI("S", "T", T1, "P", P1, fluido)   # [J/kg·K]
        h1 = h1_Jkg / 1000.0                               # [kJ/kg]
        s1 = s1_JkgK / 1000.0                              # [kJ/kg·K]

        # Proceso 1-2: Compresión real
        T2s = PropsSI("T", "S", s1_JkgK, "P", P2, fluido)  # [K]
        T2 = T1 + (T2s - T1) / eta_c                       # [K]
        h2_Jkg = PropsSI("H", "T", T2, "P", P2, fluido)    # [J/kg]
        s2_JkgK = PropsSI("S", "T", T2, "P", P2, fluido)   # [J/kg·K]
        h2 = h2_Jkg / 1000.0                               # [kJ/kg]
        s2 = s2_JkgK / 1000.0                              # [kJ/kg·K]

        # -----------------------------
        # CÁMARA DE COMBUSTIÓN CONTROLADA POR T3_max
        # -----------------------------
        # Se fija T3 directamente al límite metalúrgico de la turbina.
        T3 = T3_max
        h3_Jkg = PropsSI("H", "T", T3, "P", P3, fluido)      # [J/kg]
        h3 = h3_Jkg / 1000.0                                 # [kJ/kg]
        # Calor útil absorbido por el gas (3-2)
        Q_in = h3 - h2                                       # [kJ/kg]
        # Calor ligado al combustible (antes de pérdidas en la cámara)
        Q_sum = Q_in / eta_cc                                # [kJ/kg]

        # Flujo de combustible real requerido para alcanzar T3_max
        if m_dot_gas_real > 0.0 and Q_sum > 0.0:
            Qdot_comb = Q_sum * m_dot_gas_real                    # [kJ/s]
            m_dot_fuel_actual = Qdot_comb / PCI              # [kg/s]
        else:
            Qdot_comb = 0.0
            m_dot_fuel_actual = 0.0

        # Entropía en el estado 3 con T3 fijada a T3_max
        h3_Jkg = h3 * 1000.0
        s3_JkgK = PropsSI("S", "T", T3, "P", P3, fluido)     # [J/kg·K]
        s3 = s3_JkgK / 1000.0                                # [kJ/kg·K]

        # Proceso 3-4: Expansión real en la turbina (con el nuevo T3)
        T4s = PropsSI("T", "S", s3_JkgK, "P", P4, fluido)  # [K]
        T4 = T3 - eta_t * (T3 - T4s)                       # [K]
        h4_Jkg = PropsSI("H", "T", T4, "P", P4, fluido)    # [J/kg]
        s4_JkgK = PropsSI("S", "T", T4, "P", P4, fluido)   # [J/kg·K]
        h4 = h4_Jkg / 1000.0                               # [kJ/kg]
        s4 = s4_JkgK / 1000.0                              # [kJ/kg·K]

        # -----------------------------
        # CÁLCULO DE TRABAJOS Y EFICIENCIAS
        # -----------------------------
        Q_out = h4 - h1             # [kJ/kg]
        W_comp = h2 - h1            # [kJ/kg]
        W_turb = h3 - h4            # [kJ/kg]
        W_net = W_turb - W_comp     # [kJ/kg]
        W_ele = W_net * eta_gen     # [kJ/kg]

        # Eficiencia térmica del ciclo tomando como entrada el calor ligado al combustible
        eta_ciclo = (W_ele / Q_sum) * 100.0  # [%]

        # -----------------------------
        # FLUJO MÁSICO Y POTENCIA
        # -----------------------------
        # m_dot_gas ya fue definido a partir de la densidad del sitio y del flujo nominal.
        m_dot_fuel = m_dot_fuel_actual  # [kg/s] flujo real de combustible después del control de T3

        P_net = m_dot_gas_real * W_net    # [kW]
        P_elec = eta_gen * P_net     # [kW]

        if P_ele != 0:
            derate_P = P_elec / P_ele                         # [-] fracción de la potencia nominal
            porcentaje_error = abs(P_elec - P_ele) / P_ele * 100.0
        else:
            derate_P = 0.0
            porcentaje_error = 0.0
            
        # Heat Rate "real" calculado como el inverso de la eficiencia térmica del ciclo
        # eta_ciclo está en [%], por lo que primero se pasa a fracción.
        if eta_ciclo > 0.0:
            eta_ciclo_frac = eta_ciclo / 100.0
            HeatRate_real = 3600.0 / eta_ciclo_frac  # [kJ/kWh]
        else:
            HeatRate_real = 0.0

        # Comparación del Heat Rate real contra el CTU de placa (derate de HR)
        if CTU > 0.0 and HeatRate_real > 0.0:
            derate_HR = HeatRate_real / CTU
        else:
            derate_HR = 0.0
            
        # -----------------------------
        # MÓDULO DE COGENERACIÓN (HRSG + USUARIO)
        # -----------------------------
        # Valores por defecto (por si la cogeneración no es factible)
        cogen_factible = False
        T5_C = float("nan")
        T6_C = float("nan")
        T7_C_real = float("nan")
        T8_C = float("nan")
        Q_gc_kW = float("nan")
        Q_user_kW = float("nan")
        eta_cog = float("nan")
        tipo_servicio = None
        eta_uso = float("nan")
        
        # Lista auxiliar de estados 5–8
        estados_cogen_local = []

        # Solo tiene sentido intentar cogeneración si hay combustible y gases calientes
        if m_dot_fuel_actual > 0.0 and m_dot_gas_real > 0.0:
            try:
                # Presiones de servicio y retorno (bar -> Pa)
                P_serv = P_serv_bar * 1e5
                P_ret = P_ret_bar * 1e5

                # -----------------------------
                # Estados 6 y 8
                # -----------------------------
                # Saturaciones a presiones de servicio y retorno
                Tsat_serv_K = PropsSI("T", "P", P_serv, "Q", 0, "Water")  # T_sat a P_serv (referencia)
                Tsat8_K = PropsSI("T", "P", P_ret, "Q", 0, "Water")       # T_sat a P_ret

                # Estado 8 (retorno del usuario):
                # El usuario entrega a P_ret y T8 = T_sat(P_ret) - 15°C
                T8_K = Tsat8_K - 15.0
                h8 = PropsSI("H", "P", P_ret, "T", T8_K, "Water") / 1000.0  # [kJ/kg]

                # Estado 6 (a la entrada del HRSG, después de la bomba):
                # Misma temperatura que en 8, pero con la presión elevada a P_serv
                T6_K = T8_K
                h6 = PropsSI("H", "P", P_serv, "T", T6_K, "Water") / 1000.0  # [kJ/kg]
                
                # -----------------------------
                # Datos para el HRSG (método NTU)
                # -----------------------------
                T4_K = T3 - eta_t * (T3 - T4s)  # ya está como T4, pero por claridad
                T4_K = T4                       # usamos el T4 calculado arriba
                T5_min_K = T5_min_C + 273.15

                # Calores específicos promedio (kJ/kg·K)
                cp_gas = PropsSI("C", "T", 0.5 * (T4_K + T5_min_K), "P", P4, fluido) / 1000.0
                T_serv_K = T_serv_C + 273.15
                cp_agua = PropsSI("C", "T", 0.5 * (T6_K + T_serv_K), "P", P_serv, "Water") / 1000.0

                C_hot = m_dot_gas_real * cp_gas    # [kJ/s·K]
                C_cold = m_dot_agua * cp_agua      # [kJ/s·K]

                if C_hot > 0.0 and C_cold > 0.0 and T4_K > T6_K:
                    C_min = min(C_hot, C_cold)
                    C_max = max(C_hot, C_cold)
                    C_r = C_min / C_max

                    NTU = UA_HRSG / C_min          # adimensional
                    eps = efectividad_contra(NTU, C_r)

                    # Límite por NTU (sin considerar T5_min todavía)
                    Q_NTU = eps * C_min * (T4_K - T6_K)    # [kJ/s]

                    # Límite por T5 >= 100°C, usando entalpías para no pasarnos de la energía real en gases
                    h4_gas = h4  # [kJ/kg] ya lo tienes calculado arriba
                    h5min_Jkg = PropsSI("H", "T", T5_min_K, "P", P4, fluido)
                    h5min_gas = h5min_Jkg / 1000.0  # [kJ/kg]
                    Q_T5 = m_dot_gas_real * max(0.0, h4_gas - h5min_gas)  # [kJ/s]

                    # Límite energético por combustible:
                    # Por kg de gas: Q_sum = energía del combustible; W_ele = trabajo eléctrico.
                    # Lo que "sobra" para calor útil es Q_sum - W_ele (por kg de gas).
                    # En flujo: Q_fuel_cap = m_dot_gas_real * (Q_sum - W_ele)
                    Q_fuel_cap = m_dot_gas_real * max(0.0, Q_sum - W_ele)  # [kJ/s]

                    # Calor máximo disponible para cogeneración (respetando NTU, T5 y combustible)
                    Q_cap_thermal = min(Q_NTU, Q_T5)
                    Q_cap = min(Q_cap_thermal, Q_fuel_cap)

                    # Calor requerido para alcanzar la T de servicio deseada
                    Q_req = C_cold * (T_serv_K - T6_K)  # [kJ/s]


                    if Q_cap > 0.0 and Q_req > 0.0:
                        cogen_factible = True

                        # ¿Se alcanza la temperatura pedida?
                        if Q_req <= Q_cap:
                            # Sí: usamos la T de servicio pedida
                            Q_gc = Q_req
                            T7_K = T_serv_K
                        else:
                            # No: solo alcanzamos lo que permite el HRSG
                            Q_gc = Q_cap
                            T7_K = T6_K + Q_gc / C_cold

                        # Estado 5: salida de gases del HRSG
                        T5_K = T4_K - Q_gc / C_hot
                            
                        # Entalpía en el estado 7
                        h7 = PropsSI("H", "P", P_serv, "T", T7_K, "Water") / 1000.0  # [kJ/kg]

                        # Potencias térmicas
                        # Idealmente Q_gc_kW ≈ Q_gc, pero lo tomamos de entalpías del agua.
                        Q_gc_kW = m_dot_agua * (h7 - h6)   # [kJ/s] ~ kW
                        # No permitir que Q_gc_kW supere los topes energéticos
                        Q_gc_kW = min(Q_gc_kW, Q_cap, Q_fuel_cap)
                        
                        Q_user_kW = m_dot_agua * (h7 - h8) # [kJ/s] ~ kW

                        # Clasificación simple del tipo de servicio en 7
                        # Tsat_serv_K ya es T_sat a la presión de servicio P_serv
                        if T7_K < Tsat_serv_K - 5.0:
                            tipo_servicio = "agua_liquida"
                        elif T7_K > Tsat_serv_K + 5.0:
                            tipo_servicio = "vapor_sobrecalentado"
                        else:
                            tipo_servicio = "vapor_saturado"

                        # Factor de utilización del calor por parte del usuario
                        eta_uso = float("nan")
                        if Q_gc_kW > 0.0:
                            eta_uso = Q_user_kW / Q_gc_kW   # puede ser >1 si el retorno está muy frío

                        # Eficiencia de cogeneración global:
                        denom = m_dot_fuel_actual * PCI  # [kJ/s] = kW
                        if denom > 0.0:
                            eta_cog = (P_elec + Q_gc_kW) / denom * 100.0
                            # Por seguridad numérica, recortamos a 100.1%
                            eta_cog = min(eta_cog, 100.1)


                        # Guardar temperaturas en °C para la tabla
                        T5_C = T5_K - 273.15
                        T6_C = T6_K - 273.15
                        T7_C_real = T7_K - 273.15
                        T8_C = T8_K - 273.15
                        
                        # -----------------------------
                        # ESTADOS 5–8 PARA TABLA_ESTADOS
                        # -----------------------------
                        # Estado 5: gases a la salida del HRSG
                        h5_Jkg = PropsSI("H", "T", T5_K, "P", P4, fluido)
                        s5_JkgK = PropsSI("S", "T", T5_K, "P", P4, fluido)
                        h5 = h5_Jkg / 1000.0          # [kJ/kg]
                        s5 = s5_JkgK / 1000.0        # [kJ/kg·K]

                        # Estado 6: agua a la entrada del HRSG (ya tenemos h6, T6_K)
                        s6_JkgK = PropsSI("S", "P", P_serv, "T", T6_K, "Water")
                        s6 = s6_JkgK / 1000.0        # [kJ/kg·K]

                        # Estado 7: agua/vapor a la salida del HRSG (ya tenemos h7, T7_K)
                        s7_JkgK = PropsSI("S", "P", P_serv, "T", T7_K, "Water")
                        s7 = s7_JkgK / 1000.0        # [kJ/kg·K]

                        # Estado 8: retorno del usuario (ya tenemos h8, T8_K)
                        s8_JkgK = PropsSI("S", "P", P_ret, "T", T8_K, "Water")
                        s8 = s8_JkgK / 1000.0        # [kJ/kg·K]

                        # Guardamos estados extra en la lista auxiliar
                        estados_cogen_local = [
                            (5, T5_K, P4,     h5, s5),     # gases, P4
                            (6, T6_K, P_serv, h6, s6),     # agua antes del HRSG
                            (7, T7_K, P_serv, h7, s7),     # agua/vapor después del HRSG
                            (8, T8_K, P_ret,  h8, s8),     # agua de retorno
                        ]


            except ValueError:
                # Algún estado fuera de rango para CoolProp -> dejamos NaN y cogen_factible = False
                pass


        # -----------------------------
        # ACUMULAR TABLA DE ESTADOS
        # -----------------------------
        estados_local = [
            (1, T1, P1, h1, s1),
            (2, T2, P2, h2, s2),
            (3, T3, P3, h3, s3),
            (4, T4, P4, h4, s4),
        ]

        # Si hay cogeneración factible, añadimos 5–8 al mismo listado
        estados_local += estados_cogen_local 

        for est, T_est, P_est, h_est, s_est in estados_local:
            filas_estados.append({
                "Turbina": nombre_turbina,
                "Municipio": nombre_mpio,
                "Estado": est,
                "T [K]": T_est,
                "P [Pa]": P_est,
                "h [kJ/kg]": h_est,
                "s [kJ/kg·K]": s_est,
            })

        # -----------------------------
        # ACUMULAR RESULTADOS
        # -----------------------------
        filas_resultados.append({
            "Turbina": nombre_turbina, "Municipio": nombre_mpio,
            "Potencia_deseada [kW]": P_ele, "Q_in [kJ/kg]": Q_in,
            "Q_out [kJ/kg]": Q_out, "W_comp [kJ/kg]": W_comp,
            "W_turb [kJ/kg]": W_turb, "W_net [kJ/kg]": W_net,
            "W_ele [kJ/kg]": W_ele, "Q_sum [kJ/kg]": Q_sum,
            "Eficiencia_ciclo [%]": eta_ciclo, "m_dot_gas [kg/s]": m_dot_gas_real,
            "m_dot_fuel_nom [kg/s]": m_dot_fuel_nom,
            "m_dot_fuel_real [kg/s]": m_dot_fuel,
            "P_net [kW]": P_net, "P_elec [kW]": P_elec,
            "Error_P_elec [%]": porcentaje_error,
            "Derate_P [-]": derate_P,
            "HeatRate_real (kJ/kWh)": HeatRate_real,
            "Derate_HR [-]": derate_HR,
            "Altitud (media) [m]": loc["Altitud (media)"],
            "Temperatura_mpio [°C]": T1_C,
            "Presión_mpio [bar]": P1_bar,
            "CTU (kJ/kWh)": CTU,
            "m_aire_nom (kg/s)": m_aire_nom,
            "T3_calc [K]": T3,
            "T3_max [K]": T3_max,

            # --- Resultados de cogeneración ---
            "Q_gc_HRSG [kW]": Q_gc_kW,
            "Q_user [kW]": Q_user_kW,
            "T5_gases [°C]": T5_C,
            "T6_agua [°C]": T6_C,
            "T7_agua_serv [°C]": T7_C_real,
            "T8_agua_ret [°C]": T8_C,
            "P6_P7_servicio [bar]": P_serv_bar,
            "P8_retorno [bar]": P_ret_bar,
            "m_dot_agua [kg/s]": m_dot_agua,
            "Eficiencia_cogeneracion [%]": eta_cog,
            "Eficiencia_uso_usuario [-]": eta_uso,
            "Servicio_factible": cogen_factible,
            "Tipo_servicio": tipo_servicio,            
        })

# Construir DataFrames globales
tabla_estados = pd.DataFrame(filas_estados)
df_resultados = pd.DataFrame(filas_resultados)

# -----------------------------
# GUARDAR RESULTADOS EN EXCEL
# -----------------------------
base_filename = "resultados_brayton"
extension = ".xlsx"
filename = base_filename + extension

counter = 1
while os.path.exists(filename):
    filename = f"{base_filename}{counter:02d}{extension}"
    counter += 1

with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
    tabla_estados.to_excel(writer, sheet_name="Estados", index=False)
    df_resultados.to_excel(writer, sheet_name="Resultados", index=False)
print(f"Resultados guardados en: {filename}")
