import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Sessão: Histórico
if "historico" not in st.session_state:
    st.session_state.historico = {
        "simulacao": [],
        "ds_tq1": [],
        "perda": [],
        "vazao_tq3": []
    }
if "sim_count" not in st.session_state:
    st.session_state.sim_count = 1

# Funções base
def corrigir_densidade_farinha(densidade_base, umidade_percentual):
    return densidade_base * (1 - umidade_percentual / 100)

def estimar_ds_mistura(rho_mistura, rho_farinha, rho_tq3=998):
    return (rho_mistura - rho_tq3) / (rho_farinha - rho_tq3) * 100

def estimar_perda_solidos(vazao_mistura, ds_mistura, vazao_etapa2, ds_etapa2):
    return (vazao_mistura * ds_mistura / 100) - (vazao_etapa2 * ds_etapa2 / 100)

def pid_control(error, prev_error, integral, Kp, Ki, Kd):
    integral += error
    derivative = error - prev_error
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral, error

# Controlador Fuzzy
def create_fuzzy_controller():
    erro = ctrl.Antecedent(np.arange(-50, 51, 1), 'erro')
    delta = ctrl.Antecedent(np.arange(-20, 21, 1), 'delta')
    ajuste = ctrl.Consequent(np.arange(-500, 501, 1), 'ajuste')

    erro.automf(3)
    delta.automf(3)
    ajuste.automf(3)

    regras = [
        ctrl.Rule(erro['poor'] & delta['poor'], ajuste['poor']),
        ctrl.Rule(erro['average'] & delta['average'], ajuste['average']),
        ctrl.Rule(erro['good'] & delta['good'], ajuste['good']),
        ctrl.Rule(erro['poor'] & delta['good'], ajuste['average']),
        ctrl.Rule(erro['good'] & delta['poor'], ajuste['average'])
    ]

    sistema = ctrl.ControlSystem(regras)
    return ctrl.ControlSystemSimulation(sistema)

# Interface
st.set_page_config("Simulador DS com PID/Fuzzy", layout="centered")
st.title("🔬 Simulador de %DS com Controle PID e Fuzzy")

# Entradas
col1, col2 = st.columns(2)
with col1:
    umidade = st.slider("Umidade da farinha (%)", 0.0, 20.0, 11.0, 0.1)
    densidade_base = st.number_input("Densidade base da farinha (kg/m³)", value=1250.0)
    rho_mistura = st.number_input("Densidade da mistura (pós TQ1)", value=1105.0)
with col2:
    rho_etapa2 = st.number_input("Densidade pós etapa 2", value=1045.0)
    vazao_mistura = st.number_input("Vazão mistura (kg/h)", value=5000.0)
    vazao_etapa2 = st.number_input("Vazão etapa 2 (kg/h)", value=4900.0)

# Controle
setpoint = st.slider("Setpoint de %DS pós TQ1", 0.0, 100.0, 30.0, 0.1)
modo_controle = st.selectbox("Modo de controle", ["PID", "Fuzzy"])

# Estado do controle
if "vazao_tq3" not in st.session_state:
    st.session_state.vazao_tq3 = 1000.0
if "prev_erro" not in st.session_state:
    st.session_state.prev_erro = 0
if "integral" not in st.session_state:
    st.session_state.integral = 0
if "fuzzy_sim" not in st.session_state:
    st.session_state.fuzzy_sim = create_fuzzy_controller()

if modo_controle == "PID":
    Kp = st.number_input("Kp", value=0.5)
    Ki = st.number_input("Ki", value=0.05)
    Kd = st.number_input("Kd", value=0.01)

# Simulação
if st.button("📊 Simular"):
    rho_farinha_corr = corrigir_densidade_farinha(densidade_base, umidade)
    ds_tq1 = estimar_ds_mistura(rho_mistura, rho_farinha_corr)
    ds_etapa2 = estimar_ds_mistura(rho_etapa2, rho_farinha_corr)
    perda = estimar_perda_solidos(vazao_mistura, ds_tq1, vazao_etapa2, ds_etapa2)

    erro = setpoint - ds_tq1
    delta = erro - st.session_state.prev_erro

    if modo_controle == "PID":
        ajuste, integral, prev = pid_control(
            erro,
            st.session_state.prev_erro,
            st.session_state.integral,
            Kp, Ki, Kd
        )
        st.session_state.integral = integral
        st.session_state.prev_erro = prev
    else:
        fsim = st.session_state.fuzzy_sim
        fsim.input['erro'] = erro
        fsim.input['delta'] = delta
        fsim.compute()
        ajuste = fsim.output['ajuste']
        st.session_state.prev_erro = erro  # mantém delta funcional

    # Aplica ajuste na vazão do TQ3
    st.session_state.vazao_tq3 += ajuste
    st.session_state.vazao_tq3 = max(0, st.session_state.vazao_tq3)

    # Salva histórico
    st.session_state.historico["simulacao"].append(st.session_state.sim_count)
    st.session_state.historico["ds_tq1"].append(ds_tq1)
    st.session_state.historico["perda"].append(perda)
    st.session_state.historico["vazao_tq3"].append(st.session_state.vazao_tq3)
    st.session_state.sim_count += 1

    # Resultados
    st.success("✅ Simulação concluída!")
    st.write(f"🔹 %DS após TQ1: **{ds_tq1:.2f} %**")
    st.write(f"🔹 %DS após etapa 2: **{ds_etapa2:.2f} %**")
    st.write(f"🔹 Perda de sólidos: **{perda:.2f} kg/h**")
    st.write(f"🔧 Vazão TQ3 ({modo_controle}): **{st.session_state.vazao_tq3:.2f} kg/h**")

# Gráficos
if st.session_state.historico["simulacao"]:
    st.subheader("📈 Histórico das Simulações")
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(st.session_state.historico["simulacao"], st.session_state.historico["ds_tq1"], marker='o')
    ax[0].axhline(setpoint, color='r', linestyle='--', label="Setpoint")
    ax[0].set_ylabel("%DS após TQ1")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(st.session_state.historico["simulacao"], st.session_state.historico["perda"], marker='s', color='orange')
    ax[1].set_ylabel("Perda de sólidos (kg/h)")
    ax[1].set_xlabel("Simulação")
    ax[1].grid(True)

    st.pyplot(fig)

    st.subheader("🔧 Vazão TQ3 ajustada")
    st.line_chart(st.session_state.historico["vazao_tq3"])