import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# ---------------- CONFIGURAÇÃO DE ESTILO PROFISSIONAL ----------------
def global_style():
    st.markdown("""
        <style>
        .main {
            background: #f7f7fa !important;
        }
        h1 {
            color: #ffffff !important;
            background: linear-gradient(135deg, #7b2ff7, #f107a3);
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            font-weight: 900;
            margin-bottom: 25px;
        }
        .card {
            background: rgba(255,255,255,0.55);
            backdrop-filter: blur(10px);
            border-radius: 14px;
            padding: 18px;
            text-align: center;
            box-shadow: 0px 5px 14px rgba(0,0,0,0.08);
            font-weight: 600;
            color: #333;
        }
        .card .label {
            font-size: 0.8rem;
            opacity: 0.7;
        }
        .badge {
            padding: 10px 14px;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1rem;
            display: inline-block;
            margin-bottom: 8px;
        }
        .footer {
            text-align:center;
            font-size: 12px;
            color: #666;
            margin-top: 35px;
        }
        </style>
    """, unsafe_allow_html=True)

global_style()

st.set_page_config(page_title="Jobin - Mercado de Trabalho", layout="centered")

st.title("Jobin Inteligente – Salários & Tendências do Mercado")

@st.cache_data
def load_data():
    return pd.read_csv("cache_Jobin1.csv")

df = load_data()

st.write("Digite uma profissão e visualize projeções de salário e demanda:")

term = st.text_input("Profissão:", placeholder="Ex.: Desenvolvedor")

if term:
    filt = df[df["descricao"].str.contains(term, case=False, na=False)]

    if filt.empty:
        st.warning("Nenhuma profissão encontrada para esse termo.")
    else:
        select = st.selectbox(
            "Escolha o CBO:",
            filt.apply(lambda x: f"{x['codigo']} - {x['descricao']}", axis=1)
        )
        cbo = int(select.split(" - ")[0])
        info = filt[filt["codigo"] == cbo].iloc[0]

        st.subheader(f"{info['descricao']} • CBO {cbo}")

        # ---------------- INDICADORES EM CARDS ----------------
        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f"<div class='card'><div>💰</div>R$ {info['salario_medio_atual']:.2f}<br><span class='label'>Salário Médio</span></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'><div>🧠</div>{info['modelo_vencedor']}<br><span class='label'>Modelo</span></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'><div>📊</div>{info['score']:.3f}<br><span class='label'>Score</span></div>", unsafe_allow_html=True)
        
        # Tendência do mercado com ícone e cor profissional
        tendencia_raw = str(info.get("tendencia_mercado", "")).lower()
        if "alta" in tendencia_raw:
            color = "#4CAF50"
            icon = "📈"
        elif "baixa" in tendencia_raw:
            color = "#FF5252"
            icon = "📉"
        else:
            color = "#FFC107"
            icon = "⚖️"

        col4.markdown(
            f"<div class='card'><div>{icon}</div>{info['tendencia_mercado']}<br><span class='label'>Demanda do Mercado</span></div>",
            unsafe_allow_html=True
        )

        # ---------------- PROJEÇÃO SALARIAL ----------------
        anos = ["+5 anos", "+10 anos", "+15 anos", "+20 anos"]
        valores = [
            info["previsao_5"], info["previsao_10"],
            info["previsao_15"], info["previsao_20"]
        ]

        crescimento = ((valores[-1] - valores[0]) / valores[0]) * 100

        if crescimento > 15:
            trend_color = "#4CAF50"
            trend_msg = f"📈 Crescimento Acelerado ({crescimento:.1f}%)"
        elif crescimento > 2:
            trend_color = "#00BCD4"
            trend_msg = f"📈 Crescimento Moderado ({crescimento:.1f}%)"
        elif crescimento > -2:
            trend_color = "#9E9E9E"
            trend_msg = f"⚖️ Estável ({crescimento:.1f}%)"
        else:
            trend_color = "#FF5252"
            trend_msg = f"📉 Queda Salarial ({crescimento:.1f}%)"

        fig = go.Figure(go.Scatter(
            x=anos, y=valores,
            mode="lines+markers",
            line=dict(width=4)
        ))
        fig.update_layout(
            title="Projeção Salarial",
            template="plotly_white",
            yaxis_title="Salário (R$)",
            xaxis_title="Horizonte"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"<div class='badge' style='background:{trend_color};color:white;'>{trend_msg}</div>",
            unsafe_allow_html=True
        )

st.markdown("<div class='footer'>© 2025 Jobin Analytics</div>", unsafe_allow_html=True)
