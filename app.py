import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Configuração
st.set_page_config(
    page_title="Jobin Inteligente | Mercado de Trabalho",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Paleta e estilo do app
st.markdown("""
    <style>
    .main-title {
        font-size: 26px !important;
        font-weight: 700;
        color: white !important;
        background: linear-gradient(90deg, #6A11CB, #2575FC);
        padding: 12px 18px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        gap: 14px;
    }
    .tendencia-box {
        background-color: #f5f7ff;
        border-left: 6px solid #4a6cff;
        padding: 12px;
        border-radius: 10px;
        font-size: 15px;
        margin-top: 10px;
    }
    .footer {
        text-align:center;
        color: grey;
        margin-top: 35px;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🔎 Jobin Inteligente - Salários & Tendências do Mercado</div>", unsafe_allow_html=True)

# Carregamento do CSV
@st.cache_data
def carregar_dados():
    try:
        return pd.read_csv("cache_Jobin1.csv")
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")
        return None

df = carregar_dados()

# Se dados carregaram:
if df is not None:

    termo = st.text_input(
        "Digite parte do nome da profissão:",
        placeholder="Ex: programador, designer, mecânico..."
    )

    resultado_filtro = pd.DataFrame()
    cbo_selecionado = None

    if termo:
        resultado_filtro = df[df['descricao'].str.contains(termo, case=False, na=False)]

        if resultado_filtro.empty:
            st.warning("Nenhuma profissão encontrada. Tente outro termo!")
        else:
            st.success(f"{resultado_filtro.shape[0]} profissões encontradas! Selecione uma abaixo 👇")

            opcoes = [
                f"{row['codigo']} - {row['descricao']}"
                for _, row in resultado_filtro.iterrows()
            ]

            cbo_str = st.selectbox("Escolha o CBO desejado:", options=opcoes)
            cbo_selecionado = int(cbo_str.split(" - ")[0])

    else:
        st.info("🔍 Comece digitando o nome da profissão...")

    # Se uma profissão for selecionada:
    if cbo_selecionado:
        info = resultado_filtro[resultado_filtro['codigo'] == cbo_selecionado].iloc[0]

        st.subheader(f"✨ Profissão Selecionada: **{info['descricao']}** (CBO {info['codigo']})")

        # Métricas principais
        col1, col2, col3 = st.columns(3)

        col1.metric("💰 Salário Médio", f"R$ {float(info['salario_medio_atual']):.2f}")
        col2.metric("🤖 Modelo de Previsão", f"{info['modelo_vencedor']}")
        col3.metric("📈 Score do Modelo", f"{float(info['score']):.3f}")

        # Gráfico da projeção salarial
        anos = ["5 anos", "10 anos", "15 anos", "20 anos"]
        values = [
            float(info['previsao_5']),
            float(info['previsao_10']),
            float(info['previsao_15']),
            float(info['previsao_20'])
        ]

        # Cálculo de tendência salarial
        salario_atual = float(info["salario_medio_atual"])
        salario_20 = values[-1]
        crescimento = ((salario_20 - salario_atual) / salario_atual) * 100

        if crescimento > 18:
            status = "🚀 Crescimento Acelerado"
            cor_grafico = "#09BC8A"
        elif crescimento > 8:
            status = "📈 Crescimento Moderado"
            cor_grafico = "#4A6CFF"
        else:
            status = "⚠ Crescimento Baixo"
            cor_grafico = "#C94A4A"

        fig = go.Figure(
            go.Scatter(
                x=anos,
                y=values,
                mode='lines+markers',
                line=dict(width=3, color=cor_grafico),
                marker=dict(size=10)
            )
        )
        fig.update_layout(
            title="📉 Projeção Salarial",
            xaxis_title="Prazo da projeção",
            yaxis_title="Salário (R$)",
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tendência coerente com o gráfico
        st.markdown(f"""
        <div class="tendencia-box">
        <b>{status}</b>  
        → Projeção: <b>{crescimento:.1f}%</b> nos próximos 20 anos
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ O arquivo 'cache_Jobin1.csv' não foi encontrado ou falhou no carregamento.")

# Rodapé
st.markdown(
    "<div class='footer'>© 2025 Jobin Analytics  —  Dados do Novo CAGED</div>",
    unsafe_allow_html=True
)
