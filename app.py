import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# ========== CONFIGURAÇÃO DA PÁGINA ==========
st.set_page_config(
    page_title="Dashboard Jobin | Mercado de Trabalho",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados 🎨
custom_css = """
<style>
    /* Fundo geral */
    .main {
        background-color: #f8f9fc;
    }

    /* Caixa de inputs */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #B9B9B9;
    }

    /* Métricas */
    .stMetric {
        background: linear-gradient(135deg, #7b2ff7, #f107a3);
        color: white !important;
        padding: 18px;
        border-radius: 18px;
        text-align: center;
    }

    /* Títulos */
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(#7b2ff7, #f107a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Caixa rodapé */
    .footer {
        font-size: 14px;
        opacity: 0.6;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ========== CABEÇALHO ==========
st.title("🔎 Jobin Inteligente — Salários & Tendências do Mercado")
st.markdown("### O futuro da sua carreira, em um clique! 🚀")
st.write(
    "Busque profissões **pelo nome completo ou parcial** "
    "(ex: *desenvolvedor*, *enfermeiro*, *motorista*) e veja projeções e tendências de mercado com base no Novo CAGED 📊"
)

# ========== CARREGAMENTO DOS DADOS ==========
@st.cache_data
def carregar_dados():
    try:
        return pd.read_csv("cache_Jobin1.csv")
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

df = carregar_dados()

# ========== BUSCA ==========
if df is not None:
    termo = st.text_input(
        "🔍 Pesquisar profissão:",
        placeholder="Digite parte do nome... ex: Analista"
    )

    resultado_filtro = pd.DataFrame()
    cbo_selecionado = None
    
    if termo:
        resultado_filtro = df[df['descricao'].str.contains(termo, case=False, na=False)]
        if resultado_filtro.empty:
            st.warning("Nenhuma profissão encontrada. Tente outro termo 👀")
        else:
            st.success(f"{resultado_filtro.shape[0]} profissões encontradas!")

            opcao = st.selectbox(
                "Escolha a profissão desejada:",
                [
                    f"{row['codigo']} - {row['descricao']}" 
                    for _, row in resultado_filtro.iterrows()
                ]
            )
            cbo_selecionado = int(opcao.split(" - ")[0])

    if cbo_selecionado:
        info = resultado_filtro[resultado_filtro['codigo'] == cbo_selecionado].iloc[0]

        st.subheader(f"👔 {info['descricao']} — CBO {info['codigo']}")

        # ========== CARDS DE MÉTRICAS ==========
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Salário Médio Atual", f"R$ {info['salario_medio_atual']:.2f}")
        col2.metric("Modelo de Previsão", info['modelo_vencedor'])
        col3.metric("Score do Modelo", f"{info['score']:.3f}")
        col4.metric("Tendência Salarial", info['tendencia_salarial'])

        # ========== GRÁFICO ==========
        anos = ["+5 anos", "+10 anos", "+15 anos", "+20 anos"]
        salarios = [
            info['previsao_5'],
            info['previsao_10'],
            info['previsao_15'],
            info['previsao_20']
        ]

        fig = go.Figure(go.Scatter(
            x=anos, y=salarios,
            mode="lines+markers",
            marker={"size": 12},
        ))
        fig.update_layout(
            title=f"📈 Projeção Salarial para {info['descricao']}",
            xaxis_title="Horizonte de Tempo",
            yaxis_title="Salário (R$)",
            template="plotly_white",
            title_font_size=20
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"📊 **Tendência do Mercado**: {info['tendencia_mercado']}"
        )
else:
    st.error("Não foi possível carregar os dados. Verifique o arquivo CSV.")

# ========== RODAPÉ ==========
st.markdown(
    "<div class='footer' style='text-align:center;margin-top:40px;'>"
    "© 2025 Jobin Analytics — Powered by Streamlit 👨‍💻✨"
    "</div>",
    unsafe_allow_html=True
)
