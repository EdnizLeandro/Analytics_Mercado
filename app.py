import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# ---------------------------------------------------------------------
# Estilo personalizado do Dashboard - Mais moderno e juvenil
# ---------------------------------------------------------------------
def aplicar_css_customizado():
    st.markdown("""
        <style>
        body {
            background-color: #ffffff;
        }
        .main > div {
            background: linear-gradient(135deg, #ff007f 10%, #b300b3 90%);
            padding: 20px;
            border-radius: 12px;
        }
        h1 {
            color: white !important;
            font-weight: 900;
        }
        .stMetric {
            background: #ffffff10;
            border-radius: 14px;
            padding: 12px;
            text-align: center;
        }
        .tendencia-box {
            background: rgba(255,255,255,0.15);
            padding: 12px 20px;
            color: #fff;
            font-weight: bold;
            border-radius: 10px;
            text-align: center;
            font-size: 1.1rem;
            margin-top: 10px;
        }
        .footer {
            text-align:center;
            color:#f5f5f5;
        }
        </style>
    """, unsafe_allow_html=True)

aplicar_css_customizado()

st.set_page_config(
    page_title="Dashboard Profissões - Salários & Tendências",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🔎 Jobin Inteligente - Salários & Tendências do Mercado")

st.markdown("""
Pesquise por profissão _digitando o nome completo ou parcial_ (ex: **pintor**, **analista**, **enfermeiro**) e escolha o CBO desejado para visualizar projeções salariais e tendências de mercado.
""")

# ---------------------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("cache_Jobin1.csv")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

df = carregar_dados()


# ---------------------------------------------------------------------
# Interface de busca
# ---------------------------------------------------------------------
if df is not None:
    termo = st.text_input(
        "Digite parte do nome da profissão:",
        placeholder="Exemplo: pintor"
    )

    cbo_selecionado = None
    resultado_filtro = pd.DataFrame()
    
    if termo:
        resultado_filtro = df[df['descricao'].str.contains(termo, case=False, na=False)]

        if resultado_filtro.empty:
            st.warning("Nenhuma profissão encontrada. Tente outro termo.")
        else:
            st.write(f"**{resultado_filtro.shape[0]} resultados encontrados para:** '{termo}'")

            nomes_cbos = [
                f"{row['codigo']} - {row['descricao']}"
                for _, row in resultado_filtro.iterrows()
            ]

            cbo_str = st.selectbox(
                "Selecione o CBO e profissão desejada:",
                options=nomes_cbos
            )

            if cbo_str:
                cbo_selecionado = int(cbo_str.split(' - ')[0])
    else:
        st.info("Digite parte do nome da profissão para começar.")

    if cbo_selecionado:
        info = resultado_filtro[resultado_filtro['codigo'] == cbo_selecionado].iloc[0]
        st.subheader(f"Profissão: {info['descricao']} (CBO {info['codigo']})")

        # =====================================================================
        # Indicadores com Ícones 😎
        # =====================================================================
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Salário Médio", f"R$ {float(info['salario_medio_atual']):.2f}")
            st.metric("🧠 Modelo", f"{info['modelo_vencedor']}")
        with col2:
            st.metric("📈 Score", f"{float(info['score']):.4f}")
            # A Tendência será atualizada com base no gráfico

        # =====================================================================
        # Projeção Salarial
        # =====================================================================
        st.markdown("#### 📊 Projeção Salarial (5, 10, 15 e 20 anos)")

        anos = ["+5 anos", "+10 anos", "+15 anos", "+20 anos"]
        valores = [
            float(info['previsao_5']),
            float(info['previsao_10']),
            float(info['previsao_15']),
            float(info['previsao_20'])
        ]

        crescimento_pct = ((valores[-1] - valores[0]) / valores[0]) * 100

        if crescimento_pct > 15:
            tendencia_msg = f"🚀 Crescimento Acelerado ({crescimento_pct:.1f}% em 20 anos)"
            tendencia_cor = "#00e676"
        elif crescimento_pct > 2:
            tendencia_msg = f"📈 Crescimento Moderado ({crescimento_pct:.1f}% em 20 anos)"
            tendencia_cor = "#ffeb3b"
        elif crescimento_pct > -2:
            tendencia_msg = f"⚖️ Estabilidade ({crescimento_pct:.1f}% em 20 anos)"
            tendencia_cor = "#ffffff"
        else:
            tendencia_msg = f"📉 Queda Salarial ({crescimento_pct:.1f}% em 20 anos)"
            tendencia_cor = "#ff5252"

        fig = go.Figure(
            go.Scatter(
                x=anos,
                y=valores,
                mode='lines+markers',
                marker=dict(size=11),
                line=dict(width=4)
            )
        )
        fig.update_layout(
            title=f"Evolução Salarial de {info['descricao']}",
            xaxis_title="Tempo",
            yaxis_title="Salário (R$)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tendência com destaque visual
        st.markdown(
            f"""
            <div class="tendencia-box" style="background:{tendencia_cor};">
                {tendencia_msg}
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.error("Erro ao carregar a base de dados CSV.")

# Rodapé
st.markdown(
    "<br><div class='footer'>© 2025 Jobin Analytics | Powered by Streamlit</div>",
    unsafe_allow_html=True
)
