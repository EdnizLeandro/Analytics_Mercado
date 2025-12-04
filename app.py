import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie

# ================= CONFIGURAÇÃO DA PÁGINA ================
st.set_page_config(
    page_title="Jobin Inteligente - Salários",
    layout="wide"
)

# ================= LEITURA DOS DADOS =====================
df = pd.read_csv("cache_Jobin1.csv")

media_salarial = df["salary"].mean()
score_modelo = 0.968  # Mantido como informado
tendencia_20anos = 12.2  # Crescimento estimado conforme gráfico

# ================= CABEÇALHO =============================
st.markdown(
    """
    <h2 style="color:white; text-align:center;">
        🔎 Jobin Inteligente — Salários & Tendências do Mercado
    </h2>
    """,
    unsafe_allow_html=True
)

st.write("")  # espaçamento

# ================= CARDS SUPERIORES ======================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div style="background:#1E1E1E; padding:18px; border-radius:14px; text-align:center;">
            <h4 style="color:#FFD95A;">💰 Salário Médio</h4>
            <h3 style="color:#ffffff;">R$ {media_salarial:.2f}</h3>
        </div>
        """, unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="background:#1E1E1E; padding:18px; border-radius:14px; text-align:center;">
            <h4 style="color:#4DD4AC;">📊 Modelo</h4>
            <h3 style="color:white;">XGBoost</h3>
        </div>
        """, unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div style="background:#1E1E1E; padding:18px; border-radius:14px; text-align:center;">
            <h4 style="color:#6EB6FF;">🎯 Score</h4>
            <h3 style="color:white;">{score_modelo:.3f}</h3>
        </div>
        """, unsafe_allow_html=True
    )

st.write("")

# ================= GRÁFICO DE PROJEÇÃO SALARIAL ==========
df_proj = df.groupby("years_experience")["salary"].mean().reset_index()
fig_proj = px.line(
    df_proj,
    x="years_experience",
    y="salary",
    title="📈 Projeção Salarial por Experiência",
)
fig_proj.update_layout(
    title_font_size=20,
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font_color="white"
)
st.plotly_chart(fig_proj, use_container_width=True)

# ================= CARD DE TENDÊNCIA =====================
st.markdown(
    f"""
    <div style="background:linear-gradient(90deg,#0A324B,#1B4965); 
                padding:22px; border-radius:14px; margin-top:15px;">
        <h3 style="color:white; margin:0;">
            🚀 Crescimento Moderado (≈ {tendencia_20anos:.1f}% em 20 anos)
        </h3>
        <p style="color:#D9EFFF; font-size:16px; margin-top:8px;">
            O mercado para esta área mantém um crescimento estrutural estável,
            impulsionado por tecnologia, automação e maior demanda por habilidades digitais.
            Expectativa positiva para quem inicia agora sua carreira.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")
