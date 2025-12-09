import streamlit as st 
import pandas as pd
import plotly.graph_objs as go
import math

# Função de formatação brasileira
def brl(value):
    try:
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "R$ 0,00"

# CONFIGURAÇÃO DA PÁGINA 
st.set_page_config(
    page_title="Jobin - Salários & Tendências",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  CSS & VISUAL 
st.markdown(
    """
    <style>
    .reportview-container .main {
        background: #f5f7fb;
        padding-top: 12px;
        padding-bottom: 30px;
    }
    .title-banner {
        background: linear-gradient(90deg,#7b2ff7 0%, #f107a3 100%);
        padding: 18px 22px;
        border-radius: 12px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 6px 18px rgba(23,0,102,0.12);
        margin-bottom: 18px;
    }
    .title-banner h1 {
        margin: 0;
        font-size: 20px;
        font-weight: 900;
        color: white;
    }
    .subtitle {
        margin: 0;
        color: #f1e7ff;
        opacity: 0.95;
        font-size: 13px;
    }
    .card {
        background: rgba(255,255,255,0.7);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(15,15,20,0.04);
        min-height: 110px;
    }
    .card .icon {
        font-size: 26px;
        margin-bottom: 6px;
    }
    .card .value {
        font-size: 18px;
        font-weight: 800;
        color: #111827;
    }
    .card .label {
        display:block;
        font-size: 12px;
        color: #6b7280;
        margin-top: 6px;
        font-weight:600;
    }
    .muted {
        font-size: 12px;
        color: #6b7280;
    }
    .footer {
        text-align:center;
        color:#9aa0b4;
        font-size:13px;
        margin-top:30px;
    }
    </style>
    """, unsafe_allow_html=True
)

#  BANNER 
st.markdown(
    """
    <div class="title-banner">
        <div>
            <h1>🔎 Jobin Inteligente - Salários & Tendências do Mercado de Trabalho</h1>
            <div class="subtitle">Pesquise profissões, veja projeções salariais e demanda do mercado.</div>
        </div>
        <div style="text-align:right;font-size:13px;color:#fff;opacity:0.9;">
            © 2025 Jobin Analytics
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

#  DADOS 
@st.cache_data
def carregar_dados(path="cache_Jobin1.csv"):
    try:
        df = pd.read_csv(path, encoding="utf-8")
        return df
    except FileNotFoundError:
        st.error(f"❌ Arquivo '{path}' não foi encontrado no diretório da aplicação.")
        st.stop()
    except Exception as e:
        st.error(f"⚠ Erro ao carregar CSV: {e}")
        st.stop()

df = carregar_dados()

st.markdown("**Busque por profissão (nome parcial ou completo):**")
termo = st.text_input("", placeholder="Ex.: Analista, Enfermeiro…")

if termo:
    resultados = df[df["descricao"].str.contains(termo, case=False, na=False)]
    
    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        st.success(f"{len(resultados)} encontrados")
        escolha = st.selectbox(
            "Selecione a profissão:",
            resultados.apply(lambda x: f"{int(x['codigo'])} - {x['descricao']}", axis=1)
        )

        cbo = int(escolha.split(" - ")[0])
        info = resultados[resultados["codigo"] == cbo].iloc[0]

        st.markdown(f"### {info['descricao']}  •  CBO {cbo}")
        st.markdown("<span class='muted'>Base Jobin + Novo CAGED</span>", unsafe_allow_html=True)
        st.write("")

        # CARDS
        col1, col2, col3, col4 = st.columns(4, gap="large")
        salario = float(info.get("salario_medio_atual", 0.0))
        modelo  = str(info.get("modelo_vencedor", "—"))
        score   = float(info.get("score", 0.0))
        mercado = str(info.get("tendencia_mercado", ""))

        card_tpl = lambda icon, val, label: f"""
            <div class="card">
                <div class="icon">{icon}</div>
                <div class="value">{val}</div>
                <span class="label">{label}</span>
            </div>
        """

        col1.markdown(card_tpl("💰", brl(salario), "Salário Médio"), unsafe_allow_html=True)
        col2.markdown(card_tpl("🧠", modelo, "Modelo"), unsafe_allow_html=True)
        col3.markdown(card_tpl("📊", f"{score:.3f}".replace(".", ","), "Score"), unsafe_allow_html=True)
        col4.markdown(card_tpl("📈", mercado if mercado else "N/A", "Mercado"), unsafe_allow_html=True)

        st.write("")

        # PROJEÇÕES 
        st.markdown("#### 📊 Projeção Salarial: +5, +10, +15, +20 anos")

        anos = ["+5 anos", "+10 anos", "+15 anos", "+20 anos"]
        vals_raw = [
            float(info.get("previsao_5", 0.0)),
            float(info.get("previsao_10", 0.0)),
            float(info.get("previsao_15", 0.0)),
            float(info.get("previsao_20", 0.0))
        ]

        vals_formatados = [brl(v) for v in vals_raw]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=anos, 
            y=vals_raw, 
            mode="lines+markers",
            marker=dict(size=10), 
            line=dict(width=3, color="#7b2ff7")
        ))

        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(t=25, r=20, l=40, b=10),
            xaxis_title="Horizonte",
            yaxis_title="Salário (R$)"
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='footer'>Jobin Analytics © 2025</div>", unsafe_allow_html=True)

