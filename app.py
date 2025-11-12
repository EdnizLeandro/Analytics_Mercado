# ==========================================================
# 📊 Aplicativo Streamlit - Previsão do Mercado de Trabalho
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import openpyxl

# ==============================
# Configurações iniciais
# ==============================
st.set_page_config(page_title="Mercado de Trabalho", layout="wide")
st.title("📈 Análise e Previsão do Mercado de Trabalho")

st.sidebar.header("⚙️ Configurações")

# ==============================
# Carregamento de dados
# ==============================
@st.cache_data(show_spinner=False)
def carregar_dados():
    df = pd.read_parquet("dados.parquet")
    cbo = pd.read_excel("CBO.xlsx")
    return df, cbo

try:
    df, cbo = carregar_dados()
except Exception as e:
    st.error(f"❌ Erro ao carregar arquivos: {e}")
    st.stop()

# ==============================
# Seleção de profissão
# ==============================
if "Descricao" not in cbo.columns:
    st.error("O arquivo CBO.xlsx precisa conter a coluna 'Descricao'.")
    st.stop()

profissoes = sorted(cbo["Descricao"].dropna().unique().tolist())
profissao = st.sidebar.selectbox("Selecione uma profissão", profissoes)

if "profissao" not in df.columns or "data" not in df.columns or "valor" not in df.columns:
    st.error("O arquivo dados.parquet precisa conter as colunas: 'profissao', 'data' e 'valor'.")
    st.stop()

dados_prof = df[df["profissao"] == profissao].copy()
if dados_prof.empty:
    st.warning("Nenhum dado disponível para essa profissão.")
    st.stop()

# ==============================
# Exibição dos dados
# ==============================
st.subheader(f"📊 Histórico - {profissao}")

dados_prof["data"] = pd.to_datetime(dados_prof["data"])
dados_prof = dados_prof.sort_values("data")

fig = px.line(dados_prof, x="data", y="valor", title=f"Evolução histórica - {profissao}")
st.plotly_chart(fig, use_container_width=True)

# ==============================
# Modelagem Prophet
# ==============================
st.subheader("🔮 Previsão com Prophet")

df_prophet = dados_prof[["data", "valor"]].rename(columns={"data": "ds", "valor": "y"})
modelo_prophet = Prophet()
modelo_prophet.fit(df_prophet)

futuro = modelo_prophet.make_future_dataframe(periods=12, freq="M")
previsao = modelo_prophet.predict(futuro)

fig1 = modelo_prophet.plot(previsao)
st.pyplot(fig1, use_container_width=True)

# ==============================
# Modelagem XGBoost
# ==============================
st.subheader("🤖 Previsão com XGBoost")

dados_prof["ano"] = dados_prof["data"].dt.year
dados_prof["mes"] = dados_prof["data"].dt.month
X = dados_prof[["ano", "mes"]]
y = dados_prof["valor"]

modelo_xgb = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    random_state=42,
    max_depth=4
)
modelo_xgb.fit(X, y)

# Prever próximos 12 meses
ultimo_ano, ultimo_mes = dados_prof["ano"].max(), dados_prof["mes"].max()
futuro_xgb = []
for i in range(12):
    ultimo_mes += 1
    if ultimo_mes > 12:
        ultimo_mes = 1
        ultimo_ano += 1
    futuro_xgb.append({"ano": ultimo_ano, "mes": ultimo_mes})

futuro_df = pd.DataFrame(futuro_xgb)
futuro_df["valor_previsto"] = modelo_xgb.predict(futuro_df)
futuro_df["data"] = pd.to_datetime(futuro_df["ano"].astype(str) + "-" + futuro_df["mes"].astype(str) + "-01")

fig2 = px.line(futuro_df, x="data", y="valor_previsto", title="Previsão com XGBoost (12 meses)")
st.plotly_chart(fig2, use_container_width=True)

# ==============================
# Download CSV
# ==============================
csv = futuro_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Baixar previsões (CSV)",
    data=csv,
    file_name=f"previsoes_{profissao}.csv",
    mime="text/csv"
)

st.success("✅ Previsões concluídas!")
