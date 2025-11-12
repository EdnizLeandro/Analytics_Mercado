# ==========================================================
# 📊 Aplicativo Streamlit - Mercado de Trabalho (Versão Estável, Aprimorada)
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Mercado de Trabalho", layout="wide")
st.title("📈 Análise e Previsão do Mercado de Trabalho no Brasil")
st.sidebar.header("⚙️ Configurações")

# ==============================
# Função para carregar dados
# ==============================
@st.cache_data(show_spinner=True)
def carregar_dados():
    df = pd.read_parquet("dados.parquet")
    cbo = pd.read_excel("CBO.xlsx")
    return df, cbo

# Carregamento
try:
    df, cbo = carregar_dados()
except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {e}")
    st.stop()

# ==============================
# Validação das colunas
# ==============================
colunas_necessarias = {"profissao", "data", "valor"}
if not colunas_necessarias.issubset(df.columns):
    st.error("O arquivo 'dados.parquet' deve conter as colunas: 'profissao', 'data' e 'valor'.")
    st.stop()
if "Descricao" not in cbo.columns:
    st.error("O arquivo 'CBO.xlsx' deve conter a coluna 'Descricao'.")
    st.stop()

# ==============================
# Filtro de profissão
# ==============================
profissoes = sorted(cbo["Descricao"].dropna().unique().tolist())
prof = st.sidebar.selectbox("Selecione uma profissão:", profissoes)

dados_prof = df[df["profissao"] == prof].copy()
if dados_prof.empty:
    st.warning("Nenhum dado encontrado para essa profissão.")
    st.stop()

# TRATAMENTO DE DADOS
dados_prof["data"] = pd.to_datetime(dados_prof["data"])
dados_prof = dados_prof.sort_values("data").reset_index(drop=True)
dados_prof = dados_prof.dropna(subset=["valor"])

st.subheader(f"📊 Histórico — {prof}")
st.dataframe(dados_prof.tail(10))

# ==============================
# Gráfico da série histórica
# ==============================
fig_hist = px.line(
    dados_prof, x="data", y="valor",
    title=f"Evolução Histórica — {prof.title()}",
    markers=True,
    template="plotly_white",
)
st.plotly_chart(fig_hist, use_container_width=True)

# ==============================
# Treinamento do Modelo XGBoost
# ==============================
st.subheader("🤖 Previsão de Salários com XGBoost")

dados_prof["ano"] = dados_prof["data"].dt.year
dados_prof["mes"] = dados_prof["data"].dt.month

# Explicativas: usando lags para maior robustez
for lag in range(1, 13):
    dados_prof[f"lag_{lag}"] = dados_prof["valor"].shift(lag)
dados_model = dados_prof.dropna().reset_index(drop=True)

# Features e target
X = dados_model[["ano", "mes"] + [f"lag_{lag}" for lag in range(1, 13)]]
y = dados_model["valor"]

modelo = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
modelo.fit(X, y)

# ==============================
# Previsão Futura Multipla (12, 24, 36 meses)
# ==============================
def previsao_futura(last_known, modelo, anos=5):
    preds = []
    lags = last_known[-12:].tolist()
    ano, mes = last_known.name.year, last_known.name.month
    for i in range(anos * 12):
        mes += 1
        if mes > 12:
            mes = 1
            ano += 1
        x_input = [ano, mes] + lags
        pred = modelo.predict(pd.DataFrame([x_input], columns=X.columns))[0]
        preds.append({"data": pd.Timestamp(year=ano, month=mes, day=1), "salario_previsto": pred})
        lags = lags[1:] + [pred]
    return pd.DataFrame(preds)

# Rodar previsão para 5, 10, 15, 20 anos (60, 120, 180, 240 meses)
last_row = dados_model.iloc[-1]
previsoes_longas = previsao_futura(last_row, modelo, anos=20)

anos_futuros = [5, 10, 15, 20]
previsoes_marcos = previsoes_longas.iloc[[12*an-1 for an in anos_futuros]].copy()
previsoes_marcos["Ano"] = anos_futuros

# Mostra tabela com previsões-alvo
st.markdown("### Previsão salarial para os próximos anos")
st.table(previsoes_marcos[["Ano", "salario_previsto"]].rename(
    columns={"Ano": "Ano(s) no Futuro", "salario_previsto": "Salário Previsto"}).assign(
        **{"Salário Previsto": lambda d: d["Salário Previsto"].map(lambda x: f"R$ {x:,.2f}")}))

# Gráfico da previsão longa
fig_prev_long = px.line(
    previsoes_longas, x="data", y="salario_previsto",
    title=f"Previsão Salarial — {prof} (20 anos)",
    template="plotly_white"
)
st.plotly_chart(fig_prev_long, use_container_width=True)

# ==============================
# Avaliação e Download dos resultados
# ==============================
y_pred = modelo.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.write(f"📏 **MAE (erro médio absoluto):** R$ {mae:,.2f}")
st.write(f"📈 **R² (coeficiente de determinação):** {r2:.3f}")

csv = previsoes_longas.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Baixar todas previsões (CSV Futuro 20 anos)",
    data=csv,
    file_name=f"previsoes_{prof.replace(' ', '_')}.csv",
    mime="text/csv"
)

st.success("✅ Previsões geradas com sucesso! Explore os gráficos e resultados.")

# ==============================
# Mensagem interpretativa
# ==============================
# Avaliação de tendência nas previsões longas
dif_p20 = previsoes_marcos["salario_previsto"].iloc[-1] - previsoes_marcos["salario_previsto"].iloc[0]
if dif_p20 > 100:
    msg = "🔼 Tendência projetada de crescimento salarial no longo prazo."
elif dif_p20 < -100:
    msg = "🔽 Tendência projetada de queda salarial no longo prazo."
else:
    msg = "⏹️ Tendência projetada de estabilidade salarial no longo prazo."
st.info(msg)
