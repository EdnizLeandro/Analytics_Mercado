import streamlit as st
import pandas as pd
import numpy as np
import time
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Previsão Salarial", layout="wide")

st.title("Previsão Salarial por Profissão")
st.write("Selecione a profissão e veja as previsões futuras com explicações detalhadas dos gráficos.")

# -------------------------------
# 1. Seleção de profissão
# -------------------------------
profissao = st.selectbox(
    "Selecione a profissão:",
    ["Vendedor Pracista", "Vendedor Interno"]
)

# -------------------------------
# 2. Carregar dados (simulado)
# -------------------------------
@st.cache_data
def carregar_dados(profissao):
    # Simula histórico salarial mensal
    np.random.seed(42)
    datas = pd.date_range("2015-01-01", "2025-01-01", freq="M")
    salarios = np.random.normal(loc=3000, scale=500, size=len(datas))
    df = pd.DataFrame({"data": datas, "salario": salarios})
    return df

df = carregar_dados(profissao)

st.subheader("Histórico Salarial")
st.line_chart(df.rename(columns={"data": "index"}).set_index("index")["salario"])
st.info("Este gráfico mostra o histórico dos salários médios mensais para a profissão selecionada.")

# -------------------------------
# 3. Função de treinamento assíncrona
# -------------------------------
def treinar_modelos(df):
    """
    Treina Prophet e XGBoost e retorna previsões.
    """
    previsoes = {}

    # Prophet
    df_prophet = df.rename(columns={"data": "ds", "salario": "y"})
    modelo_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    modelo_prophet.fit(df_prophet)
    futuro = modelo_prophet.make_future_dataframe(periods=12*2, freq="M")  # 2 anos
    pred_prophet = modelo_prophet.predict(futuro)
    previsoes["prophet"] = pred_prophet[["ds", "yhat"]]

    # XGBoost (simplificado)
    df_xgb = df.copy()
    df_xgb["mes"] = df_xgb["data"].dt.month
    df_xgb["ano"] = df_xgb["data"].dt.year
    X = df_xgb[["ano", "mes"]]
    y = df_xgb["salario"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    modelo_xgb = xgb.XGBRegressor(n_estimators=100)
    modelo_xgb.fit(X_train, y_train)
    # Previsão para os próximos 24 meses
    ult_ano, ult_mes = X["ano"].iloc[-1], X["mes"].iloc[-1]
    futuros = []
    for i in range(1, 25):
        mes = ult_mes + i
        ano = ult_ano + (mes-1)//12
        mes = (mes-1)%12 + 1
        futuros.append([ano, mes])
    futuros = pd.DataFrame(futuros, columns=["ano", "mes"])
    pred_xgb = modelo_xgb.predict(futuros)
    futuros["salario"] = pred_xgb
    previsoes["xgboost"] = futuros

    return previsoes

# -------------------------------
# 4. Treinamento com spinner (assíncrono)
# -------------------------------
st.subheader("Treinamento de Modelos")
with st.spinner("Treinando modelos, isso pode levar alguns segundos..."):
    previsoes = treinar_modelos(df)
st.success("Modelos treinados com sucesso!")

# -------------------------------
# 5. Exibição de previsões
# -------------------------------
st.subheader("Previsões Futuras")

# Prophet
st.write("📈 **Previsão pelo Prophet (tendência + sazonalidade)**")
st.line_chart(previsoes["prophet"].set_index("ds")["yhat"])
st.info("Este gráfico mostra a previsão salarial baseada no Prophet, que captura tendências e padrões sazonais históricas.")

# XGBoost
st.write("📊 **Previsão pelo XGBoost (modelo de regressão)**")
xgb_chart = previsoes["xgboost"].copy()
xgb_chart["data"] = pd.to_datetime(xgb_chart[["ano", "mes"]].assign(day=1))
st.line_chart(xgb_chart.set_index("data")["salario"])
st.info("Este gráfico mostra a previsão salarial usando XGBoost, que tenta aprender padrões complexos nos dados históricos.")

# -------------------------------
# 6. Comparação de modelos
# -------------------------------
st.subheader("Resumo das Previsões")
st.write("Aqui você pode comparar visualmente as previsões dos dois modelos e analisar diferenças.")
st.line_chart(
    pd.concat([
        previsoes["prophet"].set_index("ds")["yhat"].rename("Prophet"),
        xgb_chart.set_index("data")["salario"].rename("XGBoost")
    ], axis=1)
)
st.info("Comparando os dois modelos, você pode ver como Prophet e XGBoost projetam o salário para os próximos meses.")

