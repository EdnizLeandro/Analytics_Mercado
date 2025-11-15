import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import math
import os

# ======================================================
# CONFIGURAÇÃO DO APP
# ======================================================
st.set_page_config(page_title="Plataforma Jovem Futuro", layout="wide")

PARQUET_FILE = "dados.parquet"
CBO_FILE = "cbo.xlsx"


# ======================================================
# FUNÇÃO: CARREGAR ARQUIVOS
# ======================================================
@st.cache_resource
def load_data():
    if not os.path.exists(PARQUET_FILE):
        st.error("Arquivo dados.parquet não encontrado.")
        st.stop()

    if not os.path.exists(CBO_FILE):
        st.error("Arquivo cbo.xlsx não encontrado.")
        st.stop()

    df = pd.read_parquet(PARQUET_FILE)
    cbo = pd.read_excel(CBO_FILE)
    cbo.columns = ["codigo", "descricao"]

    return df, cbo


df, df_cbo = load_data()


# ======================================================
# TRATAMENTO DE DATA
# ======================================================
if "competenciadec" in df.columns:
    df["competenciadec"] = pd.to_datetime(df["competenciadec"], errors="coerce")

df = df.dropna(subset=["competenciadec"])


# ======================================================
# INTERFACE — BUSCA POR PROFISSÃO
# ======================================================
st.title("🔎 Previsões do Mercado de Trabalho — Jovem Futuro")

query = st.text_input("Digite nome ou código da profissão:", "")


if query:
    mask = (
        df_cbo["descricao"].str.contains(query, case=False, na=False)
        | df_cbo["codigo"].astype(str).str.contains(query)
    )

    resultados = df_cbo[mask]

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
        st.stop()

    selected_code = st.selectbox("Selecione o CBO:", resultados["codigo"].astype(str))

    if selected_code:

        df_job = df[df["cbo2002ocupacao"].astype(str) == selected_code]

        if df_job.empty:
            st.warning("Não existem registros para este CBO.")
            st.stop()

        st.subheader("📈 Evolução da demanda de trabalho")

        # Gráfico histórico
        fig_hist = px.line(
            df_job.sort_values("competenciadec"),
            x="competenciadec",
            y="saldomovimentacao",
            title="Histórico de contratações",
        )
        st.plotly_chart(fig_hist, use_container_width=True)


        # ======================================================
        # ML — PROPHET → MODELO PRINCIPAL (RMSE)
        # ======================================================

        df_ml = df_job[["competenciadec", "saldomovimentacao"]].rename(
            columns={"competenciadec": "ds", "saldomovimentacao": "y"}
        )

        df_ml = df_ml.dropna()

        if len(df_ml) < 12:
            st.warning("Dados insuficientes para previsão.")
            st.stop()

        model = Prophet()
        model.fit(df_ml)

        future = model.make_future_dataframe(periods=12, freq="M")
        forecast = model.predict(future)

        # Cálculo de RMSE
        df_eval = forecast.tail(len(df_ml))
        rmse = math.sqrt(mean_squared_error(df_ml["y"], df_eval["yhat"]))

        # ======================================================
        # RESULTADOS
        # ======================================================
        st.subheader("🏆 Melhor Modelo Selecionado:")
        st.success("**Prophet — RMSE {:.2f}**".format(rmse))

        # Gráfico da previsão
        st.subheader("🔮 Previsão para os próximos 12 meses")

        fig_forecast = px.line(
            forecast,
            x="ds",
            y="yhat",
            title="Previsão de demanda futura",
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # Exibição da previsão formatada
        st.subheader("📌 Previsão numérica (12 meses)")

        last_12 = forecast[["ds", "yhat"]].tail(12).copy()
        last_12["yhat"] = last_12["yhat"].apply(
            lambda x: f"{x:,.0f}".replace(",", ".")  # formatação brasileira
        )

        st.write(last_12)
