# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================
# Funções utilitárias
# ==========================
def formatar_moeda(valor):
    if pd.isna(valor):
        return "N/A"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ==========================
# Classe de processamento
# ==========================
class MercadoTrabalho:
    def __init__(self, df):
        self.df = df
        self.coluna_cbo = None
        self.coluna_data = None
        self.coluna_salario = None
        self.identificar_colunas()

    def identificar_colunas(self):
        cols = [c.lower().replace("_","").replace(" ","") for c in self.df.columns]
        for i, c in enumerate(cols):
            if "cbo" in c and "ocupa" in c:
                self.coluna_cbo = self.df.columns[i]
            if "competencia" in c or "data" in c:
                self.coluna_data = self.df.columns[i]
            if "salario" in c and "fixo" in c:
                self.coluna_salario = self.df.columns[i]

    def filtrar_cbo(self, cbo_codigo):
        if self.coluna_cbo is None:
            return pd.DataFrame()
        return self.df[self.df[self.coluna_cbo].astype(str) == str(cbo_codigo)].copy()

    def prever_salario(self, df_cbo, anos_futuros=[5,10,15,20]):
        if self.coluna_data is None or self.coluna_salario is None:
            return pd.DataFrame({"Anos Futuro": anos_futuros, "Salário Previsto":[np.nan]*len(anos_futuros)})
        df_cbo[self.coluna_data] = pd.to_datetime(df_cbo[self.coluna_data], errors="coerce")
        df_cbo = df_cbo.dropna(subset=[self.coluna_data])
        if df_cbo.empty:
            return pd.DataFrame({"Anos Futuro": anos_futuros, "Salário Previsto":[np.nan]*len(anos_futuros)})
        df_cbo["tempo_meses"] = ((df_cbo[self.coluna_data].dt.year - 2020) * 12 +
                                 df_cbo[self.coluna_data].dt.month)
        df_mensal = df_cbo.groupby("tempo_meses")[self.coluna_salario].mean().reset_index()
        salario_atual = df_cbo[self.coluna_salario].mean()
        if len(df_mensal)<2:
            return pd.DataFrame({"Anos Futuro": anos_futuros, "Salário Previsto":[salario_atual]*len(anos_futuros)})
        X = df_mensal[["tempo_meses"]]
        y = df_mensal[self.coluna_salario]
        model = LinearRegression()
        model.fit(X,y)
        ult_mes = df_mensal["tempo_meses"].max()
        previsoes = []
        for anos in anos_futuros:
            mes_futuro = ult_mes + anos*12
            pred = model.predict(np.array([[mes_futuro]]))[0]
            previsoes.append(pred)
        return pd.DataFrame({"Anos Futuro": anos_futuros, "Salário Previsto":previsoes})

# ==========================
# Configuração do App
# ==========================
st.set_page_config(page_title="Jobin – Analytics & Mercado", page_icon="📊", layout="wide")
st.title("📊 Jobin – Analytics & Mercado")
st.markdown("Dashboard para análise e previsão do mercado de trabalho jovem.")

# ==========================
# Carregamento de dados
# ==========================
try:
    df = pd.read_parquet("dados.parquet")
    st.success("✅ Dados carregados com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {e}")
    st.stop()

mercado = MercadoTrabalho(df)

# ==========================
# Sidebar
# ==========================
st.sidebar.header("Filtros")
if mercado.coluna_cbo:
    profissoes_disponiveis = sorted(df[mercado.coluna_cbo].astype(str).unique())
    cbo_codigo = st.sidebar.selectbox("Selecione o código CBO:", profissoes_disponiveis)
else:
    st.sidebar.warning("Coluna CBO não encontrada.")
    cbo_codigo = None

anos_futuros = st.sidebar.multiselect("Períodos de previsão (anos):", [5,10,15,20], default=[5,10])

# ==========================
# Filtrar dados
# ==========================
if cbo_codigo:
    df_cbo = mercado.filtrar_cbo(cbo_codigo)
else:
    df_cbo = df.copy()

if df_cbo.empty:
    st.warning("Nenhum registro encontrado para esta seleção.")
    st.stop()

# ==========================
# Indicadores gerais
# ==========================
st.subheader(f"📌 Profissão: {cbo_codigo if cbo_codigo else 'Todos'}")
salario_medio = df_cbo[mercado.coluna_salario].mean() if mercado.coluna_salario else np.nan
saldo_total = df_cbo["saldomovimentacao"].sum() if "saldomovimentacao" in df_cbo.columns else np.nan

col1,col2,col3 = st.columns(3)
col1.metric("Salário médio", formatar_moeda(salario_medio))
col2.metric("Saldo de movimentação", f"{saldo_total:+,.0f}" if not pd.isna(saldo_total) else "N/A")
col3.metric("Registros analisados", f"{len(df_cbo):,}")

# ==========================
# Gráficos
# ==========================
st.markdown("### 📊 Visualizações")
tab1,tab2,tab3 = st.tabs(["💰 Salário","📅 Tendência de Vagas","🌎 Distribuição Geográfica"])

with tab1:
    df_prev = mercado.prever_salario(df_cbo, anos_futuros)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_prev["Anos Futuro"], y=df_prev["Salário Previsto"], mode="lines+markers",
                             name="Previsão Salarial", line=dict(color="#4CAF50", width=3)))
    fig.update_layout(title="Previsão de Salário Médio por Ano", xaxis_title="Anos Futuro", 
                      yaxis_title="Salário Previsto (R$)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if "saldomovimentacao" in df_cbo.columns and mercado.coluna_data:
        df_cbo["ano"] = pd.to_datetime(df_cbo[mercado.coluna_data], errors="coerce").dt.year
        df_saldo = df_cbo.groupby("ano")["saldomovimentacao"].sum().reset_index()
        fig2 = px.bar(df_saldo, x="ano", y="saldomovimentacao", color="saldomovimentacao", color_continuous_scale="RdYlGn",
                      labels={"saldomovimentacao":"Saldo de Vagas","ano":"Ano"}, title="Saldo de Vagas por Ano")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Coluna de movimentação ou data não disponível.")

with tab3:
    if "uf" in df_cbo.columns:
        df_geo = df_cbo["uf"].value_counts().reset_index()
        df_geo.columns=["UF","Quantidade"]
        fig3 = px.choropleth(df_geo, locations="UF", locationmode="ISO-3", color="Quantidade",
                             title="Distribuição Geográfica por UF")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Dados geográficos não disponíveis.")
