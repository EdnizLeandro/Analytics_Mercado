# ==========================================
# 📊 Jobin – Analytics & Mercado Dashboard
# ==========================================
# Autor: Equipe Jobin
# Descrição: Dashboard interativo em Streamlit
# para análise e previsão do mercado de trabalho por profissão (CBO)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# ==========================
# Funções Utilitárias
# ==========================

def formatar_moeda(valor):
    """Formata valor para padrão brasileiro"""
    if pd.isna(valor):
        return "N/A"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ==========================
# Classe de Processamento
# ==========================

class MercadoTrabalho:
    def __init__(self, df):
        self.df = df
        self._identificar_colunas()

    def _identificar_colunas(self):
        """Identifica automaticamente colunas-chave"""
        for col in self.df.columns:
            col_lower = col.lower().replace(" ", "").replace("_", "")
            if "cbo" in col_lower and "ocupa" in col_lower:
                self.coluna_cbo = col
            if "competencia" in col_lower:
                self.coluna_data = col
            if "salario" in col_lower and ("fixo" in col_lower or "medio" in col_lower):
                self.coluna_salario = col
            if "saldo" in col_lower and "mov" in col_lower:
                self.coluna_saldo = col

    def listar_profissoes(self):
        """Lista CBOs únicos"""
        return sorted(self.df[self.coluna_cbo].dropna().unique())

    def filtrar_cbo(self, cbo_codigo):
        """Filtra registros pelo código CBO"""
        return self.df[self.df[self.coluna_cbo].astype(str) == str(cbo_codigo)].copy()

    def prever_salario(self, df_cbo, anos_futuros=[5, 10, 15, 20]):
        """Previsão linear de salário"""
        df_cbo[self.coluna_data] = pd.to_datetime(df_cbo[self.coluna_data], errors="coerce")
        df_cbo = df_cbo.dropna(subset=[self.coluna_data, self.coluna_salario])
        df_cbo["tempo_meses"] = ((df_cbo[self.coluna_data].dt.year - 2020) * 12 +
                                 df_cbo[self.coluna_data].dt.month)
        df_mensal = df_cbo.groupby("tempo_meses")[self.coluna_salario].mean().reset_index()
        salario_atual = df_cbo[self.coluna_salario].mean()

        if len(df_mensal) < 2:
            return pd.DataFrame({
                "Anos Futuro": anos_futuros,
                "Salário Previsto": [salario_atual] * len(anos_futuros)
            })

        X = df_mensal[["tempo_meses"]]
        y = df_mensal[self.coluna_salario]
        model = LinearRegression()
        model.fit(X, y)
        ult_mes = df_mensal["tempo_meses"].max()

        previsoes = []
        for anos in anos_futuros:
            mes_futuro = ult_mes + anos * 12
            pred = model.predict(np.array([[mes_futuro]]))[0]
            previsoes.append(pred)

        return pd.DataFrame({
            "Anos Futuro": anos_futuros,
            "Salário Previsto": previsoes
        })


# ==========================
# Configuração do App
# ==========================

st.set_page_config(
    page_title="Jobin – Analytics & Mercado",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Jobin – Analytics & Mercado")
st.markdown("""
Plataforma de **inteligência de mercado** para análise e previsão do mercado de trabalho jovem em Recife.  
Explore dados, tendências e previsões salariais por profissão (CBO).
""")

# ==========================
# Carregamento de Dados
# ==========================

@st.cache_data
def carregar_dados():
    try:
        base_path = os.path.dirname(__file__)
        arquivo = os.path.join(base_path, "dados.parquet")

        if not os.path.exists(arquivo):
            raise FileNotFoundError("Arquivo 'dados.parquet' não encontrado no diretório do app.")

        df = pd.read_parquet(arquivo)
        if df.empty:
            raise ValueError("O arquivo 'dados.parquet' está vazio.")
        st.success("✅ Dados carregados com sucesso!")
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar os dados: {e}")
        return None


df = carregar_dados()
if df is None:
    st.stop()

mercado = MercadoTrabalho(df)

# ==========================
# Sidebar de Filtros
# ==========================

with st.sidebar:
    st.header("🎯 Filtros de Análise")
    cbo_lista = mercado.listar_profissoes()

    if not cbo_lista:
        st.error("Nenhuma profissão (CBO) encontrada no dataset.")
        st.stop()

    cbo_codigo = st.selectbox("Selecione o código da profissão (CBO):", cbo_lista)
    anos_futuros = st.multiselect(
        "Períodos de previsão (anos):",
        options=[5, 10, 15, 20],
        default=[5, 10, 15, 20]
    )

# ==========================
# Análise Principal
# ==========================

df_cbo = mercado.filtrar_cbo(cbo_codigo)
if df_cbo.empty:
    st.warning("Nenhum registro encontrado para essa profissão.")
    st.stop()

st.subheader(f"📌 Análise da Profissão: **{cbo_codigo}**")

# ======================
# Métricas Gerais
# ======================

st.markdown("### 📈 Indicadores Gerais")

salario_medio = df_cbo[mercado.coluna_salario].mean()
saldo_total = df_cbo[mercado.coluna_saldo].sum() if hasattr(mercado, 'coluna_saldo') and mercado.coluna_saldo in df_cbo.columns else np.nan

col1, col2, col3 = st.columns(3)
col1.metric("💰 Salário médio", formatar_moeda(salario_medio))
col2.metric("📊 Saldo de movimentação", f"{saldo_total:+,.0f}" if not np.isnan(saldo_total) else "N/A")
col3.metric("🧾 Registros analisados", f"{len(df_cbo):,}")

# ======================
# Abas de Visualização
# ======================

st.markdown("### 📊 Análises e Visualizações")
tab1, tab2, tab3 = st.tabs(["💰 Salário", "📅 Tendência de Vagas", "🌎 Distribuição Geográfica"])

# ---- Aba 1: Previsão Salarial ----
with tab1:
    df_prev = mercado.prever_salario(df_cbo, anos_futuros)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_prev["Anos Futuro"],
        y=df_prev["Salário Previsto"],
        mode="lines+markers",
        name="Previsão Salarial",
        line=dict(color="#4CAF50", width=3)
    ))
    fig.update_layout(
        title="Previsão de Salário Médio por Ano",
        xaxis_title="Anos no Futuro",
        yaxis_title="Salário Previsto (R$)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_prev.style.format({"Salário Previsto": "R$ {:,.2f}"}))

# ---- Aba 2: Tendência de Vagas ----
with tab2:
    if hasattr(mercado, 'coluna_saldo') and mercado.coluna_saldo in df_cbo.columns:
        df_cbo["ano"] = pd.to_datetime(df_cbo[mercado.coluna_data], errors='coerce').dt.year
        df_saldo = df_cbo.groupby("ano")[mercado.coluna_saldo].sum().reset_index()
        fig2 = px.bar(df_saldo, x="ano", y=mercado.coluna_saldo, title="Saldo de Vagas por Ano",
                      labels={mercado.coluna_saldo: "Saldo de Vagas", "ano": "Ano"},
                      color=mercado.coluna_saldo, color_continuous_scale="RdYlGn")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Coluna de movimentação não disponível no dataset.")

# ---- Aba 3: Distribuição Geográfica ----
with tab3:
    if "uf" in df_cbo.columns:
        df_geo = df_cbo["uf"].value_counts().reset_index()
        df_geo.columns = ["UF", "Quantidade"]
        fig3 = px.bar(df_geo, x="UF", y="Quantidade", title="Distribuição de Registros por UF",
                      color="Quantidade", color_continuous_scale="Blues")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Dados de UF não disponíveis no dataset.")
