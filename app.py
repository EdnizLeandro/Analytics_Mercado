import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==========================================
# Classe principal
# ==========================================
class MercadoTrabalhoPredictor:
    def __init__(self, csv_paths, codigos_path):
        self.csv_paths = csv_paths
        self.codigos_path = codigos_path
        self.df = None
        self.df_codigos = None
        self.cleaned = False
        self.coluna_cbo = None
        self.coluna_data = None
        self.coluna_salario = None

    def formatar_moeda(self, valor):
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def carregar_dados(self):
        # Lê e concatena todos os CSVs presentes na pasta
        dfs = [pd.read_csv(path) for path in self.csv_paths]
        self.df = pd.concat(dfs, ignore_index=True)
        self.df_codigos = pd.read_excel(self.codigos_path)
        self.df_codigos.columns = ['cbo_codigo', 'cbo_descricao']
        self.df_codigos['cbo_codigo'] = self.df_codigos['cbo_codigo'].astype(str)

    def limpar_dados(self):
        obj_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in obj_cols:
            self.df[col] = self.df[col].astype(str)
        for col in self.df.select_dtypes(include=['number']).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.cleaned = True
        self._identificar_colunas()

    def _identificar_colunas(self):
        for col in self.df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if 'cbo' in col_lower and 'ocupa' in col_lower:
                self.coluna_cbo = col
            if 'competencia' in col_lower and 'mov' in col_lower:
                self.coluna_data = col
            if 'salario' in col_lower and 'fixo' in col_lower:
                self.coluna_salario = col

    def buscar_profissao(self, entrada):
        if not self.cleaned:
            return pd.DataFrame()
        if entrada.isdigit():
            return self.df_codigos[self.df_codigos['cbo_codigo'] == entrada]
        mask = self.df_codigos['cbo_descricao'].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    def prever_mercado(self, cbo_codigo):
        if not self.cleaned:
            return None, None
        df_cbo = self.df[self.df[self.coluna_cbo].astype(str) == str(cbo_codigo)].copy()
        if df_cbo.empty:
            return None, None
        df_cbo[self.coluna_data] = pd.to_datetime(df_cbo[self.coluna_data], errors='coerce')
        df_cbo = df_cbo.dropna(subset=[self.coluna_data])
        df_cbo['tempo_meses'] = ((df_cbo[self.coluna_data].dt.year - 2020) * 12 +
                                  df_cbo[self.coluna_data].dt.month)
        df_mensal = df_cbo.groupby('tempo_meses')[self.coluna_salario].mean().reset_index()
        X = df_mensal[['tempo_meses']]
        y = df_mensal[self.coluna_salario]
        model = LinearRegression()
        model.fit(X, y)
        ult_mes = df_mensal['tempo_meses'].max()
        salario_atual = df_cbo[self.coluna_salario].mean()
        previsoes = []
        for anos in [5, 10, 15, 20]:
            mes_futuro = ult_mes + anos * 12
            pred = model.predict(np.array([[mes_futuro]]))[0]
            previsoes.append((anos, pred, ((pred - salario_atual) / salario_atual) * 100))
        return df_mensal, previsoes

# ==========================================
# Interface Streamlit (SEM UPLOAD, leitura direta dos arquivos)
# ==========================================
st.set_page_config(page_title="Previsão do Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho")
st.markdown("Analise tendências salariais e de emprego com base nos dados do CAGED/CBO.")

# Lista de arquivos esperados, todos na mesma pasta do código
csv_paths = [
    "2020_PE1.csv",
    "2021_PE1.csv",
    "2022_PE1.csv",
    "2023_PE1.csv",
    "2024_PE1.csv",
    "2025_PE1.csv"
]
codigos_path = "cbo.xlsx"

try:
    app = MercadoTrabalhoPredictor(csv_paths, codigos_path)
    app.carregar_dados()
    app.limpar_dados()
    st.success("✅ Dados carregados e preparados!")
    
    busca = st.text_input("🔍 Digite o nome ou código da profissão:")
    if busca:
        resultados = app.buscar_profissao(busca)
        if resultados.empty:
            st.warning("Nenhuma profissão encontrada.")
        else:
            cbo_opcao = st.selectbox(
                "Selecione o CBO:",
                resultados['cbo_codigo'] + " - " + resultados['cbo_descricao']
            )
            cbo_codigo = cbo_opcao.split(" - ")[0]
            if st.button("Gerar Previsão"):
                df_mensal, previsoes = app.prever_mercado(cbo_codigo)
                if df_mensal is None:
                    st.error("Sem dados suficientes para prever.")
                else:
                    st.subheader("📈 Evolução Salarial Média")
                    fig, ax = plt.subplots()
                    ax.plot(df_mensal['tempo_meses'], df_mensal[app.coluna_salario], marker='o')
                    ax.set_xlabel("Tempo (meses desde 2020)")
                    ax.set_ylabel("Salário Médio (R$)")
                    ax.set_title("Tendência Histórica de Salário")
                    st.pyplot(fig)
                    st.subheader("🔮 Projeções Futuras")
                    for anos, valor, variacao in previsoes:
                        st.write(f"**{anos} anos → R$ {app.formatar_moeda(valor)} ({variacao:+.1f}%)**")
except Exception as e:
    st.error("Erro ao carregar/processar arquivos! Verifique se todos existem na pasta do código.")
    st.exception(e)
