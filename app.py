import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

class MercadoTrabalhoPredictor:
    def __init__(self, parquet_file: str, codigos_filepath: str):
        self.parquet_file = parquet_file
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False

    # -----------------------
    # Formatação moeda BR
    # -----------------------
    def formatar_moeda(self, valor):
        try:
            return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return valor

    # -----------------------
    # Carregar dados
    # -----------------------
    def carregar_dados(self):
        # Lê o parquet principal
        self.df = pd.read_parquet(self.parquet_file)

        # Lê a tabela CBO
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ['cbo_codigo', 'cbo_descricao']
        self.df_codigos['cbo_codigo'] = self.df_codigos['cbo_codigo'].astype(str)

        # Preencher salários faltantes com a mediana
        if "salario" in self.df.columns:
            salario_mediana = pd.to_numeric(self.df["salario"], errors='coerce').median()
            self.df["salario"] = pd.to_numeric(self.df["salario"], errors="coerce").fillna(salario_mediana)

        self.cleaned = True

    # -----------------------
    # Buscar profissão
    # -----------------------
    def buscar_profissao(self, entrada: str) -> pd.DataFrame:
        if not self.cleaned:
            return pd.DataFrame()

        if entrada.isdigit():
            return self.df_codigos[self.df_codigos['cbo_codigo'] == entrada]

        mask = self.df_codigos['cbo_descricao'].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    # -----------------------
    # Relatório completo
    # -----------------------
    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5, 10, 15, 20]):

        df = self.df.copy()

        col_cbo = "cbo2002ocupacao"
        col_data = "competenciamov"
        col_salario = "salario"
        col_saldo = "saldomovimentacao"

        # -----------------------
        # Título da profissão
        # -----------------------
        prof_info = self.df_codigos[self.df_codigos['cbo_codigo'] == cbo_codigo]
        st.subheader(
            f"Profissão: {prof_info.iloc[0]['cbo_descricao']}"
            if len(prof_info) > 0 else f"CBO: {cbo_codigo}"
        )

        # Filtrar dados para o CBO
        df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()

        if df_cbo.empty:
            st.warning("Nenhum registro encontrado para a profissão selecionada.")
            return

        st.write(f"**Registros processados:** {len(df_cbo):,}")

        # -----------------------
        # Processamento das datas
        -----------------------
        df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors='coerce')
        df_cbo = df_cbo.dropna(subset=[col_data])

        if df_cbo.empty:
            st.warning("Não há dados temporais válidos para previsões.")
            return

        df_cbo["tempo_meses"] = ((df_cbo[col_data].dt.year - 2020) * 12) + df_cbo[col_data].dt.month

        # -----------------------
        # MÉDIA SALARIAL ATUAL
        # -----------------------
        salario_atual = df_cbo[col_salario].mean()
        st.subheader("Previsão Salarial (5, 10, 15, 20 anos)")
        st.write(f"Salário médio atual: **R$ {self.formatar_moeda(salario_atual)}**")

        # -----------------------
        # Regressão para previsão salarial
        # -----------------------
        df_mensal = df_cbo.groupby('tempo_meses')[col_salario].mean().reset_index()

        if len(df_mensal) >= 2:

            X = df_mensal[['tempo_meses']]
            y = df_mensal[col_salario]
            model = LinearRegression().fit(X, y)

            ult_mes = df_mensal['tempo_meses'].max()

            previsoes = []
            for anos in anos_futuros:
                mes_fut = ult_mes + anos * 12
                pred = model.predict([[mes_fut]])[0]
                variacao = ((pred - salario_atual) / salario_atual) * 100

                previsoes.append([
                    anos,
                    f"R$ {self.formatar_moeda(pred)}",
                    f"{variacao:+.1f}%"
                ])

            st.write("### Salários previstos")
            st.table(pd.DataFrame(previsoes, columns=["Anos", "Salário Previsto", "Variação (%)"]))
        else:
            st.info("Não há dados suficientes para previsão salarial.")

        # -----------------------
        # Previsão de saldo (tendência de vagas)
        # -----------------------
        st.subheader("Tendência de Vagas (5, 10, 15, 20 anos)")

        if col_saldo in df_cbo.columns:

            df_saldo = df_cbo.groupby("tempo_meses")[col_saldo].sum().reset_index()

            if len(df_saldo) >= 2:

                Xs = df_saldo[['tempo_meses']]
                ys = df_saldo[col_saldo]

                mod = LinearRegression().fit(Xs, ys)
                ult_mes = df_saldo['tempo_meses'].max()

                tendencia_rows = []

                for anos in anos_futuros:
                    mes_fut = ult_mes + anos * 12
                    pred = mod.predict([[mes_fut]])[0]

                    if pred > 100: status = "ALTA DEMANDA"
                    elif pred > 50: status = "CRESCIMENTO MODERADO"
                    elif pred > 0: status = "CRESCIMENTO LEVE"
                    elif pred > -50: status = "RETRAÇÃO LEVE"
                    elif pred > -100: status = "RETRAÇÃO MODERADA"
                    else: status = "RETRAÇÃO FORTE"

                    tendencia_rows.append([
                        anos,
                        f"{pred:+,.0f}".replace(",", "."),
                        status
                    ])

                st.write("### Tendência futura de vagas")
                st.table(pd.DataFrame(tendencia_rows, columns=["Anos", "Vagas Previstas/mês", "Tendência"]))
            else:
                st.info("Dados insuficientes para previsão de vagas.")

# ----------------------------------------------------------
# STREAMLIT
# ----------------------------------------------------------
st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")

st.title("📊 Previsão do Mercado de Trabalho (CAGED/CBO)")

parquet_file = "dados.parquet"
codigos = "cbo.xlsx"

with st.spinner("Carregando dados..."):
    app = MercadoTrabalhoPredictor(parquet_file, codigos)
    app.carregar_dados()

st.success("Sistema pronto!")

busca = st.text_input("Digite o nome ou código da profissão:")

if busca:
    resultados = app.buscar_profissao(busca)
    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        cbo_opcao = st.selectbox(
            "Selecione:",
            resultados['cbo_codigo'] + " - " + resultados['cbo_descricao']
        )

        cbo_selecionado = cbo_opcao.split(" - ")[0]

        if st.button("Gerar análise completa"):
            app.relatorio_previsao(cbo_selecionado)
