import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import streamlit as st

# ------------------------------
# CLASSE DE PREDIÇÃO DE MERCADO
# ------------------------------
class MercadoTrabalhoPredictor:
    def __init__(self, parquet_file: str, codigos_filepath: str):
        self.parquet_file = parquet_file
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False

    # Formata moeda brasileira
    def formatar_moeda(self, valor):
        try:
            return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(valor)

    # Carrega dados
    def carregar_dados(self):
        self.df = pd.read_parquet(self.parquet_file)
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ["cbo_codigo", "cbo_descricao"]
        self.df_codigos["cbo_codigo"] = self.df_codigos["cbo_codigo"].astype(str)

        if "salario" in self.df.columns:
            self.df["salario"] = pd.to_numeric(self.df["salario"].astype(str).str.replace(",", "."), errors="coerce")
            mediana = self.df["salario"].median()
            self.df["salario"] = self.df["salario"].fillna(mediana)

        self.cleaned = True

    # Busca profissão por nome ou código
    def buscar_profissao(self, entrada: str):
        if not self.cleaned:
            return pd.DataFrame()
        entrada = entrada.strip()
        if entrada.isdigit():
            return self.df_codigos[self.df_codigos["cbo_codigo"] == entrada]
        mask = self.df_codigos["cbo_descricao"].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    # Relatório de previsão
    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5,10,15,20]):
        df = self.df
        col_cbo = "cbo2002ocupacao"
        col_data = "competenciamov"
        col_salario = "salario"
        col_saldo = "saldomovimentacao"

        # Nome da profissão
        prof_info = self.df_codigos[self.df_codigos["cbo_codigo"] == cbo_codigo]
        titulo = prof_info.iloc[0]["cbo_descricao"] if not prof_info.empty else f"CBO {cbo_codigo}"

        # Container principal
        main_container = st.container()
        with main_container:
            st.header(f"📌 Profissão: {titulo}")

            df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()
            if df_cbo.empty:
                st.warning("Nenhum dado disponível para esta profissão.")
                return

            # ---------------- Demografia ----------------
            with st.expander("Perfil Demográfico"):
                if "idade" in df_cbo.columns:
                    media_idade = pd.to_numeric(df_cbo["idade"], errors="coerce").mean()
                    st.write(f"Idade média: **{media_idade:.1f} anos**")
                if "sexo" in df_cbo.columns:
                    sexo_map = {"1":"Masculino","3":"Feminino"}
                    contagem = df_cbo["sexo"].astype(str).value_counts()
                    txt = ", ".join(f"{sexo_map.get(k,k)}: {(v/len(df_cbo))*100:.1f}%" for k,v in contagem.items())
                    st.write("Distribuição por sexo:", txt)

            # ---------------- Previsão Salarial ----------------
            st.subheader("💰 Previsão Salarial")
            df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors="coerce")
            df_cbo = df_cbo.dropna(subset=[col_data])
            df_cbo["tempo_meses"] = (df_cbo[col_data].dt.year - 2020) * 12 + df_cbo[col_data].dt.month

            salario_atual = df_cbo[col_salario].mean()
            st.write(f"Salário médio atual: **R$ {self.formatar_moeda(salario_atual)}**")

            # Agrupa mensal
            df_mensal = df_cbo.groupby("tempo_meses")[col_salario].mean().reset_index()
            if len(df_mensal) < 2:
                st.info("Sem dados suficientes para fazer previsões.")
                return

            X = df_mensal[["tempo_meses"]]
            y = df_mensal[col_salario]

            # ---------------- Treina modelos ----------------
            modelos = {
                "LinearRegression": LinearRegression(),
                "XGBoost": XGBRegressor(n_estimators=100, objective="reg:squarederror")
            }

            resultados = {}
            for nome, model in modelos.items():
                model.fit(X, y)
                pred = model.predict(X)
                r2 = r2_score(y, pred)
                mae = mean_absolute_error(y, pred)
                resultados[nome] = {"model": model, "r2": r2, "mae": mae}

            # Escolhe melhor modelo pelo R² (maior)
            melhor_nome = max(resultados, key=lambda k: resultados[k]["r2"])
            melhor_modelo = resultados[melhor_nome]["model"]
            r2_melhor = resultados[melhor_nome]["r2"]*100
            mae_melhor = resultados[melhor_nome]["mae"]

            st.write(f"Modelo vencedor: **{melhor_nome} (R²={r2_melhor:.2f}%, MAE={mae_melhor:.2f})**")

            # Previsão futura
            ult_mes = df_mensal["tempo_meses"].max()
            previsoes = []
            for anos in anos_futuros:
                futuro = ult_mes + anos*12
                pred = melhor_modelo.predict([[futuro]])[0]
                variacao = ((pred - salario_atual)/salario_atual)*100
                previsoes.append([anos, f"R$ {self.formatar_moeda(pred)}", f"{variacao:+.1f}%"])

            st.write("### Previsão Salarial Futura:")
            st.table(pd.DataFrame(previsoes, columns=["Ano","Salário Previsto","Variação"]))

            # ---------------- Previsão de Vagas ----------------
            st.subheader("📈 Previsão de Vagas")
            if col_saldo not in df_cbo.columns:
                st.info("Sem dados de movimentação.")
                return

            df_saldo = df_cbo.groupby("tempo_meses")[col_saldo].sum().reset_index()
            if len(df_saldo) < 2:
                st.info("Dados insuficientes para prever vagas.")
                return

            Xs = df_saldo[["tempo_meses"]]
            ys = df_saldo[col_saldo]
            mod_saldo = LinearRegression().fit(Xs, ys)
            ult_mes_s = df_saldo["tempo_meses"].max()

            tendencia = []
            for anos in anos_futuros:
                futuro = ult_mes_s + anos*12
                pred = mod_saldo.predict([[futuro]])[0]

                if pred > 100: status="ALTA DEMANDA"
                elif pred > 50: status="CRESCIMENTO MODERADO"
                elif pred > 0: status="CRESCIMENTO LEVE"
                elif pred > -50: status="RETRAÇÃO LEVE"
                else: status="RETRAÇÃO"

                tendencia.append([anos, f"{pred:+.0f}", status])

            st.table(pd.DataFrame(tendencia, columns=["Ano","Vagas Previstas/mês","Tendência"]))

# ------------------------------
# APLICATIVO STREAMLIT
# ------------------------------
st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho (CAGED / CBO)")

PARQUET_FILE = "dados.parquet"
CBO_FILE = "cbo.xlsx"

with st.spinner("Carregando dados..."):
    app = MercadoTrabalhoPredictor(PARQUET_FILE, CBO_FILE)
    app.carregar_dados()

busca = st.text_input("Digite nome ou código da profissão:")

if busca:
    resultados = app.buscar_profissao(busca)

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        lista = resultados["cbo_codigo"] + " - " + resultados["cbo_descricao"]
        escolha = st.selectbox("Selecione o CBO:", lista)
        cbo_codigo = escolha.split(" - ")[0]

        if st.button("Gerar Relatório Completo"):
            app.relatorio_previsao(cbo_codigo)
