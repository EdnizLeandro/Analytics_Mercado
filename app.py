import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import streamlit as st

class MercadoTrabalhoPredictor:
    def __init__(self, parquet_file: str, codigos_filepath: str):
        self.parquet_file = parquet_file
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False

    def formatar_moeda(self, valor):
        try:
            return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(valor)

    def carregar_dados(self):
        self.df = pd.read_parquet(self.parquet_file)
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ["cbo_codigo", "cbo_descricao"]
        self.df_codigos["cbo_codigo"] = self.df_codigos["cbo_codigo"].astype(str)

        if "salario" in self.df.columns:
            self.df["salario"] = pd.to_numeric(
                self.df["salario"].astype(str).str.replace(",", "."),
                errors="coerce"
            )
            mediana = self.df["salario"].median()
            self.df["salario"] = self.df["salario"].fillna(mediana)

        self.cleaned = True

    def buscar_profissao(self, entrada: str):
        if not self.cleaned:
            return pd.DataFrame()

        entrada = entrada.strip()
        if entrada.isdigit():
            return self.df_codigos[self.df_codigos["cbo_codigo"] == entrada]

        mask = self.df_codigos["cbo_descricao"].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5, 10, 15, 20]):
        df = self.df
        col_cbo = "cbo2002ocupacao"
        col_data = "competenciamov"
        col_salario = "salario"
        col_saldo = "saldomovimentacao"

        prof_info = self.df_codigos[self.df_codigos["cbo_codigo"] == cbo_codigo]
        titulo = prof_info.iloc[0]["cbo_descricao"] if not prof_info.empty else f"CBO {cbo_codigo}"

        # Placeholder para saída em estilo console
        console_output = []
        console_output.append(f"Profissão: {titulo}\n")

        df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()
        if df_cbo.empty:
            console_output.append("Nenhum dado disponível para esta profissão.")
            return "\n".join(console_output)

        # Salário
        df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors="coerce")
        df_cbo = df_cbo.dropna(subset=[col_data])
        df_cbo["tempo_meses"] = (df_cbo[col_data].dt.year - 2020) * 12 + df_cbo[col_data].dt.month
        salario_atual = df_cbo[col_salario].mean()
        console_output.append(f"Salário médio atual: R$ {self.formatar_moeda(salario_atual)}\n")

        # Preparar dados para modelos
        df_mensal = df_cbo.groupby("tempo_meses")[col_salario].mean().reset_index()
        if len(df_mensal) < 2:
            console_output.append("Sem dados suficientes para fazer previsões salariais.")
            return "\n".join(console_output)

        X = df_mensal[["tempo_meses"]]
        y = df_mensal[col_salario]

        # Treinar LinearRegression
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred_lr = lr.predict(X)
        r2_lr = r2_score(y, y_pred_lr)
        mae_lr = mean_absolute_error(y, y_pred_lr)

        # Treinar XGBoost
        xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgb.fit(X, y)
        y_pred_xgb = xgb.predict(X)
        r2_xgb = r2_score(y, y_pred_xgb)
        mae_xgb = mean_absolute_error(y, y_pred_xgb)

        # Escolher melhor modelo pelo R²
        if r2_xgb >= r2_lr:
            modelo_vencedor = xgb
            modelo_nome = "XGBoost"
            r2 = r2_xgb
            mae = mae_xgb
        else:
            modelo_vencedor = lr
            modelo_nome = "Linear Regression"
            r2 = r2_lr
            mae = mae_lr

        console_output.append(f"Modelo vencedor: {modelo_nome} (R²={r2*100:.2f}%, MAE={mae:.2f})\n")
        console_output.append("Previsão salarial futura do melhor modelo:")

        ult_mes = df_mensal["tempo_meses"].max()
        for anos in anos_futuros:
            futuro = ult_mes + anos*12
            pred = modelo_vencedor.predict([[futuro]])[0]
            console_output.append(f"  {anos} anos → R$ {self.formatar_moeda(pred)}")

        console_output.append("* Tendência de crescimento do salário no longo prazo.\n")
        console_output.append("="*70)
        console_output.append("TENDÊNCIA DE MERCADO (Projeção de demanda para a profissão):")
        console_output.append("="*70)

        if col_saldo not in df_cbo.columns:
            console_output.append("Sem dados de movimentação.")
            return "\n".join(console_output)

        df_saldo = df_cbo.groupby("tempo_meses")[col_saldo].sum().reset_index()
        if len(df_saldo) < 2:
            console_output.append("Dados insuficientes para prever vagas.")
            return "\n".join(console_output)

        Xs = df_saldo[["tempo_meses"]]
        ys = df_saldo[col_saldo]
        mod_saldo = LinearRegression().fit(Xs, ys)
        ult_mes_s = df_saldo["tempo_meses"].max()

        # Situação histórica
        ultima_situacao = ys.iloc[-1]
        if ultima_situacao > 100:
            situacao = "ALTA DEMANDA"
        elif ultima_situacao > 50:
            situacao = "CRESCIMENTO MODERADO"
        elif ultima_situacao > 0:
            situacao = "CRESCIMENTO LEVE"
        elif ultima_situacao > -50:
            situacao = "RETRAÇÃO LEVE"
        else:
            situacao = "RETRAÇÃO"

        console_output.append(f"Situação histórica recente: {situacao}\n")
        console_output.append("Projeção de saldo de vagas (admissões - desligamentos):")
        for anos in anos_futuros:
            futuro = ult_mes_s + anos*12
            pred = mod_saldo.predict([[futuro]])[0]
            if pred > 0:
                seta = "↑"
            elif pred < 0:
                seta = "↓"
            else:
                seta = "→"
            console_output.append(f"  {anos} anos: {int(pred)} ({seta})")

        return "\n".join(console_output)

# -------------------- Streamlit --------------------
st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho (CAGED / CBO)")

PARQUET_FILE = "dados.parquet"
CBO_FILE = "cbo.xlsx"

with st.spinner("Carregando dados..."):
    app = MercadoTrabalhoPredictor(PARQUET_FILE, CBO_FILE)
    app.carregar_dados()

busca = st.text_input("Digite nome ou código da profissão:")
saida_placeholder = st.empty()  # placeholder para saída

if busca:
    resultados = app.buscar_profissao(busca)
    if resultados.empty:
        saida_placeholder.text("Nenhuma profissão encontrada.")
    else:
        lista = resultados["cbo_codigo"] + " - " + resultados["cbo_descricao"]
        escolha = st.selectbox("Selecione o CBO:", lista)
        cbo_codigo = escolha.split(" - ")[0]

        if st.button("Gerar Relatório Completo"):
            saida = app.relatorio_previsao(cbo_codigo)
            saida_placeholder.text(saida)
