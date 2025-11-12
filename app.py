import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# ---------- CLASSE PARA ANÁLISE ----------
class MercadoTrabalhoPredictor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaned = False
        self.coluna_cbo = None
        self.coluna_data = None
        self.coluna_salario = None
        self._preparar_dados()

    def formatar_moeda(self, valor):
        """Formata valor para padrão brasileiro: 1.234,56"""
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def _preparar_dados(self):
        # Limpeza básica
        obj_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in obj_cols:
            self.df[col] = self.df[col].astype(str)

        for col in self.df.select_dtypes(include=['number']).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.cleaned = True

        # Identificação automática de colunas
        for col in self.df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if 'cbo' in col_lower and 'ocupa' in col_lower:
                self.coluna_cbo = col
            if 'competencia' in col_lower or 'data' in col_lower or 'mov' in col_lower:
                self.coluna_data = col
            if 'salario' in col_lower and 'fixo' in col_lower:
                self.coluna_salario = col

    def buscar_profissao(self, entrada: str) -> pd.DataFrame:
        if not self.cleaned:
            return pd.DataFrame()
        if entrada.isdigit():
            resultados = self.df[self.df[self.coluna_cbo].astype(str) == entrada]
        else:
            resultados = self.df[self.df[self.coluna_cbo].str.contains(entrada, case=False, na=False)]
        return resultados

    def prever_mercado(self, df_cbo: pd.DataFrame, anos_futuros=[5, 10, 15, 20]):
        if df_cbo.empty:
            st.warning("Nenhum dado disponível para análise.")
            return

        st.subheader("✅ Análise do Mercado de Trabalho")
        # 1. SALDO DE MOVIMENTAÇÃO
        if 'saldomovimentacao' in df_cbo.columns:
            saldo_total = df_cbo['saldomovimentacao'].sum()
            st.write(f"**Saldo de movimentação (admissões - desligamentos):** {saldo_total:+,.0f}")
            if saldo_total > 0:
                st.success("Mercado em EXPANSÃO")
            elif saldo_total < 0:
                st.error("Mercado em RETRAÇÃO")
            else:
                st.info("Mercado ESTÁVEL")

        # 2. PERFIL DEMOGRÁFICO
        st.subheader("Perfil Demográfico")
        if 'idade' in df_cbo.columns:
            st.write(f"- Idade média: {df_cbo['idade'].mean():.1f} anos")
        if 'sexo' in df_cbo.columns:
            sexo_dist = df_cbo['sexo'].value_counts(normalize=True) * 100
            st.write("- Distribuição por sexo:")
            for sexo, pct in sexo_dist.items():
                sexo_label = "Masculino" if str(sexo) == "1.0" else "Feminino" if str(sexo) == "3.0" else sexo
                st.write(f"    • {sexo_label}: {pct:.1f}%")

        # 3. DISTRIBUIÇÃO GEOGRÁFICA
        if 'uf' in df_cbo.columns:
            st.subheader("Distribuição Geográfica (Top 5 UFs)")
            uf_dist = df_cbo['uf'].value_counts().head(5)
            for uf, count in uf_dist.items():
                st.write(f"- {uf}: {count} registros")

        # 4. PREVISÃO SALARIAL
        if self.coluna_data in df_cbo.columns and self.coluna_salario in df_cbo.columns:
            df_cbo[self.coluna_data] = pd.to_datetime(df_cbo[self.coluna_data], errors='coerce')
            df_cbo = df_cbo.dropna(subset=[self.coluna_data])
            if not df_cbo.empty:
                df_cbo['tempo_meses'] = ((df_cbo[self.coluna_data].dt.year - 2020) * 12 +
                                          df_cbo[self.coluna_data].dt.month)
                df_mensal = df_cbo.groupby('tempo_meses')[self.coluna_salario].mean().reset_index()
                salario_atual = df_cbo[self.coluna_salario].mean()
                st.subheader(f"Salário médio atual: R$ {self.formatar_moeda(salario_atual)}")

                if len(df_mensal) > 1:
                    X = df_mensal[['tempo_meses']]
                    y = df_mensal[self.coluna_salario]
                    model = LinearRegression()
                    model.fit(X, y)
                    ult_mes = df_mensal['tempo_meses'].max()
                    st.subheader("Projeção Salarial")
                    for anos in anos_futuros:
                        mes_futuro = ult_mes + anos * 12
                        pred = model.predict(np.array([[mes_futuro]]))[0]
                        variacao = ((pred - salario_atual) / salario_atual) * 100
                        st.write(f"- {anos} anos → R$ {self.formatar_moeda(max(pred,0))} ({variacao:+.1f}%)")
                else:
                    st.info("Dados insuficientes para previsão salarial.")

# ---------- STREAMLIT APP ----------
st.title("📊 Análise do Mercado de Trabalho")
st.write("Selecione uma profissão ou digite o código CBO para visualizar análises.")

# Carregar dados
filepath = os.path.join(os.path.dirname(__file__), "dados.parquet")
df = pd.read_parquet(filepath)
app = MercadoTrabalhoPredictor(df)

# Seleção de profissão
entrada = st.text_input("Código ou descrição da profissão:")
if entrada:
    resultados = app.buscar_profissao(entrada)
    if resultados.empty:
        st.warning("Nenhum registro encontrado.")
    else:
        st.write(f"**{len(resultados)} registro(s) encontrado(s)**")
        app.prever_mercado(resultados)
