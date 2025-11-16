import streamlit as st
import pandas as pd
from unidecode import unidecode
import numpy as np

# -------------------------------
# Funções de carregamento e busca
# -------------------------------
@st.cache_data
def carregar_dados_cbo(cbo_path="cbo.xlsx"):
    df = pd.read_excel(cbo_path)
    df["Código"] = df["Código"].astype(str).str.strip()
    df["Descrição"] = df["Descrição"].str.strip()
    return df

@st.cache_data
def carregar_dados_historico(dados_path="dados.parquet"):
    df = pd.read_parquet(dados_path)
    df["cbo2002ocupação"] = df["cbo2002ocupação"].astype(str).str.strip()
    df["salário"] = df["salário"].fillna(0)
    return df

def buscar_profissao(df, entrada):
    entrada_limpa = unidecode(entrada.lower().strip())
    if entrada.isdigit():
        resultado = df[df["Código"] == entrada]
    else:
        resultado = df[df["Descrição"].apply(lambda x: entrada_limpa in unidecode(str(x).lower()))]
    return resultado

# -------------------------------
# Função de previsão salarial simples
# -------------------------------
def prever_salario(salario_atual):
    anos = [5, 10, 15, 20]
    crescimento_anual = 0.02  # 2% ao ano
    previsao = [round(salario_atual * ((1 + crescimento_anual) ** ano), 2) for ano in anos]
    return dict(zip(anos, previsao))

# -------------------------------
# Função de tendência de mercado
# -------------------------------
def tendencia_mercado(df_historico, cbo_codigo):
    df = df_historico[df_historico["cbo2002ocupação"] == cbo_codigo].copy()
    if df.empty:
        return "Sem dados suficientes", {5:0,10:0,15:0,20:0}
    # saldo = admissoes - desligamentos, usando coluna 'saldomovimentação'
    df["saldo"] = df["saldomovimentação"]
    anos_projecao = [5, 10, 15, 20]
    saldo_projecao = {ano: int(df["saldo"].mean()) for ano in anos_projecao}
    
    saldo_medio = df["saldo"].mean()
    if saldo_medio > 10:
        situacao = "CRESCIMENTO ACELERADO"
    elif saldo_medio > 0:
        situacao = "CRESCIMENTO LEVE"
    elif saldo_medio < -10:
        situacao = "QUEDA ACELERADA"
    elif saldo_medio < 0:
        situacao = "QUEDA LEVE"
    else:
        situacao = "ESTÁVEL"
    
    return situacao, saldo_projecao

# -------------------------------
# Streamlit Interface
# -------------------------------
st.set_page_config(page_title="Previsão Salarial e Mercado de Trabalho", layout="wide")
st.title("📊 Previsão Mercado de Trabalho (Novo Caged)")

# Carregar dados
df_cbo = carregar_dados_cbo()
df_historico = carregar_dados_historico()

entrada = st.text_input("Digite nome ou código da profissão:")

if entrada:
    resultado = buscar_profissao(df_cbo, entrada)
    
    if resultado.empty:
        st.error("Profissão não encontrada. Digite outro nome ou código.")
    elif len(resultado) > 1:
        st.warning("Encontramos múltiplas opções. Por favor, selecione uma:")
        opcao = st.selectbox("Selecione a profissão:", resultado["Descrição"] + " (" + resultado["Código"] + ")")
        cbo_codigo = resultado[resultado["Descrição"] + " (" + resultado["Código"] + ")" == opcao]["Código"].values[0]
    else:
        cbo_codigo = resultado["Código"].values[0]
    
    # Buscar salário médio no histórico
    df_salario = df_historico[df_historico["cbo2002ocupação"] == cbo_codigo]
    if not df_salario.empty:
        salario_atual = df_salario["salário"].mean()
    else:
        salario_atual = 0
    
    st.subheader(f"Profissão: {resultado.loc[resultado['Código']==cbo_codigo, 'Descrição'].values[0]}")
    st.write(f"Salário médio atual: R$ {salario_atual:,.2f}")
    
    if salario_atual > 0:
        previsao = prever_salario(salario_atual)
        st.markdown("**Previsão salarial futura do melhor modelo:**")
        for ano, valor in previsao.items():
            st.write(f"{ano} anos → R$ {valor:,.2f}")
        st.write("* Tendência de crescimento do salário no longo prazo.")
    
    situacao, saldo_projecao = tendencia_mercado(df_historico, cbo_codigo)
    st.markdown("======================================================================")
    st.markdown("**TENDÊNCIA DE MERCADO (Projeção de demanda para a profissão):**")
    st.markdown("======================================================================")
    st.write(f"Situação histórica recente: {situacao}")
    st.write("Projeção de saldo de vagas (admissões - desligamentos):")
    for ano, saldo in saldo_projecao.items():
        seta = "→" if saldo==0 else ("↑" if saldo>0 else "↓")
        st.write(f"  {ano} anos: {saldo} ({seta})")
