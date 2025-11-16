import streamlit as st
import pandas as pd
import numpy as np
import unicodedata

# -----------------------------
# Função para normalizar textos
# -----------------------------
def normalizar(texto):
    """
    Remove acentos, transforma em minúsculas e remove espaços extras.
    Se o valor não for string, retorna vazio.
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

# -----------------------------
# Carregar dados do CBO
# -----------------------------
@st.cache_data
def carregar_dados_cbo():
    """
    Lê arquivo Excel com códigos e descrições de profissões.
    Adiciona coluna normalizada para facilitar buscas.
    """
    df = pd.read_excel("cbo.xlsx")
    df.columns = ["Código", "Descrição"]

    # Garantir que colunas são strings limpas
    df["Código"] = df["Código"].astype(str).str.strip()
    df["Descrição"] = df["Descrição"].astype(str).str.strip()
    
    # Coluna normalizada para busca por texto
    df["Descrição_norm"] = df["Descrição"].apply(normalizar)
    return df

# -----------------------------
# Carregar histórico de salários
# -----------------------------
@st.cache_data
def carregar_historico():
    """
    Lê arquivo Parquet com histórico de salários e movimentações.
    Detecta automaticamente colunas de CBO e salário.
    """
    df = pd.read_parquet("dados.parquet")

    # Normalizar nomes das colunas (remove acentos e deixa minúsculo)
    cols_norm = {}
    for col in df.columns:
        col_norm = "".join(
            c for c in unicodedata.normalize("NFD", col.lower())
            if unicodedata.category(c) != "Mn"
        )
        cols_norm[col] = col_norm
    df.columns = cols_norm.values()

    # Detectar coluna CBO
    col_cbo = next((col for col in df.columns if "cbo" in col), None)
    if col_cbo is None:
        st.error("Arquivo não contém coluna de CBO.")
        st.stop()

    # Detectar coluna salarial
    col_sal = next((col for col in df.columns if "sal" in col), None)
    if col_sal is None:
        st.error("Arquivo não contém coluna salarial.")
        st.stop()

    # Garantir tipos corretos
    df[col_cbo] = df[col_cbo].astype(str).str.strip()
    df[col_sal] = pd.to_numeric(df[col_sal], errors="coerce").fillna(0)

    return df, col_cbo, col_sal

# -----------------------------
# Função para buscar profissões
# -----------------------------
def buscar_profissoes(df_cbo, texto):
    """
    Busca profissões por código ou descrição.
    Retorna DataFrame filtrado.
    """
    tnorm = normalizar(texto)
    if texto.isdigit():
        return df_cbo[df_cbo["Código"] == texto]
    return df_cbo[df_cbo["Descrição_norm"].str.contains(tnorm, na=False)]

# -----------------------------
# Função de previsão salarial
# -----------------------------
def prever_salario(sal):
    """
    Previsão de salário para 5, 10, 15 e 20 anos
    considerando crescimento anual de 2%.
    """
    anos = [5, 10, 15, 20]
    taxa = 0.02
    return {ano: sal * ((1 + taxa) ** ano) for ano in anos}

# -----------------------------
# Função de tendência de mercado
# -----------------------------
def tendencia(df, col_cbo, cbo_cod):
    """
    Calcula tendência de crescimento ou queda com base na
    média da coluna 'saldomovimentacao'.
    """
    df2 = df[df[col_cbo] == cbo_cod]
    if df2.empty:
        return "Sem dados", {i: 0 for i in [5, 10, 15, 20]}
    
    # Usar 'saldomovimentacao' se existir, senão 0
    saldo = df2.get("saldomovimentacao", pd.Series([0]*len(df2))).mean()

    # Definir status baseado no saldo médio
    if saldo > 10:
        status = "CRESCIMENTO ACELERADO"
    elif saldo > 0:
        status = "CRESCIMENTO LEVE"
    elif saldo < -10:
        status = "QUEDA ACELERADA"
    elif saldo < 0:
        status = "QUEDA LEVE"
    else:
        status = "ESTÁVEL"

    return status, {i: int(saldo) for i in [5, 10, 15, 20]}

# -----------------------------
# Interface Streamlit
# -----------------------------
st.set_page_config(page_title="Mercado de Trabalho", layout="wide")
st.title("Previsão do Mercado de Trabalho (Novo CAGED)")

# Carregar dados
df_cbo = carregar_dados_cbo()
df_hist, COL_CBO, COL_SALARIO = carregar_historico()

# -----------------------------
# Widgets de entrada
# -----------------------------
entrada = st.text_input("Digite nome ou código da profissão:")

lista_profissoes = []
if entrada.strip():
    resultados = buscar_profissoes(df_cbo, entrada)
    if not resultados.empty:
        lista_profissoes = (
            resultados["Descrição"] + " (" + resultados["Código"] + ")"
        ).tolist()

escolha = st.selectbox("Selecione a profissão:", [""] + lista_profissoes)

# -----------------------------
# Mostrar resultados
# -----------------------------
if escolha != "":
    cbo_codigo = escolha.split("(")[-1].replace(")", "").strip()
    descricao = escolha.split("(")[0].strip()

    st.header(f"Profissão: {descricao}")

    dados_prof = df_hist[df_hist[COL_CBO] == cbo_codigo]

    if not dados_prof.empty:
        salario_atual = dados_prof[COL_SALARIO].mean()
        st.subheader("Salário Médio Atual")
        st.write(f"R$ {salario_atual:,.2f}")

        st.subheader("Previsão Salarial")
        prev = prever_salario(salario_atual)
        for ano, val in prev.items():
            st.write(f"{ano} anos → **R$ {val:,.2f}**")

        st.subheader("Tendência de Mercado")
        status, vagas = tendencia(df_hist, COL_CBO, cbo_codigo)
        st.write(f"Situação histórica: **{status}**")

        for ano, val in vagas.items():
            seta = "↑" if val > 0 else "↓" if val < 0 else "→"
            st.write(f"{ano} anos: {val} ({seta})")
    else:
        st.error("Sem dados suficientes para esta profissão.")
