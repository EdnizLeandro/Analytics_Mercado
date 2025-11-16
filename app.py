import streamlit as st
import pandas as pd
import numpy as np
import unicodedata

# -----------------------------
# Função para normalizar textos
# -----------------------------
def normalizar(texto):
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
    df = pd.read_excel("cbo.xlsx")
    df.columns = ["Código", "Descrição"]
    df["Código"] = df["Código"].astype(str).str.strip()
    df["Descrição"] = df["Descrição"].astype(str).str.strip()
    df["Descrição_norm"] = df["Descrição"].apply(normalizar)
    return df

# -----------------------------
# Carregar histórico de salários
# -----------------------------
@st.cache_data
def carregar_historico():
    df = pd.read_parquet("dados.parquet")
    cols_norm = {}
    for col in df.columns:
        col_norm = "".join(
            c for c in unicodedata.normalize("NFD", col.lower())
            if unicodedata.category(c) != "Mn"
        )
        cols_norm[col] = col_norm
    df.columns = cols_norm.values()

    col_cbo = next((col for col in df.columns if "cbo" in col), None)
    if col_cbo is None:
        st.error("Arquivo não contém coluna de CBO.")
        st.stop()

    col_sal = next((col for col in df.columns if "sal" in col), None)
    if col_sal is None:
        st.error("Arquivo não contém coluna salarial.")
        st.stop()

    df[col_cbo] = df[col_cbo].astype(str).str.strip()
    df[col_sal] = pd.to_numeric(df[col_sal], errors="coerce").fillna(0)

    return df, col_cbo, col_sal

# -----------------------------
# Funções auxiliares
# -----------------------------
def buscar_profissoes(df_cbo, texto):
    tnorm = normalizar(texto)
    if texto.isdigit():
        return df_cbo[df_cbo["Código"] == texto]
    return df_cbo[df_cbo["Descrição_norm"].str.contains(tnorm, na=False)]

def prever_salario(sal):
    anos = [5, 10, 15, 20]
    taxa = 0.02
    return {ano: sal * ((1 + taxa) ** ano) for ano in anos}

def tendencia(df, col_cbo, cbo_cod):
    df2 = df[df[col_cbo] == cbo_cod]
    if df2.empty:
        return "Sem dados", {i: 0 for i in [5, 10, 15, 20]}
    saldo = df2.get("saldomovimentacao", pd.Series([0]*len(df2))).mean()
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
# Entrada do usuário
# -----------------------------
entrada = st.text_input("Digite nome ou código da profissão:")

lista_profissoes = []

if entrada.strip():
    resultados = buscar_profissoes(df_cbo, entrada)
    if not resultados.empty:
        lista_profissoes = (
            resultados["Descrição"] + " (" + resultados["Código"] + ")"
        ).tolist()
        st.success(f"{len(resultados)} profissão(ões) encontrada(s).")
    else:
        st.warning("Nenhuma profissão encontrada. Verifique a digitação ou tente outro termo.")

# Define índice inicial do selectbox
# Se houver profissões, seleciona a primeira; caso contrário, vazio
if lista_profissoes:
    escolha = st.selectbox("Selecione a profissão:", [""] + lista_profissoes, index=1)
else:
    escolha = st.selectbox("Selecione a profissão:", [""])

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

        # Criar colunas lado a lado
        col1, col2 = st.columns(2)

        # -----------------------------
        # Coluna 1: Previsão Salarial
        # -----------------------------
        with col1:
            st.subheader("Previsão Salarial")
            prev = prever_salario(salario_atual)
            for ano, val in prev.items():
                st.write(f"{ano} anos → **R$ {val:,.2f}**")

        # -----------------------------
        # Coluna 2: Tendência de Mercado
        # -----------------------------
        with col2:
            st.subheader("Tendência de Mercado")
            status, vagas = tendencia(df_hist, COL_CBO, cbo_codigo)
            st.write(f"Situação histórica: **{status}**")
            for ano, val in vagas.items():
                seta = "↑" if val > 0 else "↓" if val < 0 else "→"
                st.write(f"{ano} anos: {val} ({seta})")

    else:
        st.error("Sem dados suficientes para esta profissão.")
