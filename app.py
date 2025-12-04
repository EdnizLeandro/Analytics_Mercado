import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# =============================
# CONFIGURAÇÃO DO LAYOUT
# =============================
st.set_page_config(
    page_title="Dashboard Profissões - Salários & Tendências",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================
# CSS GLOBAL COM TEXTOS PRETOS
# E BRANCOS APENAS NOS LOCAIS SOLICITADOS
# =============================
st.markdown("""
<style>
/* Tudo preto por padrão */
* {
    color: black !important;
}

/* TÍTULO PRINCIPAL E INTRODUÇÃO — BRANCO */
#titulo_principal h1,
#titulo_principal p {
    color: white !important;
}

/* Label do input (Digite parte do nome...) — BRANCO */
label[for="Digite parte do nome da profissão:"] {
    color: white !important;
}

/* Texto "Foram encontrados..." — BRANCO */
.resultados-encontrados {
    color: white !important;
}

/* Label "Selecione o CBO" — BRANCO */
.cbo-label {
    color: white !important;
}

/* Profissão selecionada — BRANCO */
.profissao-titulo {
    color: white !important;
}

/* Título do gráfico — BRANCO */
.projecao-titulo {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# =============================
# TÍTULO + INTRODUÇÃO (BRANCOS)
# =============================
st.markdown("""
<div id="titulo_principal">
    <h1>🟣 Previsão Inteligente do Mercado de Trabalho (Jobin + Novo CAGED)</h1>
    <p>
    Encontre sua profissão, descubra <strong>tendências reais do mercado</strong>, veja valores de salário no futuro<br>
    e receba <strong>dicas práticas para se destacar</strong>.<br><br>
    Baseado em dados oficiais do <strong>Novo CAGED</strong>.
    </p>
</div>
""", unsafe_allow_html=True)


# =============================
# CARREGAR OS DADOS
# =============================
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("cache_Jobin1.csv")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

df = carregar_dados()


# =============================
# COMPONENTE: TERMÔMETRO DO MERCADO
# =============================
def mostrar_termometro(estado):
    estados = {
        "alta": ("🟢 Mercado em Alta", "#22c55e", "O setor está crescendo e abrindo oportunidades."),
        "baixa": ("🔴 Mercado em Baixa", "#ef4444", "As vagas diminuíram, mas ainda há chances."),
        "estavel": ("🟡 Mercado Estável", "#eab308", "Poucas mudanças — preparação faz diferença."),
        "recuperacao": ("🟣 Mercado em Recuperação", "#a855f7", "O mercado está voltando a crescer."),
        "volatil": ("🔥 Mercado Volátil", "#fb923c", "O mercado está instável — fique de olho."),
    }

    estado_key = None
    for k in estados:
        if k in estado.lower():
            estado_key = k
            break
    
    titulo, cor, texto = estados.get(
        estado_key,
        ("⚪ Tendência Indefinida", "#9ca3af", "Ainda não há dados claros suficientes.")
    )

    st.markdown(f"""
    <div style="
        background:white;
        border-radius:12px;
        padding:1.3em;
        border:3px solid {cor};
        margin-top:1.5em;
        color:black !important;
    ">
        <h3 style='margin:0;'>{titulo}</h3>
        <p style='margin-top:.5em;'>{texto}</p>
    </div>
    """, unsafe_allow_html=True)


# =============================
# COMPONENTE: DICAS PARA JOVENS
# =============================
def dicas_para_jovens(profissao, tendencia):
    profissão = profissao.lower()

    if "pintor" in profissão:
        return "Monte um portfólio com fotos reais. Pequenos serviços no bairro aumentam sua reputação."
    if "analista" in profissão or "tecnologia" in profissão:
        return "Crie pequenos projetos e coloque no GitHub — isso te destaca muito."
    if "enfermeiro" in profissão or "cuidador" in profissão:
        return "Cursos de certificação aumentam suas chances de contratação."
    if "assistente" in profissão or "auxiliar" in profissão:
        return "Cursos curtos aumentam seu salário de entrada."
    if "motorista" in profissão:
        return "Documentação e comunicação aumentam sua renda."

    if "alta" in tendencia.lower():
        return "Aproveite o momento: candidaturas rápidas aumentam as chances."
    elif "baixa" in tendencia.lower():
        return "Use o período para se qualificar — isso te destaca."
    else:
        return "O mercado pode mudar rápido — fique atento."


# =============================
# MÉTRICAS ESTILIZADAS (PRETAS)
# =============================
def metric_card(titulo, valor, cor="#7c3aed", icone="📌"):
    st.markdown(f"""
    <div style="
        background:white;
        padding:1em;
        border-radius:12px;
        border-left:6px solid {cor};
        margin-bottom:1em;
        color:black !important;
    ">
        <h4 style="margin:0;">{icone} {titulo}</h4>
        <p style="font-size:1.3em;margin-top:.3em;"><b>{valor}</b></p>
    </div>
    """, unsafe_allow_html=True)


# =============================
# BUSCA E FILTRO
# =============================
if df is not None:

    termo = st.text_input(
        "Digite parte do nome da profissão:",
        placeholder="Exemplo: pintor"
    )

    resultado_filtro = pd.DataFrame()
    cbo_selecionado = None

    if termo:
        resultado_filtro = df[df['descricao'].str.contains(termo, case=False, na=False)]
        
        if resultado_filtro.empty:
            st.warning("Nenhuma profissão encontrada.")
        else:
            st.markdown(
                f"<p class='resultados-encontrados'>Foram encontrados {resultado_filtro.shape[0]} resultados:</p>",
                unsafe_allow_html=True
            )

            nomes_cbos = [
                f"{row['codigo']} - {row['descricao']}" 
                for _, row in resultado_filtro.iterrows()
            ]

            st.markdown("<p class='cbo-label'>Selecione o CBO:</p>", unsafe_allow_html=True)

            cbo_str = st.selectbox("", options=nomes_cbos)

            if cbo_str:
                cbo_selecionado = int(cbo_str.split(" - ")[0])


    # =============================
    # EXIBIÇÃO DOS RESULTADOS
    # =============================
    if cbo_selecionado:
        info = resultado_filtro[resultado_filtro['codigo'] == cbo_selecionado].iloc[0]

        st.markdown(
            f"<h3 class='profissao-titulo'>👤 Profissão: {info['descricao']} (CBO {info['codigo']})</h3>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            metric_card("Salário Médio Atual", f"R$ {float(info['salario_medio_atual']):.2f}", "#7c3aed", "💰")
            metric_card("Modelo da Previsão", info['modelo_vencedor'], "#9333ea", "🧠")

        with col2:
            metric_card("Confiabilidade do Modelo", f"{float(info['score']):.4f}", "#7c3aed", "📊")
            metric_card("Tendência Salarial", info['tendencia_salarial'], "#a855f7", "📈")


        mostrar_termometro(info['tendencia_mercado'])

        st.markdown("<h3 class='projecao-titulo'>📈 Projeção Salarial (5/10/15/20 anos)</h3>", unsafe_allow_html=True)

        anos_futuro = ["+5 anos", "+10 anos", "+15 anos", "+20 anos"]
        salarios_futuro = [
            float(info['previsao_5']),
            float(info['previsao_10']),
            float(info['previsao_15']),
            float(info['previsao_20'])
        ]

        fig = go.Figure(
            go.Scatter(
                x=anos_futuro,
                y=salarios_futuro,
                mode='lines+markers',
                line=dict(color='black'),
                marker=dict(size=10, color='black')
            )
        )
        fig.update_layout(
            title=f"Salário Previsto para {info['descricao']}",
            xaxis_title="Horizonte",
            yaxis_title="Salário (R$)",
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)


        st.markdown("### 💡 Dicas para você")
        st.markdown(f"""
        <div style="
            background:#f3e8ff;
            border-left:6px solid #7c3aed;
            padding:1em;
            border-radius:10px;
            color:black !important;
        ">
            <strong>Recomendação:</strong><br>
            {dicas_para_jovens(info['descricao'], info['tendencia_mercado'])}
        </div>
        """, unsafe_allow_html=True)


else:
    st.error("Erro ao carregar 'cache_Jobin1.csv'.")


# =============================
# FOOTER
# =============================
st.markdown(
    "<hr style='margin-top:2em;margin-bottom:1em;'>"
    "<div style='text-align:center;'>"
    "© 2025 Jobin Analytics | Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)
