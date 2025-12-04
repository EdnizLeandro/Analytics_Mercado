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

# Estilização global
st.markdown("""
<style>
h1,h2,h3,h4,h5 {
    color: #6d28d9 !important;
}
div.streamlit-expanderHeader {
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)


# =============================
# TÍTULO E INTRODUÇÃO
# =============================
st.title("🟣 Previsão Inteligente do Mercado de Trabalho (Jobin + Novo CAGED)")
st.markdown("""
Encontre sua profissão, descubra **tendências reais do mercado**, veja valores de salário no futuro  
e receba **dicas práticas para se destacar**.

Baseado em dados oficiais do **Novo CAGED**.
""")

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
        "baixa": ("🔴 Mercado em Baixa", "#ef4444", "As vagas diminuíram, mas ainda há chances para quem se destaca."),
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
    ">
        <h3 style='margin:0;'>{titulo}</h3>
        <p style='margin-top:.5em;color:#444;'>{texto}</p>
    </div>
    """, unsafe_allow_html=True)


# =============================
# COMPONENTE: DICAS PARA JOVENS
# =============================
def dicas_para_jovens(profissao, tendencia):
    profissao = profissao.lower()

    if "pintor" in profissao:
        return "Monte um portfólio com fotos reais. Pequenos serviços no bairro ajudam muito a ganhar reputação."
    if "analista" in profissao or "tecnologia" in profissao:
        return "Crie pequenos projetos online. Um GitHub organizado te coloca na frente da concorrência."
    if "enfermeiro" in profissao or "cuidador" in profissao:
        return "Cursos de certificação fazem diferença imediata na contratação."
    if "auxiliar" in profissao or "assistente" in profissao:
        return "Mostre disposição para aprender rápido. Cursos curtos aumentam seu valor."
    if "motorista" in profissao:
        return "Documentação e comunicação com clientes aumentam nota e salário."

    # fallback baseado na tendência
    if "alta" in tendencia.lower():
        return "Aproveite o momento — candidaturas rápidas aumentam suas chances."
    elif "baixa" in tendencia.lower():
        return "Período ideal para se qualificar e subir de nível."
    else:
        return "Fique atento — o mercado pode virar a qualquer momento."


# =============================
# COMPONENTE: MÉTRICAS ESTILIZADAS
# =============================
def metric_card(titulo, valor, cor="#7c3aed", icone="📌"):
    st.markdown(f"""
    <div style="
        background:white;
        padding:1em;
        border-radius:12px;
        border-left:6px solid {cor};
        margin-bottom:1em;
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

    cbo_selecionado = None
    resultado_filtro = pd.DataFrame()

    if termo:
        resultado_filtro = df[df['descricao'].str.contains(termo, case=False, na=False)]
        if resultado_filtro.empty:
            st.warning("Nenhuma profissão encontrada para o termo digitado. Tente outro termo.")
        else:
            st.write(f"**Foram encontrados {resultado_filtro.shape[0]} resultados para:** '{termo}'")

            nomes_cbos = [
                f"{row['codigo']} - {row['descricao']}" 
                for _, row in resultado_filtro.iterrows()
            ]

            cbo_str = st.selectbox(
                "Selecione o CBO:",
                options=nomes_cbos,
                format_func=lambda x: x
            )

            if cbo_str:
                cbo_selecionado = int(cbo_str.split(' - ')[0])

    elif termo == "":
        st.info("Digite parte do nome da profissão para iniciar a busca.")


    # =============================
    # EXIBIÇÃO DOS RESULTADOS
    # =============================
    if cbo_selecionado:
        info = resultado_filtro[resultado_filtro['codigo'] == cbo_selecionado].iloc[0]

        st.subheader(f"👤 Profissão: {info['descricao']} (CBO {info['codigo']})")

        col1, col2 = st.columns(2)
        with col1:
            metric_card("Salário Médio Atual", f"R$ {float(info['salario_medio_atual']):.2f}", "#7c3aed", "💰")
            metric_card("Modelo da Previsão", info['modelo_vencedor'], "#9333ea", "🧠")

        with col2:
            metric_card("Confiabilidade do Modelo", f"{float(info['score']):.4f}", "#7c3aed", "📊")
            metric_card("Tendência Salarial", info['tendencia_salarial'], "#a855f7", "📈")


        # ---- Termômetro ----
        mostrar_termometro(info['tendencia_mercado'])


        # ---- Gráfico ----
        st.markdown("### 📈 Projeção Salarial (5/10/15/20 anos)")

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
                line=dict(color='royalblue'),
                marker=dict(size=10)
            )
        )
        fig.update_layout(
            title=f"Salário Previsto para {info['descricao']}",
            xaxis_title="Horizonte",
            yaxis_title="Salário (R$)",
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)


        # ---- Dicas para jovens ----
        st.markdown("### 💡 Dicas para você")
        st.success(dicas_para_jovens(info['descricao'], info['tendencia_mercado']))


else:
    st.error("Erro ao carregar dados. Verifique o arquivo 'cache_Jobin1.csv'.")


# =============================
# FOOTER
# =============================
st.markdown(
    "<hr style='margin-top:2em;margin-bottom:1em;'>"
    "<div style='text-align:center; color:grey;'>"
    "© 2025 Jobin Analytics | Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)
