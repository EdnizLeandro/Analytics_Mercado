import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

class MercadoTrabalhoPredictor:
    def __init__(self, csv_files: list, codigos_filepath: str):
        self.csv_files = csv_files
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False
        self.coluna_cbo = None
        self.coluna_data = None
        self.coluna_salario = None

    def formatar_moeda(self, valor):
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def carregar_dados(self):
        dfs = [pd.read_csv(path, encoding='utf-8', sep=';', on_bad_lines='skip') for path in self.csv_files]
        self.df = pd.concat(dfs, ignore_index=True)
        self.df_codigos = pd.read_excel(self.codigos_filepath)
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
        possiveis_cbo = [c for c in self.df.columns if "cbo" in c.lower() and "ocup" in c.lower()]
        if not possiveis_cbo:
            possiveis_cbo = [c for c in self.df.columns if "cbo" in c.lower()]
        self.coluna_cbo = possiveis_cbo[0] if possiveis_cbo else None

        possiveis_data = [c for c in self.df.columns if "compet" in c.lower() and "mov" in c.lower()]
        if not possiveis_data:
            possiveis_data = [c for c in self.df.columns if "compet" in c.lower() or "mes" in c.lower() or "data" in c.lower()]
        self.coluna_data = possiveis_data[0] if possiveis_data else None

        salario_candidates = [c for c in self.df.columns if "salário" in c.lower()]
        if not salario_candidates:
            salario_candidates = [c for c in self.df.columns if "salario" in c.lower()]
        self.coluna_salario = salario_candidates[0] if salario_candidates else None

    def buscar_profissao(self, entrada: str) -> pd.DataFrame:
        if not self.cleaned:
            return pd.DataFrame()
        if entrada.isdigit():
            return self.df_codigos[self.df_codigos['cbo_codigo'] == entrada]
        mask = self.df_codigos['cbo_descricao'].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    def relatorio_previsao(self, cbo_codigo: str, anos_futuros=[5, 10, 15, 20]):
        if not self.cleaned or not all([self.coluna_cbo, self.coluna_data, self.coluna_salario]):
            st.warning("Colunas não identificadas.")
            return

        prof_info = self.df_codigos[self.df_codigos['cbo_codigo'] == cbo_codigo]
        if not prof_info.empty:
            st.subheader(f"Profissão selecionada: {prof_info.iloc[0]['cbo_descricao']}")

        df_cbo = self.df[self.df[self.coluna_cbo].astype(str) == cbo_codigo].copy()
        if df_cbo.empty:
            st.warning("Nenhum registro encontrado para a profissão selecionada.")
            return

        st.markdown(f"<b>Registros encontrados:</b> {len(df_cbo):,}", unsafe_allow_html=True)

        # Perfil demográfico
        with st.expander("Perfil Demográfico"):
            if 'idade' in df_cbo.columns:
                idade_media = df_cbo['idade'].mean()
                st.write(f"Idade média: {idade_media:.1f} anos")
            if 'sexo' in df_cbo.columns:
                sexo_dist = df_cbo['sexo'].value_counts()
                sexo_lista = [
                    f"{'Masculino' if str(sex)=='1.0' else 'Feminino' if str(sex)=='3.0' else str(sex)}: {(count/len(df_cbo))*100:.1f}%"
                    for sex,count in sexo_dist.items()
                ]
                st.write("Distribuição por sexo:", ", ".join(sexo_lista))
            if 'graudeinstrucao' in df_cbo.columns:
                escolaridade = df_cbo['graudeinstrucao'].value_counts().head(3)
                escolaridade_map = {
                    '1': 'Analfabeto','2': 'Até 5ª inc. Fundamental','3': '5ª completo Fundamental',
                    '4': '6ª a 9ª Fundamental','5': 'Fundamental completo','6': 'Médio incompleto',
                    '7': 'Médio completo','8': 'Superior incompleto','9': 'Superior completo',
                    '10': 'Mestrado','11': 'Doutorado','80': 'Pós-graduação'
                }
                esc_strings = []
                for nivel,count in escolaridade.items():
                    try:
                        nivel_nome = escolaridade_map.get(str(int(float(nivel))), str(nivel))
                    except:
                        nivel_nome = str(nivel)
                    esc_strings.append(f"{nivel_nome}: {(count/len(df_cbo))*100:.1f}%")
                st.write("Principais níveis:", ", ".join(esc_strings))
            if 'uf' in df_cbo.columns:
                uf_map = {'11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO','21':'MA','22':'PI','23':'CE','24':'RN','25':'PB','26':'PE','27':'AL','28':'SE','29':'BA','31':'MG','32':'ES','33':'RJ','35':'SP','41':'PR','42':'SC','43':'RS','50':'MS','51':'MT','52':'GO','53':'DF'}
                uf_dist = df_cbo['uf'].value_counts().head(5)
                uf_lista = []
                for uf_cod,count in uf_dist.items():
                    try: uf_nome = uf_map.get(str(int(float(uf_cod))),str(uf_cod))
                    except: uf_nome = str(uf_cod)
                    uf_lista.append(f"{uf_nome}: {count:,} ({(count/len(df_cbo))*100:.1f}%)")
                st.write("Principais UF:", ", ".join(uf_lista))

        # Saldo de movimentação
        if 'saldomovimentacao' in df_cbo.columns:
            saldo_total = df_cbo['saldomovimentacao'].sum()
            msg = f"<b>Saldo movimentação (adm-desl):</b> {saldo_total:+,.0f}"
            if saldo_total > 0: msg += " → Mercado em expansão"
            elif saldo_total < 0: msg += " → Mercado em retração"
            else: msg += " → Mercado estável"
            st.markdown(msg, unsafe_allow_html=True)

        # PREPROCESSAMENTO DE SALÁRIO (CORREÇÃO DO ERRO)
        df_cbo[self.coluna_salario] = pd.to_numeric(
            df_cbo[self.coluna_salario]
                .astype(str)
                .str.replace(",", ".")
                .str.replace(" ", ""),
            errors='coerce'
        )
        df_cbo = df_cbo.dropna(subset=[self.coluna_salario])

        # Previsão salarial
        st.subheader("Previsão Salarial")
        df_cbo[self.coluna_data] = pd.to_datetime(df_cbo[self.coluna_data], errors='coerce')
        df_cbo = df_cbo.dropna(subset=[self.coluna_data])
        if df_cbo.empty:
            st.warning("Não há dados temporais válidos.")
            return
        df_cbo['tempo_meses'] = ((df_cbo[self.coluna_data].dt.year - 2020) * 12 +
                                 df_cbo[self.coluna_data].dt.month)
        df_mensal = df_cbo.groupby('tempo_meses')[self.coluna_salario].mean().reset_index()
        salario_atual = df_cbo[self.coluna_salario].mean()
        st.write(f"Salário médio atual: R$ {self.formatar_moeda(salario_atual)}")
        if len(df_mensal) >= 2:
            X = df_mensal[['tempo_meses']]
            y = df_mensal[self.coluna_salario]
            model = LinearRegression()
            model.fit(X, y)
            ult_mes = df_mensal['tempo_meses'].max()
            previsoes = []
            for anos in anos_futuros:
                mes_futuro = ult_mes + anos * 12
                pred = model.predict(np.array([[mes_futuro]]))[0]
                variacao = ((pred - salario_atual) / salario_atual) * 100
                previsoes.append((anos, self.formatar_moeda(max(pred,0)), f"{variacao:+.1f}%"))
            st.write("Tabela de previsão salarial (anos | valor | variação):")
            st.table(pd.DataFrame(previsoes, columns=["Anos","Valor Previsto","Variação"]))
        else:
            st.info("Previsão baseada apenas na média atual.")

        # Tendência do mercado (saldo de movimentação)
        if 'saldomovimentacao' in df_cbo.columns:
            st.subheader("Tendência de vagas")
            df_saldo_mensal = df_cbo.groupby('tempo_meses')['saldomovimentacao'].sum().reset_index()
            if len(df_saldo_mensal) >= 2:
                X_saldo = df_saldo_mensal[['tempo_meses']]
                y_saldo = df_saldo_mensal['saldomovimentacao']
                model_saldo = LinearRegression()
                model_saldo.fit(X_saldo, y_saldo)
                ult_mes = df_saldo_mensal['tempo_meses'].max()
                tendencia_lista = []
                for anos in anos_futuros:
                    mes_futuro = ult_mes + anos * 12
                    pred_saldo = model_saldo.predict(np.array([[mes_futuro]]))[0]
                    if pred_saldo > 100: tendencia = "ALTA DEMANDA"
                    elif pred_saldo > 50: tendencia = "CRESCIMENTO MODERADO"
                    elif pred_saldo > 0: tendencia = "CRESCIMENTO LEVE"
                    elif pred_saldo > -50: tendencia = "RETRAÇÃO LEVE"
                    elif pred_saldo > -100: tendencia = "RETRAÇÃO MODERADA"
                    else: tendencia = "RETRAÇÃO FORTE"
                    tendencia_lista.append((anos, f"{pred_saldo:+,.0f}", tendencia))
                st.write("Tabela previsão de vagas (anos | vagas/mês | tendência):")
                st.table(pd.DataFrame(tendencia_lista, columns=["Anos","Vagas Previstas","Tendência"]))
            else:
                saldo_total = df_cbo['saldomovimentacao'].sum()
                if saldo_total > 100: tendencia = "ALTA DEMANDA"
                elif saldo_total > 0: tendencia = "CRESCIMENTO LEVE"
                elif saldo_total > -100: tendencia = "RETRAÇÃO LEVE"
                else: tendencia = "RETRAÇÃO FORTE"
                st.info(f"Status atual: {tendencia} — Projeção baseada na tendência média.")

# ------------- Streamlit Interface -------------

st.set_page_config(page_title="Análise Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho")
csv_files = [
    "2020_PE1.csv",
    "2021_PE1.csv",
    "2022_PE1.csv",
    "2023_PE1.csv",
    "2024_PE1.csv",
    "2025_PE1.csv"
]
codigos_filepath = "cbo.xlsx"

with st.spinner("Carregando arquivos e preparando dados..."):
    app = MercadoTrabalhoPredictor(csv_files, codigos_filepath)
    app.carregar_dados()
    app.limpar_dados()
st.success("Dados prontos!")

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
        if st.button("Gerar análise e previsão"):
            app.relatorio_previsao(cbo_codigo)
