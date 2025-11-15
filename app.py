"""
Streamlit App — Plataforma Jovem Futuro (versão robusta)
- Carrega dados CAGED (parquet) + códigos CBO (xlsx)
- Visualizações interativas (Plotly)
- Previsões por vários modelos (Prophet, ARIMA, XGBoost com lags, LSTM opcional)
- Modular, com caching, tratamento de erros e proteção contra problemas de DOM
"""

import os
import math
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Optional libraries - import in try blocks (app works without some)
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    HAS_TF = True
except Exception:
    HAS_TF = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Plataforma Jovem Futuro — Previsões", layout="wide")
st.title("📊 Plataforma Jovem Futuro — Previsões do Mercado de Trabalho")

# ---------------------------
# UTILIDADES e CACHING
# ---------------------------

@st.cache_data(ttl=60*60)
def safe_read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data(ttl=60*60)
def safe_read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def format_brl(x):
    try:
        return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(x)

def find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    cols = [c.lower().replace(" ", "").replace("_", "") for c in df.columns]
    for cand in candidates:
        for i, c in enumerate(cols):
            if cand.lower().replace(" ", "").replace("_", "") in c:
                return df.columns[i]
    return None

# ---------------------------
# CLASSE DO SISTEMA
# ---------------------------

class MercadoPredictor:
    def __init__(self, df: pd.DataFrame, df_codigos: pd.DataFrame):
        self.df = df.copy()
        self.df_codigos = df_codigos.copy()
        self.col_cbo = None
        self.col_date = None
        self.col_salary = None
        self.col_saldo = None
        self._identify_columns()

    def _identify_columns(self):
        # Tentativas comuns (sempre extensível)
        self.col_cbo = find_column(self.df, ["cbo", "cbo2002ocupacao", "ocupacao", "ocupação"])
        self.col_date = find_column(self.df, ["competencia", "competenciamov", "data", "mes", "ano"])
        self.col_salary = find_column(self.df, ["salario", "valorsalariofixo", "remuneracao"])
        self.col_saldo = find_column(self.df, ["saldomovimentacao", "saldomovimentação", "saldo"])
        # fallback prints
        st.write("Colunas detectadas:", dict(
            cbo=self.col_cbo, date=self.col_date, salary=self.col_salary, saldo=self.col_saldo
        ))

    def filter_by_cbo(self, cbo_code: str) -> pd.DataFrame:
        if self.col_cbo is None:
            return pd.DataFrame()
        return self.df[self.df[self.col_cbo].astype(str) == str(cbo_code)].copy()

    def prepare_time_series(self, df_cbo: pd.DataFrame, value_col: str, date_col: str, freq='MS'):
        # create data frame with date index and value_col aggregated (mean)
        df_cbo[date_col] = pd.to_datetime(df_cbo[date_col], errors='coerce')
        df_cbo = df_cbo.dropna(subset=[date_col, value_col])
        series = df_cbo.set_index(date_col)[value_col].resample(freq).mean().ffill()
        series = series.rename('y').reset_index()
        return series

    # ---------------------------
    # Model wrappers
    # ---------------------------
    def linear_trend_forecast(self, series: pd.DataFrame, periods: int):
        X = np.arange(len(series)).reshape(-1,1)
        y = series['y'].values
        model = LinearRegression().fit(X, y)
        future_X = np.arange(len(series), len(series)+periods).reshape(-1,1)
        preds = model.predict(future_X)
        return preds, model

    def prophet_forecast(self, series: pd.DataFrame, periods: int):
        if not HAS_PROPHET:
            raise RuntimeError("Prophet não disponível")
        dfp = series.rename(columns={'index':'ds'}) if 'index' in series.columns else series.copy()
        dfp = dfp.rename(columns={'date':'ds'} if 'date' in dfp.columns else {})
        dfp = dfp.rename(columns={'y':'y'})
        # prophet needs ds and y
        dfp = dfp[['ds','y']] if 'ds' in dfp.columns else dfp
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=periods, freq='M')
        fc = m.predict(future)
        preds = fc['yhat'].iloc[-periods:].values
        return preds, m

    def xgb_lag_forecast(self, series: pd.DataFrame, periods: int, lags=12):
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost não disponível")
        # create lagged dataset
        s = series['y'].reset_index(drop=True)
        dfX = pd.concat([s.shift(i) for i in range(1, lags+1)], axis=1)
        dfX.columns = [f'lag_{i}' for i in range(1, lags+1)]
        dfX['y'] = s
        dfX = dfX.dropna()
        X = dfX.drop(columns='y').values
        y = dfX['y'].values
        model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        model.fit(X, y)
        # iterative forecasting
        last_X = X[-1].copy()
        preds = []
        for _ in range(periods):
            p = model.predict(last_X.reshape(1,-1))[0]
            preds.append(p)
            last_X = np.roll(last_X, -1)
            last_X[-1] = p
        return preds, model

    def arima_forecast(self, series: pd.Series, periods: int):
        if not HAS_ARIMA:
            raise RuntimeError("ARIMA/Statsmodels não disponível")
        y = series.values
        # simple ARIMA(1,1,1) fallback
        model = ARIMA(y, order=(1,1,1))
        fit = model.fit()
        preds = fit.forecast(steps=periods)
        return preds, fit

    def lstm_forecast(self, series: pd.Series, periods: int, lags=12, epochs=40):
        if not HAS_TF:
            raise RuntimeError("TensorFlow não disponível")
        s = series.values
        # build training lag dataset
        X, y = [], []
        for i in range(lags, len(s)):
            X.append(s[i-lags:i])
            y.append(s[i])
        X = np.array(X); y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential()
        model.add(LSTM(64, input_shape=(X.shape[1],1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mae')
        model.fit(X,y, epochs=epochs, batch_size=16, verbose=0)
        last = s[-lags:].reshape((1,lags,1))
        preds = []
        for _ in range(periods):
            p = model.predict(last, verbose=0)[0,0]
            preds.append(p)
            last = np.roll(last, -1)
            last[0,-1,0] = p
        return preds, model

# ---------------------------
# UI helpers
# ---------------------------

def plot_series(series: pd.DataFrame, title="Série temporal", preds: Optional[Dict[str, Any]] = None):
    if HAS_PLOTLY:
        fig = px.line(series, x=series.columns[0], y='y', title=title, labels={series.columns[0]:'Data', 'y':'Valor'})
        if preds:
            # build future index
            last_date = pd.to_datetime(series[series.columns[0]].iloc[-1])
            future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=len(list(preds.values())[0]), freq='MS')
            for name, arr in preds.items():
                fig.add_trace(go.Scatter(x=future_dates, y=arr, mode='lines+markers', name=name))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(series.set_index(series.columns[0])['y'])

# ---------------------------
# APP LÓGICA
# ---------------------------

with st.sidebar:
    st.header("Configuração")
    parquet_path = st.text_input("Caminho do parquet (dados CAGED)", value="dados.parquet")
    cbo_path = st.text_input("Caminho do arquivo CBO (xlsx)", value="cbo.xlsx")
    years_min = st.slider("Anos mínimos para considerar histórica", 1, 10, 3)

# Load
if not os.path.exists(parquet_path) or not os.path.exists(cbo_path):
    st.sidebar.error("Forneça caminhos válidos para os arquivos (parquet + cbo.xlsx).")
    st.stop()

with st.spinner("Carregando dados..."):
    df = safe_read_parquet(parquet_path)
    df_cod = safe_read_excel(cbo_path)
    predictor = MercadoPredictor(df, df_cod)

st.success("Dados carregados e verificados.")

# Search & select
st.markdown("### 🔎 Buscar profissão (nome ou código)")
query = st.text_input("Nome ou código CBO")
if query:
    res = predictor.df_codigos[predictor.df_codigos['cbo_descricao'].str.contains(query, case=False, na=False)] \
          if not query.isdigit() else predictor.df_codigos[predictor.df_codigos['cbo_codigo'] == query]
    if res.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        option = st.selectbox("Selecione a profissão", [f"{r['cbo_codigo']} - {r['cbo_descricao']}" for _, r in res.iterrows()])
        cbo = option.split(" - ")[0]

        # Use session_state to avoid DOM removeChild issues
        if 'generate' not in st.session_state:
            st.session_state.generate = False

        if st.button("Gerar análise e previsões"):
            st.session_state.generate = True

        if st.session_state.generate:
            placeholder = st.container()
            with placeholder:
                df_cbo = predictor.filter_by_cbo(cbo)
                st.subheader(f"📋 Perfil e dados de: {option.split(' - ',1)[1]}")
                st.write(f"Registros encontrados: {len(df_cbo):,}")

                # Demografia resumida
                with st.expander("👥 Perfil demográfico"):
                    if 'idade' in df_cbo.columns:
                        st.write("Idade média:", df_cbo['idade'].astype(float).mean())
                    if 'sexo' in df_cbo.columns:
                        st.write("Distribuição por sexo:")
                        st.write(df_cbo['sexo'].astype(str).value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

                # Salário série temporal
                if predictor.col_salary and predictor.col_date:
                    try:
                        series = predictor.prepare_time_series(df_cbo, predictor.col_salary, predictor.col_date)
                        st.markdown("#### Série salarial (mensal)")
                        plot_series(series, title="Salário médio mensal")
                    except Exception as e:
                        st.error(f"Erro ao construir série salarial: {e}")
                else:
                    st.info("Colunas de salário/data não encontradas — impossível gerar série temporal salarial.")

                # Previsões: execute modelos e compare
                st.markdown("----")
                st.subheader("🔮 Previsões (comparativo de modelos)")

                periods_years = st.multiselect("Horizontes (anos) para previsão", [1,3,5,10], default=[1,3,5])
                periods = max(periods_years) * 12

                results = {}
                errors = []

                if predictor.col_salary and predictor.col_date:
                    try:
                        s = predictor.prepare_time_series(df_cbo, predictor.col_salary, predictor.col_date)
                        # ensure enough history
                        if len(s) < 12:
                            st.warning("Histórico curto (<12 meses): previsões simples via média/linear serão usadas.")
                            # linear fallback
                            preds_lin, _ = predictor.linear_trend_forecast(s, periods)
                            results['Linear'] = preds_lin
                        else:
                            # Prophet
                            if HAS_PROPHET:
                                try:
                                    preds_p, _ = predictor.prophet_forecast(s.rename(columns={s.columns[0]:'ds'}), periods)
                                    results['Prophet'] = preds_p
                                except Exception as e:
                                    errors.append(f"Prophet erro: {e}")
                            # ARIMA
                            if HAS_ARIMA:
                                try:
                                    preds_a, _ = predictor.arima_forecast(s['y'], periods)
                                    results['ARIMA'] = np.array(preds_a)
                                except Exception as e:
                                    errors.append(f"ARIMA erro: {e}")
                            # XGBoost lag
                            if HAS_XGBOOST:
                                try:
                                    preds_xgb, _ = predictor.xgb_lag_forecast(s, periods, lags=12)
                                    results['XGBoost-Lags'] = np.array(preds_xgb)
                                except Exception as e:
                                    errors.append(f"XGBoost erro: {e}")
                            # LSTM optional
                            if HAS_TF:
                                try:
                                    preds_lstm, _ = predictor.lstm_forecast(s['y'], periods, lags=12, epochs=30)
                                    results['LSTM'] = np.array(preds_lstm)
                                except Exception as e:
                                    errors.append(f"LSTM erro: {e}")
                    except Exception as e:
                        st.error(f"Erro ao preparar previsões: {e}")
                else:
                    st.info("Sem colunas de salário/data — previsões desabilitadas.")

                # Mostrar resultados resumidos
                if results:
                    # Align lengths: choose shortest predictions length to compare
                    min_len = min(len(v) for v in results.values())
                    # build future dates base
                    last_date = pd.to_datetime(df_cbo[predictor.col_date]).max()
                    future_idx = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=min_len, freq='MS')
                    if HAS_PLOTLY:
                        fig = go.Figure()
                        # historical series if available
                        if predictor.col_date and predictor.col_salary:
                            try:
                                hist = predictor.prepare_time_series(df_cbo, predictor.col_salary, predictor.col_date)
                                fig.add_trace(go.Scatter(x=hist[predictor.col_date], y=hist['y'], mode='lines', name='Histórico'))
                            except:
                                pass
                        for name, arr in results.items():
                            fig.add_trace(go.Scatter(x=future_idx, y=arr[:min_len], mode='lines+markers', name=name))
                        fig.update_layout(title="Comparativo de previsões", xaxis_title="Data", yaxis_title="Salário")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # fallback to simple display
                        st.write("Predições (amostras):")
                        for name, arr in results.items():
                            st.write(f"- {name}: {arr[:min_len].tolist()}")
                else:
                    st.info("Nenhum resultado de previsão gerado (verifique disponibilidade de libs).")

                if errors:
                    with st.expander("Erros / detalhes técnicos"):
                        for e in errors:
                            st.write("- " + str(e))

                # Export CSV button for predictions
                if results:
                    # produce dataframe of future predictions (min_len)
                    df_out = pd.DataFrame({name: arr[:min_len] for name, arr in results.items()}, index=future_idx)
                    csv = df_out.reset_index().rename(columns={'index':'date'}).to_csv(index=False)
                    st.download_button("📥 Baixar previsões (CSV)", data=csv, file_name=f"previsoes_{cbo}.csv", mime="text/csv")

                # Reset generate if needed
                if st.button("Nova busca / Limpar"):
                    st.session_state.generate = False
                    placeholder.empty()
