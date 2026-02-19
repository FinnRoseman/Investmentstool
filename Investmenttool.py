import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")


# --- 1. SETUP ---
st.sidebar.header("Portfolio Zusammenstellung")
if 'meine_ticker' not in st.session_state:
    st.session_state.meine_ticker = []
ticker_input = st.sidebar.text_input("Ticker-Symbol eingeben & Enter")
if ticker_input:
    neuer_t = ticker_input.strip().upper()
    if neuer_t not in st.session_state.meine_ticker:
        st.session_state.meine_ticker.append(neuer_t)
        st.session_state.run_analysis = False 
        st.rerun()
ticker_liste = st.sidebar.multiselect(
    "Aktive Auswahl:",
    options=st.session_state.meine_ticker,
    default=st.session_state.meine_ticker
)
st.session_state.meine_ticker = ticker_liste
anteile_orig = []
fx_map = {}
for t in ticker_liste:
      with st.sidebar.expander(f"Einstellungen für {t}"):
            gewicht = st.number_input(
                f"Gewicht (%)", 
                min_value=0.0, max_value=100.0, value=0.0, step=0.1, 
                format="%.2f", key=f"w_in_{t}"
            )
            anteile_orig.append(gewicht / 100)
            
            fremd_check = st.checkbox(f"In EUR umrechnen?", key=f"check_{t}")
            if fremd_check:
                waehrung = st.selectbox(
                    f"Ursprüngliche Währung von {t}",
                    options=["USD", "JPY", "GBP", "CHF", "SEK", "CAD"],
                    index=0,
                    key=f"curr_{t}"
                )
                fx_map[t] = f"{waehrung}EUR=X"

st.sidebar.markdown("---")
go_button = st.sidebar.button("Go", use_container_width=True)
if go_button:
    st.session_state.run_analysis = True

risk_free_rate = 0.02

# --- 2. DESIGN ---
st.sidebar.header("Benchmarkauswahl")
bench_optionen = {
    "100/0 (MSCI World)": "EUNL.DE",
    "80/20 (LifeStrategy 80% Equity)": "V80A.DE",
    "60/40 (LifeStrategy 60% Equity)": "V60A.DE",
    "40/60 (LifeStrategy 40% Equity)": "V40A.DE",
    "20/80 (LifeStrategy 20% Equity)": "V20A.DE",
    "0/100 (Global Bonds)": "EUNA.DE"
}
ausgewaehlter_name = st.sidebar.selectbox("Vergleichs-Index", list(bench_optionen.keys()))
benchmark = bench_optionen[ausgewaehlter_name]

st.sidebar.header("Zeitraumauswahl")
zeitraum_optionen = {
    "1 Jahr": "1y",
    "3 Jahre": "3y",
    "5 Jahre": "5y",
    "10 Jahre": "10y",
    "20 Jahre": "20y"
}
ausgewaehlter_zeitraum = st.sidebar.selectbox("Zeitraum", list(zeitraum_optionen.keys()), index=2)
period_yf = zeitraum_optionen[ausgewaehlter_zeitraum]

zuordnung = dict(zip(ticker_liste, anteile_orig))

ticker_namen = {}
for t in ticker_liste:
    try:
        ticker_namen[t] = yf.Ticker(t).info.get('longName', t)
    except:
        ticker_namen[t] = t

st.title("📈 Portfolio Backtest Dashboard")
with st.expander("📋 Portfolio-Zusammensetzung (Namen & Gewichtung)"):
    legende_df = pd.DataFrame({
        "Ticker": ticker_liste,
        "Name": [ticker_namen.get(t, t) for t in ticker_liste],
        "Anteil": [f"{zuordnung.get(t, 0)*100:.1f}%" for t in ticker_liste]
    })
    legende_df.index = range(1, len(legende_df) + 1)
    st.table(legende_df)
    summe_anteile = sum(anteile_orig)
    if abs(summe_anteile - 1.0) > 0.001:
        st.warning(f"⚠️ Die Summe der Anteile liegt bei {summe_anteile*100:.1f}%.")
    else:
        st.success("✅ Die Anteile ergeben 100%.")

if not ticker_liste:
    st.info("👋 Das Portfolio ist gerade noch leer. Füge die Zielpositionen ein und komm dann wieder! Viel Erfolg beim Investieren.")
    st.stop()

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

if not st.session_state.run_analysis:
    if ticker_liste:
        st.warning("👈 Gewichtung einstellen und auf 'Go' klicken, um die Analyse zu starten.")
    else:
        st.info("Bitte füge Ticker in der Sidebar hinzu.")
    st.stop()

raw_data = {}
for t in ticker_liste + [benchmark]:
    df = yf.download(t, period=period_yf, progress=False)
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            price = df['Close'][t].copy()
        else:
            price = df['Close'].copy()
        
        if t in fx_map:
            fx_df = yf.download(fx_map[t], period=period_yf, progress=False)
            if not fx_df.empty:
                fx = fx_df['Close'].iloc[:, 0] if isinstance(fx_df.columns, pd.MultiIndex) else fx_df['Close']
                comb = pd.concat([price, fx], axis=1).ffill().dropna()
                price = comb.iloc[:, 0] * comb.iloc[:, 1]
        raw_data[t] = price

daten = pd.concat(raw_data, axis=1).ffill().dropna()
renditen = daten.pct_change().dropna()

# --- 3. KENNZAHLEN ---
verfuegbare = [t for t in ticker_liste if t in renditen.columns]
summe_anteile_input = sum(anteile_orig)
if summe_anteile_input <= 0:
    st.warning("Gewichtung hinzufügen um Analyse zu starten.")
    st.stop()
anteile = [anteile_orig[ticker_liste.index(t)] for t in verfuegbare]
anteile = [a/sum(anteile) for a in anteile]

port_rendite = (renditen[verfuegbare] * anteile).sum(axis=1)
bench_rendite = renditen.loc[port_rendite.index, benchmark]
diff_rendite = port_rendite - bench_rendite

jahre = (daten.index[-1] - daten.index[0]).days / 365.25
total_ret = (1 + port_rendite).prod() - 1
cagr = (1 + total_ret)**(1/jahre) - 1
vola = port_rendite.std() * np.sqrt(252)
max_drawdown = (((1 + port_rendite).cumprod() / (1 + port_rendite).cumprod().cummax()) - 1).min()
tracking_error = diff_rendite.std() * np.sqrt(252)

sharpe_ratio = (cagr - risk_free_rate) / vola
beta = port_rendite.cov(bench_rendite) / bench_rendite.var()
bench_cagr = ((1 + bench_rendite).prod())**(1/jahre) - 1
alpha = cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))

# Risikokennzahlen
var_95_para = 1.645 * vola
var_95_hist = abs(port_rendite.quantile(0.05)) * np.sqrt(252)
es_95_para = 2.06 * (port_rendite.std() * np.sqrt(252))
es_95_tag_hist = port_rendite[port_rendite <= port_rendite.quantile(0.05)].mean()
es_95_hist = abs(es_95_tag_hist) * np.sqrt(252)

# Monte Carlo
simulations = 10000
tage = 252
daily_rets = port_rendite.dropna()
centered_rets = daily_rets - daily_rets.mean()
sim_returns = np.random.choice(centered_rets, size=(tage, simulations), replace=True)
paths = np.prod(1 + sim_returns, axis=0) - 1
schwellenwert = np.percentile(paths, 5)
mc_var_95_jahr = schwellenwert * -1
mc_es_95_jahr = paths[paths <= schwellenwert].mean() * -1

# Euro Rechner
st.sidebar.markdown("---")
st.sidebar.header("Kapitalauswahl")
startkapital = st.sidebar.number_input("Startkapital (€)", value=0, min_value=0, step=1000)
endsumme = startkapital * (1 + total_ret)
absoluter_gewinn = endsumme - startkapital
farbe = "#28a745" if total_ret >= 0 else "#dc3545"
st.subheader(f"Wertentwicklung bei {startkapital:,.0f} € Investment")
e1, e2, e3 = st.columns([1.5, 1.2, 1])
e1.metric("Endwert Heute", f"{endsumme:,.2f} €")
if startkapital > 0:
    e2.metric("Seit Kauf Absolut", f"{absoluter_gewinn:,.2f} €")
    e3.metric("Seit Kauf Relativ", f"{total_ret:.2%}")
else:
    e2.metric("Seit Kauf Absolut", "0.00€")
    e3.metric("Seit Kauf Relativ", "0.00%")

# --- 4. ANZEIGEN ---

# Kennzahlen