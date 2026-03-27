import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas_datareader.data as web

# --- CACHING FUNKTION ---
@st.cache_data(show_spinner="Marktdaten werden geladen...")
def get_cached_data(ticker_tuple, period):
    df = yf.download(list(ticker_tuple), period=period, progress=False)
    return df
def get_factor_loadings(portfolio_returns):
    import statsmodels.api as sm
    ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010-01-01')[0]
    ff_data = ff_data / 100
    ff_data.index = ff_data.index.to_timestamp().to_period('M')
    port_monthly = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    port_monthly.index = port_monthly.index.to_period('M')
    combined = pd.concat([port_monthly, ff_data], axis=1).dropna()
    if len(combined) < 5:
        raise ValueError(f"Überschneidung zu gering: Nur {len(combined)} Monate gefunden.")
    Y = combined.iloc[:, 0] - combined['RF']
    X = combined[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model.params[1:]

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")


# --- 1. SETUP ---
st.sidebar.header("Portfolio Zusammenstellung")
def clear_ticker_input():
    st.session_state.ticker_temp = st.session_state.widget_eingabe
    st.session_state.widget_eingabe = ""
if 'meine_ticker' not in st.session_state:
    st.session_state.meine_ticker = []
if 'regionen_daten' not in st.session_state:
    st.session_state.regionen_daten = {}
st.sidebar.text_input("Ticker-Symbol eingeben & Enter", key="widget_eingabe", on_change=clear_ticker_input)
ticker_input = st.session_state.get('ticker_temp', None)
if ticker_input:
    neuer_t = ticker_input.strip().upper()
    if neuer_t not in st.session_state.meine_ticker:
        st.session_state.meine_ticker.append(neuer_t)
        st.session_state.run_analysis = False 
    st.session_state.ticker_temp = None
    st.rerun()
ticker_liste = st.sidebar.multiselect(
    "Aktive Auswahl:",
    options=st.session_state.meine_ticker,
    default=st.session_state.meine_ticker
)
st.session_state.meine_ticker = ticker_liste
if 'ticker_kontrolle' not in st.session_state:
    st.session_state.ticker_kontrolle = list(ticker_liste)
if ticker_liste != st.session_state.ticker_kontrolle:
    st.session_state.run_analysis = False
    st.session_state.ticker_kontrolle = list(ticker_liste)
    
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
                w_options = ["USD", "JPY", "GBP", "CHF", "SEK", "CAD"]
                if f"val_{t}" not in st.session_state:
                    st.session_state[f"val_{t}"] = "USD"
                waehrung = st.selectbox(
                    f"Ursprüngliche Währung von {t}",
                    options=w_options,
                    index=w_options.index(st.session_state[f"val_{t}"]),
                    key=f"curr_{t}"
                )
                st.session_state[f"val_{t}"] = waehrung
                fx_map[t] = f"{waehrung}EUR=X"
            st.markdown("**Regionale Verteilung (%)**")
            r_data = st.session_state.regionen_daten.get(t, {"NA": 0.0, "SA": 0.0, "EU": 0.0, "AP": 0.0, "AF": 0.0})
            c1, c2, c3 = st.columns(3)
            with c1: na = st.number_input("NAM", 0.0, 100.0, r_data["NA"], key=f"na_{t}")
            with c2: sa = st.number_input("SAM", 0.0, 100.0, r_data["SA"], key=f"sa_{t}")
            with c3: eu = st.number_input("Europa", 0.0, 100.0, r_data["EU"], key=f"eu_{t}")
            c4, c5, _ = st.columns(3)
            with c4: ap = st.number_input("APAC", 0.0, 100.0, r_data["AP"], key=f"ap_{t}")
            with c5: af = st.number_input("Afrika", 0.0, 100.0, r_data["AF"], key=f"af_{t}")
            st.session_state.regionen_daten[t] = {"NA": na, "SA": sa, "EU": eu, "AP": ap, "AF": af}

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
    "0/100 (Global Bonds)": "EUNA.DE",
    "Individueller Ticker...": "CUSTOM"
}
ausgewaehlter_name = st.sidebar.selectbox("Vergleichs-Index", list(bench_optionen.keys()))

if ausgewaehlter_name == "Individueller Ticker...":
    benchmark = st.sidebar.text_input("Gib den Ticker ein (z.B. SPY):", "SPY").strip().upper()
else:
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

st.sidebar.header("Kapitalauswahl")
startkapital = st.sidebar.number_input("Startkapital (€)", value=0, min_value=0, step=1000, key="mein_kapital")

zuordnung = dict(zip(ticker_liste, anteile_orig))

ticker_namen = {}
for t in ticker_liste:
    try:
        ticker_namen[t] = yf.Ticker(t).info.get('longName', t)
    except:
        ticker_namen[t] = t

st.title("Portfolio Backtest Dashboard")
with st.expander("Portfolio-Zusammensetzung (Name & Gewichtung)"):
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
    st.info("Das Portfolio ist gerade noch leer. Starte mit der Zusammenstellung.")
    st.stop()
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False    
if not st.session_state.run_analysis:
    if ticker_liste:
        st.warning("👈 Gewichtung einstellen und auf 'Go' klicken, um die Analyse zu starten.")
    else:
        st.info("Bitte füge Ticker in der Sidebar hinzu.")
    st.stop()

alle_ticker = tuple(ticker_liste + [benchmark])
data_full = get_cached_data(alle_ticker, period_yf)

raw_data = {}
for t in alle_ticker:
    if isinstance(data_full.columns, pd.MultiIndex):
        price = data_full['Close'][t].copy()
    else:
        price = data_full['Close'].copy()
    if price.dropna().empty:
        st.error(f"⚠️ Keine Daten für {t}")
        st.stop()
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
total_na, total_sa, total_eu, total_ap, total_af = 0.0, 0.0, 0.0, 0.0, 0.0
for i, t in enumerate(verfuegbare):
    reg = st.session_state.regionen_daten.get(t, {"NA": 0.0, "SA": 0.0, "EU": 0.0, "AP": 0.0, "AF": 0.0})
    gewicht = anteile[i]
    total_na += reg["NA"] * gewicht
    total_sa += reg["SA"] * gewicht
    total_eu += reg["EU"] * gewicht
    total_ap += reg["AP"] * gewicht
    total_af += reg["AF"] * gewicht

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
capm_erwartung_pa = risk_free_rate + beta * (bench_cagr - risk_free_rate)

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

# Risikoverteilung
cov_matrix = renditen[verfuegbare].cov() * 252
port_vola = np.sqrt(np.dot(anteile, np.dot(cov_matrix, anteile)))
marginal_contrib = np.dot(cov_matrix, anteile) / port_vola
abs_risk_contrib = anteile * marginal_contrib
rel_risk_contrib = abs_risk_contrib / port_vola
has_negative_risk = any(val < 0 for val in rel_risk_contrib)

# Euro Rechner
st.markdown("---")
endsumme = startkapital * (1 + total_ret)
absoluter_gewinn = endsumme - startkapital
port_current_yield = 0.0
port_yoc = 0.0
total_div_euro = 0.0
cal_data = []
for i, t in enumerate(verfuegbare):
    ticker_obj = yf.Ticker(t)
    gewicht = anteile[i]
    div_history = ticker_obj.dividends
    if not div_history.empty:
        divs_clean = div_history.index.tz_localize(None)
        divs_in_period = div_history[divs_clean.isin(daten.index)]
        anzahl_jahre = max(jahre, 1.0)
        try:
            last_date = daten.index[-1]
            price_raw = ticker_obj.history(start=last_date - pd.Timedelta(days=5), end=last_date + pd.Timedelta(days=1))['Close'].iloc[-1]
            price_eur = daten[t].iloc[-1]
            fx_faktor = price_eur / price_raw
        except:
            fx_faktor = 1.0
        for date, amount in divs_in_period.items():
            stueckzahl_start = (startkapital * gewicht) / daten[t].iloc[0]
            euro_zahlung_avg = (amount * fx_faktor * stueckzahl_start) / anzahl_jahre
            if euro_zahlung_avg > 0:
                cal_data.append({
                    "Monat": date.strftime("%B"), "Monat_Nr": date.month,
                    "Ticker": t, "Ausschüttung": euro_zahlung_avg
                })
        last_year_divs_eur = div_history[divs_clean > (pd.Timestamp.now() - pd.Timedelta(days=365))].sum() * fx_faktor
        stueckzahl_heute = (startkapital * gewicht) / daten[t].iloc[0]
        total_div_euro += (last_year_divs_eur * stueckzahl_heute)
        if daten[t].iloc[-1] > 0:
            ticker_yield = last_year_divs_eur / daten[t].iloc[-1]
            port_current_yield += ticker_yield * gewicht
            price_at_start_eur = daten[t].iloc[0]
            try:
                start_date = daten.index[0]
                price_orig_start = ticker_obj.history(start=start_date, end=start_date + pd.Timedelta(days=7))['Close'].iloc[0]
            except:
                price_orig_start = price_at_start_eur 
            div_orig_ltm = last_year_divs_eur / fx_faktor if fx_faktor > 0 else last_year_divs_eur
            ticker_yoc = div_orig_ltm / price_orig_start if price_orig_start > 0 else 0
            port_yoc += ticker_yoc * gewicht
avg_capital = (startkapital + endsumme) / 2
st.subheader(f"Wertentwicklung bei {startkapital:,.0f} € Investment")
farbe_perf = "#27AE60" if absoluter_gewinn >= 0 else "#EB5757"
div_wert = 0.00 if np.isnan(total_div_euro) else total_div_euro
abs_anzeige = f"{absoluter_gewinn:,.2f} €" if startkapital > 0 else "0.00 €"
rel_anzeige = f"{total_ret:.2%}" if startkapital > 0 else "0.00%"
div_anzeige = f"{div_wert:,.2f} €" if startkapital > 0 else "0.00 €"
st.markdown("""
<style>
    .header-grid {
        display: grid;
        grid-template-columns: 1.5fr 1.2fr 1fr 1.2fr;
        gap: 15px;
        margin-bottom: 25px;
    }
    .header-card {
        background-color: rgba(100,100,100,0.1);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4A90E2;
    }
    .header-label { color: #9CA3AF !important; font-size: 14px !important; margin: 0 !important; }
    .header-value { color: white !important; font-size: 28px !important; font-weight: bold !important; margin: 5px 0 0 0 !important; }
</style>
""", unsafe_allow_html=True)

html_code = f"""
<div class="header-grid">
<div class="header-card">
<p class="header-label">Endwert Heute</p>
<p class="header-value">{endsumme:,.2f} €</p>
</div>
<div class="header-card" style="border-left-color: {farbe_perf};">
<p class="header-label">Seit Kauf Absolut</p>
<p class="header-value" style="color: {farbe_perf} !important;">{abs_anzeige}</p>
</div>
<div class="header-card" style="border-left-color: {farbe_perf};">
<p class="header-label">Seit Kauf Relativ</p>
<p class="header-value" style="color: {farbe_perf} !important;">{rel_anzeige}</p>
</div>
<div class="header-card">
<p class="header-label">Ausschüttungen (LTM)</p>
<p class="header-value">{div_anzeige}</p>
</div>
</div>
"""

st.write(html_code, unsafe_allow_html=True)
st.markdown("---")

# --- 4. ANZEIGEN ---

# Kennzahlen-Kacheln
st.subheader("🔢 Key Performance Indicators")
st.markdown("""
<style>
    /* Grid für die erste Reihe (5 Kacheln) */
    .kpi-grid-5-row1 {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 15px;
        margin-bottom: 15px;
    }
    /* Grid für die zweite Reihe (5 Kacheln) */
    .kpi-grid-5-row2 {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 15px;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: rgba(100,100,100,0.1);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4A90E2; /* Einheitliches Blau für ALLE Ränder */
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        background-color: rgba(100,100,100,0.15);
    }
    .kpi-label {
        color: #9CA3AF;
        font-size: 13px;
        margin: 0;
    }
    .kpi-value {
        color: white; /* Einheitliches Weiß für ALLE Werte */
        font-size: 24px;
        font-weight: bold;
        margin: 5px 0 0 0;
    }
    /* Responsives Verhalten für kleinere Bildschirme */
    @media (max-width: 1200px) {
        .kpi-grid-5-row1, .kpi-grid-5-row2 {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    @media (max-width: 800px) {
        .kpi-grid-5-row1, .kpi-grid-5-row2 {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 500px) {
        .kpi-grid-5-row1, .kpi-grid-5-row2 {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-grid-5-row1">
    <div class="kpi-card"><p class="kpi-label">Erwartete Rendite (CAPM)</p><p class="kpi-value">{capm_erwartung_pa:.2%}</p></div>
    <div class="kpi-card"><p class="kpi-label">Rendite p.a. (CAGR)</p><p class="kpi-value">{cagr:.2%}</p></div>
    <div class="kpi-card"><p class="kpi-label">Alpha</p><p class="kpi-value">{alpha:.2%}</p></div> <div class="kpi-card"><p class="kpi-label">Sharpe Ratio</p><p class="kpi-value">{sharpe_ratio:.2f}</p></div>
    <div class="kpi-card"><p class="kpi-label">Dividendenrendite</p><p class="kpi-value">{port_current_yield:.2%}</p></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-grid-5-row2">
    <div class="kpi-card"><p class="kpi-label">Max Drawdown</p><p class="kpi-value">{max_drawdown:.2%}</p></div> <div class="kpi-card"><p class="kpi-label">Volatilität p.a.</p><p class="kpi-value">{vola:.2%}</p></div>
    <div class="kpi-card"><p class="kpi-label">Beta</p><p class="kpi-value">{beta:.2f}</p></div>
    <div class="kpi-card"><p class="kpi-label">Tracking Error</p><p class="kpi-value">{tracking_error:.2%}</p></div>
    <div class="kpi-card"><p class="kpi-label">Yield on Cost</p><p class="kpi-value">{port_yoc:.2%}</p></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# 1. Performance-Chart (Full Width oben)
st.subheader("📈 Performance & Trends", help="Vergleich der Portfolio-Performance gegen die Benchmark inklusive gleitender Durchschnitte (100/200 Tage).")
fig_perf, ax_perf = plt.subplots(figsize=(10, 5), constrained_layout=True)
port_kum = ((1 + port_rendite).cumprod() - 1) * 100
bench_kum = ((1 + bench_rendite).cumprod() - 1) * 100
sma100 = port_kum.rolling(window=100).mean()
sma200 = port_kum.rolling(window=200).mean()
ax_perf.plot(port_kum, label='Portfolio', linewidth=2, color='blue')
ax_perf.plot(bench_kum, label='Benchmark', linewidth=1.5, color='grey', linestyle='--', alpha=0.6)
ax_perf.plot(sma100, label='100-Tage-Linie', color='orange', linewidth=1, alpha=0.8)
ax_perf.plot(sma200, label='200-Tage-Linie', color='red', linewidth=1, alpha=0.8)
ax_perf.set_ylabel('Entwicklung (%)')
ax_perf.legend(loc='upper left', fontsize=9)
ax_perf.grid(True, alpha=0.2)
st.pyplot(fig_perf)

st.markdown("---")

# 2. Spalten für Rendite-Check und Regionen-Verteilung
col_rendite, col_regionen = st.columns([1, 1])

with col_rendite:
    st.subheader("💰 Rendite-Analyse", help="Jährliche Renditen und die kumulierte Rendite über feste Zeiträume.")
    yearly_ret = port_rendite.groupby(port_rendite.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    periods = {"1Y": 252, "3Y": 756, "5Y": 1260, "10Y": 2520, "20Y": 5040}
    period_rets = {label: ((1 + port_rendite.iloc[-days:]).prod() - 1) * 100 
                   for label, days in periods.items() if len(port_rendite) >= days}
    fig_balken, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), constrained_layout=True) 
    colors_y = ['#27AE60' if x > 0 else '#EB5757' for x in yearly_ret]
    ax1.bar(yearly_ret.index.astype(str), yearly_ret.values, color=colors_y)
    ax1.set_title("Annualisiert (%)", fontsize=10, fontweight='bold')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.tick_params(axis='both', labelsize=8)
    labels_p = list(period_rets.keys())
    values_p = list(period_rets.values())
    colors_p = ['#27AE60' if x > 0 else '#EB5757' for x in values_p]
    ax2.bar(labels_p, values_p, color=colors_p)
    ax2.set_title("Kumuliert (%)", fontsize=10, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.tick_params(axis='both', labelsize=8)
    
    st.pyplot(fig_balken)

with col_regionen:
    st.subheader("🌎 Regionale Verteilung", help="Geografische Gewichtung des Portfolios.")
    if (total_na + total_sa + total_eu + total_ap + total_af) > 0:
        reg_labels = ['Nordamerika', 'Südamerika', 'Europa', 'Asien-Pazifik', 'Afrika']
        reg_values = [total_na, total_sa, total_eu, total_ap, total_af]
        reg_colors = ['#4A90E2', '#9B51E0', '#F2994A', '#27AE60', '#EB5757']
        labels_filtered = [l for l, v in zip(reg_labels, reg_values) if v > 0]
        values_filtered = [v for v in reg_values if v > 0]
        colors_filtered = [c for c, v in zip(reg_colors, reg_values) if v > 0]
        fig_reg, ax_reg = plt.subplots(figsize=(5, 5), constrained_layout=True) 
        fig_reg.patch.set_facecolor('white')
        ax_reg.pie(
            values_filtered, 
            labels=labels_filtered, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=colors_filtered,
            textprops={'color':"black", 'weight':'bold', 'fontsize': 6.5}, 
            pctdistance=0.7, 
            labeldistance=1.1 
        )
        ax_reg.axis('equal') 
        st.pyplot(fig_reg)
    else:
        st.info("Daten in Sidebar eintragen.")

st.markdown("---")

# Rolling Returns & Rendite Verteilung
att_col1, att_col2 = st.columns(2)

with att_col1:
    st.subheader("🔄 Rolling Returns (12 Monate)", help="Zeigt die Rendite eines Zeitpunktes im Vergleich zu dem gleichen Zeitpunkt vor einem Jahr.")
    rolling_1y = port_rendite.rolling(window=252).apply(lambda x: (1 + x).prod() - 1)
    
    fig_roll, ax_roll = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax_roll.plot(rolling_1y * 100, color='blue', alpha=0.8)
    ax_roll.axhline(rolling_1y.mean() * 100, color='red', linestyle='--', label='Schnitt')
    ax_roll.axhline(0, color='black', linewidth=1)
    ax_roll.set_ylabel('Rendite p.a. (%)')
    ax_roll.fill_between(rolling_1y.index, rolling_1y * 100, 0, where=(rolling_1y > 0), facecolor='green', alpha=0.1)
    ax_roll.fill_between(rolling_1y.index, rolling_1y * 100, 0, where=(rolling_1y < 0), facecolor='red', alpha=0.1)
    ax_roll.grid(True, alpha=0.3)
    st.pyplot(fig_roll)

with att_col2:
    st.subheader("📊 Rendite-Verteilung", help="Gibt an, welche Position wie viel zur Gesamtrendite beiträgt")
    beitraege = []
    for t in verfuegbare:
        einzel_ret = (1 + renditen[t]).prod() - 1
        gewicht = anteile[verfuegbare.index(t)]
        beitraege.append(einzel_ret * gewicht)
    fig_att, ax_att = plt.subplots(figsize=(10, 6), constrained_layout=True)
    y_pos_att = np.arange(len(verfuegbare))
    bars_att = ax_att.barh(y_pos_att, np.array(beitraege) * 100, color='skyblue')
    for bar in bars_att:
        width = bar.get_width()
        ax_att.text(width, bar.get_y() + bar.get_height()/2, 
                    f' {width:.1f}%', 
                    va='center', fontweight='bold', fontsize=11)

    ax_att.set_yticks(y_pos_att)
    ax_att.set_yticklabels(verfuegbare)
    ax_att.invert_yaxis()
    ax_att.set_xlabel('Beitrag zur Gesamtrendite (%)')
    ax_att.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig_att)

st.markdown("---")

# Dividendenkalender
st.subheader("📅 Dividenden-Kalender", help="Zeigt, wann wie viele Ausschüttungen zu erwarten sind")
if cal_data:
    df_cal = pd.DataFrame(cal_data)
    monate_de = {"January": "Jan", "February": "Feb", "March": "Mär", "April": "Apr",
                 "May": "Mai", "June": "Jun", "July": "Jul", "August": "Aug",
                 "September": "Sep", "October": "Okt", "November": "Nov", "December": "Dez"}
    df_cal['Monat'] = df_cal['Monat'].map(monate_de)
    df_pivot = df_cal.pivot_table(index='Monat_Nr', columns='Ticker', values='Ausschüttung', aggfunc='sum').fillna(0)
    df_pivot = df_pivot.reindex(range(1, 13), fill_value=0)
    fig_div, ax_div = plt.subplots(figsize=(12, 4)) 
    df_pivot.plot(kind='bar', stacked=True, ax=ax_div, color=plt.cm.Paired.colors, width=0.7)
    monats_namen = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    ax_div.set_xticklabels(monats_namen, rotation=0, fontsize=9)
    ax_div.set_title("Durchschnittlicher Betrag pro Monat (€)", fontsize=10, pad=10)
    ax_div.set_ylabel("Euro (€)", fontsize=9)
    ax_div.set_xlabel("")
    ax_div.grid(axis='y', linestyle='--', alpha=0.3)
    ax_div.legend(title="Ticker", fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    st.pyplot(fig_div)
else:
    st.info("Keine historischen Dividenden im gewählten Zeitraum gefunden.")

st.markdown("---")

# Mean-Variance-Optimization
st.subheader("🎯 Mean-Variance-Optimization", help="10.000 Simulationen des Portfolios zur optimalen Gewichtung für das maximale Sharpe Ratio auf Basis der erwarteten Rendite.")

if len(verfuegbare) > 1:
    np.random.seed(42)
    opt_simulations = 10000
    mu_list = []
    for t in verfuegbare:
        asset_beta = renditen[t].cov(bench_rendite) / bench_rendite.var()
        expected_ret = risk_free_rate + asset_beta * (bench_cagr - risk_free_rate)
        mu_list.append(expected_ret)
    mu = np.array(mu_list)
    cov = renditen[verfuegbare].cov() * 252  
    results = np.zeros((3, opt_simulations))
    weights_record = []

    for i in range(opt_simulations):
        w = np.random.random(len(verfuegbare))
        w /= np.sum(w)
        weights_record.append(w)
        p_ret = np.sum(mu * w)
        p_std = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        
        results[0,i] = p_ret
        results[1,i] = p_std
        results[2,i] = (p_ret - risk_free_rate) / p_std
        
    max_sharpe_idx = np.argmax(results[2])
    best_w = weights_record[max_sharpe_idx]
    opt_ret = results[0, max_sharpe_idx]
    opt_vol = results[1, max_sharpe_idx]

    opt_col1, opt_col2 = st.columns([2, 1], vertical_alignment="center")

    with opt_col1:
        dynamische_hoehe = 4.1 + (len(verfuegbare) * 0.3)
        fig_ef, ax_ef = plt.subplots(figsize=(10, dynamische_hoehe))
        scatter = ax_ef.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', alpha=0.3)
        ax_ef.scatter(vola, capm_erwartung_pa, color='red', marker='o', s=200, label='Aktuelles Portfolio')
        ax_ef.scatter(opt_vol, opt_ret, color='orange', marker='o', s=200, label='Optimiertes Portfolio')
        ax_ef.set_xticklabels([f'{x*100:.0f}%' for x in ax_ef.get_xticks()])
        ax_ef.set_yticklabels([f'{y*100:.0f}%' for y in ax_ef.get_yticks()])
        ax_ef.set_xlabel('Volatilität p.a.')
        ax_ef.set_ylabel('Erwartete Rendite p.a.')
        ax_ef.legend()
        plt.colorbar(scatter, label='Sharpe Ratio')
        fig_ef.tight_layout()
        fig_ef.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
        st.pyplot(fig_ef) 
    with opt_col2:
        opt_weights_df = pd.DataFrame({
            "Ticker": verfuegbare,
            "Aktuell": [f"{a*100:.1f}%" for a in anteile],
            "Optimiert": [f"{w*100:.1f}%" for w in best_w]
        })
        st.table(opt_weights_df.set_index('Ticker'))
        st.info(f"""
        - Optimierte Erwartete Rendite: {opt_ret:.2%}
        - Optimierte Volatilität: {opt_vol:.2%}
        - Optimiertes Sharpe Ratio: {results[2, max_sharpe_idx]:.2f}
        """)
else: 
    st.info("Die Portfolio-Optimierung steht erst ab zwei Positionen zur Verfügung.")

st.markdown("---")

g_col1, g_col2 = st.columns(2)

with g_col1:
    st.subheader("⛓️ Korrelationsmatrix", help="Zeigt, wie stark sich Assets gemeinsam bewegen. 1.0 = Gleichlaufend, 0 = kein Zusammenhang, -1.0 = Gegenlaufend.")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.heatmap(renditen[verfuegbare].corr(), annot=True, cmap='RdYlGn_r', center=0.3, fmt=".2f", linewidths=0.5, ax=ax_corr)
    st.pyplot(fig_corr)

with g_col2:
    st.subheader("🚨 Risiko-Verteilung", help="Gibt an, welche Position wie stark zur Gesamtvolatilität beiträgt")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 8), constrained_layout=True)
    colors = ['#9B51E0' if x > 0 else '#22a884' for x in rel_risk_contrib]
    y_pos = np.arange(len(verfuegbare))
    bars = ax_bar.barh(y_pos, rel_risk_contrib * 100, color=colors)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(verfuegbare, fontweight='bold')
    ax_bar.invert_yaxis() 
    ax_bar.axvline(0, color='black', linewidth=1, alpha=0.5)
    ax_bar.set_xlabel('Beitrag zur Volatilität (%)', fontweight='bold')
    ax_bar.grid(axis='x', linestyle='--', alpha=0.7)
    for bar in bars:
        width = bar.get_width()
        ax_bar.text(width, bar.get_y() + bar.get_height()/2, 
                    f' {width:.1f}%', 
                    va='center', fontweight='bold', fontsize=11)
    
    st.pyplot(fig_bar)

st.markdown("---")

# Risiko-Tabelle
st.subheader("🔎 Risiko-Analyse (NTM)")
st.markdown("""
<style>
    .risk-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 20px;
    }
    .risk-card {
        background-color: rgba(100,100,100,0.1);
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #4A90E2;
        flex: 1;
        min-width: 250px;
    }
    .risk-title {
        color: #9CA3AF;
        font-size: 13px;
        margin-bottom: 12px;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .risk-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .risk-label { color: #9CA3AF; font-size: 13px; margin: 0; }
    .risk-num { color: white; font-size: 18px; font-weight: bold; margin: 0; }
</style>
""", unsafe_allow_html=True)
methoden_namen = ["Parametrisch", "Historisch", "Monte-Carlo"]
var_werte = [var_95_para, var_95_hist, mc_var_95_jahr]
es_werte = [es_95_para, es_95_hist, mc_es_95_jahr]
cards_html = ""
for m, v, e in zip(methoden_namen, var_werte, es_werte):
    cards_html += f"""
    <div class="risk-card">
        <div class="risk-title">{m}</div>
        <div class="risk-row">
            <span class="risk-label">VaR 95%</span>
            <span class="risk-num">{v:.2%}</span>
        </div>
        <div class="risk-row">
            <span class="risk-label">Exp. Shortfall</span>
            <span class="risk-num">{e:.2%}</span>
        </div>
    </div>
    """
st.write(f'<div class="risk-grid">{cards_html}</div>', unsafe_allow_html=True)
st.markdown("---")

# Monte Carlo Pfadsimulation (10 Jahre)
st.markdown("---")
st.subheader("🎲 Monte Carlo Pfadsimulation (10 Jahre)", help="Simuliert 100 mögliche Zukunftsszenarien basierend auf der historischen Volatilität und Rendite.")

mc_jahre = 10
aktuelles_jahr = pd.Timestamp.now().year
mc_tage = 252 * mc_jahre
mc_pfade = 100 
mc_startkapital = startkapital if startkapital > 0 else 10000 
mu_daily = port_rendite.mean()
sigma_daily = port_rendite.std()
np.random.seed(42)
rand_rets = np.random.normal(mu_daily, sigma_daily, (mc_tage, mc_pfade))
mc_pfade_daten = mc_startkapital * (1 + rand_rets).cumprod(axis=0)

fig_mc_path, ax_mc_path = plt.subplots(figsize=(12, 6))
zeit_achse = np.linspace(aktuelles_jahr, aktuelles_jahr + mc_jahre, mc_tage)
ax_mc_path.plot(zeit_achse, mc_pfade_daten, color='blue', alpha=0.05)
median_pfad = np.percentile(mc_pfade_daten, 50, axis=1)
top_pfad = np.percentile(mc_pfade_daten, 95, axis=1)
bottom_pfad = np.percentile(mc_pfade_daten, 5, axis=1)
mc_cagr_median = (median_pfad[-1] / mc_startkapital)**(1/mc_jahre) - 1
mc_cagr_pessimist = (bottom_pfad[-1] / mc_startkapital)**(1/mc_jahre) - 1
mc_cagr_optimist = (top_pfad[-1] / mc_startkapital)**(1/mc_jahre) - 1

ax_mc_path.plot(zeit_achse, median_pfad, color='black', linewidth=2, label='Median (50%)')
ax_mc_path.plot(zeit_achse, top_pfad, color='green', linestyle='--', label='Optimistisch (95%)')
ax_mc_path.plot(zeit_achse, bottom_pfad, color='red', linestyle='--', label='Pessimistisch (5%)')

ax_mc_path.set_title(f"Simulation von {mc_pfade} möglichen Verläufen bei {mc_startkapital:,.0f}€ Startwert")
ax_mc_path.set_xlabel("Jahr")
ax_mc_path.set_ylabel("Portfoliowert (€)")
ax_mc_path.legend(loc='upper left')
ax_mc_path.grid(True, alpha=0.2)

st.pyplot(fig_mc_path)

st.info(f"""
**Ergebnis nach {mc_jahre} Jahren (Projektion):**
- Mittleres Szenario (Median): **{median_pfad[-1]:,.2f} €** (**{mc_cagr_median:.2%} p.a.**)
- Pessimistisches Szenario (5%): **{bottom_pfad[-1]:,.2f} €** (**{mc_cagr_pessimist:.2%} p.a.**)
- Optimistisches Szenario (95%): **{top_pfad[-1]:,.2f} €** (**{mc_cagr_optimist:.2%} p.a.**)
""")

st.markdown("---")
st.subheader("🧬 Faktorenanalyse", help="Diese Analyse zeigt, welche wissenschaftlichen Faktoren (Betas) dein Portfolio antreiben.")

try:
    with st.spinner("Berechne Faktor-Exposures..."):
        factors = get_factor_loadings(port_rendite)
        fig_fac, ax_fac = plt.subplots(figsize=(12, 6))
        colors_fac = ['#4A90E2', '#27AE60', '#F2994A', '#9B51E0', '#D488FF']
        factors.index = [
            'Market', 
            'Size', 
            'Value', 
            'Quality I: Profitability', 
            'Quality II: Investmentbehavior'
        ]
        factors.plot(kind='barh', color=colors_fac, ax=ax_fac)
        ax_fac.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax_fac.set_title("Faktorladungen", fontsize=14)
        ax_fac.set_xlabel("Beta-Wert", fontsize=10)
        ax_fac.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_fac)

except Exception as e:
    st.error(f"Faktoranalyse konnte nicht geladen werden: {e}")

@st.fragment
def render_simulation_area(factors, beta, endsumme, ausgewaehlter_name):
    st.markdown("---")
    szenarien = {
        "Stagflation": [-0.35, -0.05, +0.15, +0.08, +0.05], 
        "Inflation": [-0.20, 0.00, 0.10, 0.12, 0.05],
        "Schwere Rezession": [-0.45, -0.10, +0.05, +0.15, +0.10],
        "Tech-Blase": [-0.40, -0.05, +0.35, +0.15, +0.10],
        "Small Cap Rallye": [+0.15, +0.20, +0.05, -0.05, -0.05]
    }
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        st.markdown("### 🔮 Szenario-Analyse")
        auswahl = st.selectbox(
            "Szenario:", 
            list(szenarien.keys()), 
            help="Simuliert verschiedene Szenarien unter Berücksichtigung des Fünf-Faktoren-Modells (Fama-French)."
        )
        schocks = szenarien[auswahl]
        verlust_pro_faktor = factors.values * schocks
        gesamt_sz_ret = np.sum(verlust_pro_faktor)
        verlust_sz_euro = endsumme * gesamt_sz_ret
        farbe_sz = "#EB5757" if gesamt_sz_ret < 0 else "#27AE60"
        st.markdown(f"""
            <div style="background-color: rgba(100,100,100,0.1); padding: 15px; border-radius: 10px; border-left: 5px solid {farbe_sz};">
                <p style="margin:0; font-size:14px; color:gray;">Portfolio:</p>
                <h2 style="margin:0; color:{farbe_sz};">{gesamt_sz_ret:.2%}</h2>
                <p style="margin:0; font-weight:bold;">{verlust_sz_euro:,.2f} €</p>
            </div>
        """, unsafe_allow_html=True)
    with sim_col2:
        st.markdown("### 🕹️ Benchmark-Sensitivität")
        eigener_schock = st.slider(
            "Benchmark:", 
            -50.0, 50.0, 0.0, 1.0, 
            help="Simuliert eine Bewegung der gewählten Benchmark in dem Portfolio."
        )
        gesamt_sens_ret = (beta * eigener_schock / 100)
        verlust_sens_euro = endsumme * gesamt_sens_ret
        farbe_sens = "#EB5757" if gesamt_sens_ret < 0 else "#27AE60"
        st.markdown(f"""
            <div style="background-color: rgba(100,100,100,0.1); padding: 15px; border-radius: 10px; border-left: 5px solid {farbe_sens};">
                <p style="margin:0; font-size:14px; color:gray;">Portfolio:</p>
                <h2 style="margin:0; color:{farbe_sens};">{gesamt_sens_ret:.2%}</h2>
                <p style="margin:0; font-weight:bold;">{verlust_sens_euro:,.2f} €</p>
            </div>
        """, unsafe_allow_html=True)
render_simulation_area(factors, beta, endsumme, ausgewaehlter_name)

st.markdown("---")
st.caption(f"Datenzeitraum: {daten.index[0].strftime('%d.%m.%Y')} bis {daten.index[-1].strftime('%d.%m.%Y')}")
