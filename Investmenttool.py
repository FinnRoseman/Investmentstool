import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import zipfile
import io
import statsmodels.api as sm

# --- CACHING FUNKTION ---
@st.cache_data(show_spinner="Marktdaten werden geladen...")
def get_cached_data(ticker_tuple, period):
    df = yf.download(list(ticker_tuple), period=period, progress=False)
    return df
@st.cache_data(show_spinner=False)
def get_ticker_name(t):
    try:
        return yf.Ticker(t).info.get('longName', t)
    except:
        return t
def calculate_annual_rebalancing(returns_df, target_weights):
    """
    Berechnet die Portfolio-Rendite lückenlos mit jährlichem Rebalancing.
    Berücksichtigt den Drift der Gewichte innerhalb des Jahres.
    """
    weights = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    current_weights = np.array(target_weights)
    portfolio_returns = pd.Series(index=returns_df.index, dtype=float)
    for year, year_data in returns_df.groupby(returns_df.index.year):
        cumulative_growth = (1 + year_data).cumprod()
        position_values = cumulative_growth * current_weights
        total_value = position_values.sum(axis=1)
        year_returns = total_value.pct_change()
        first_day_idx = year_data.index[0]
        year_returns.iloc[0] = (year_data.iloc[0] * current_weights).sum()        
        portfolio_returns.update(year_returns)
    return portfolio_returns
def get_factor_loadings(portfolio_returns):
    try:
        # 1. Daten-Download
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return "Fehler: Webseite von Kenneth French nicht erreichbar."
            
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(z.namelist()[0]) as f:
                ff_data = pd.read_csv(f, skiprows=3, index_col=0)
        
        # 2. Datenbereinigung
        ff_data.columns = ff_data.columns.str.strip()
        if " Annual Factors: " in ff_data.index:
            stop_idx = ff_data.index.get_loc(" Annual Factors: ")
            ff_data = ff_data.iloc[:stop_idx]
        
        ff_data = ff_data.apply(pd.to_numeric, errors='coerce')
        ff_data = ff_data.dropna()
        
        # Datums-Index sauber konvertieren
        ff_data.index = pd.to_datetime(ff_data.index.astype(str), format='%Y%m', errors='coerce')
        ff_data = ff_data.dropna()
        ff_data = ff_data / 100
        
        # --- FIX FÜR DUPLIKATE ---
        # Wir stellen sicher, dass jeder Monat im FF-Datensatz nur EINMAL vorkommt
        ff_data.index = ff_data.index.to_period('M')
        ff_data = ff_data[~ff_data.index.duplicated(keep='last')]
        
        # 3. Portfolio-Renditen vorbereiten
        port_returns = portfolio_returns.copy()
        if hasattr(port_returns.index, 'tz'):
            port_returns.index = port_returns.index.tz_localize(None)
            
        # Monatliche Aggregation
        port_monthly = port_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        port_monthly.index = port_monthly.index.to_period('M')
        
        # Sicherstellen, dass auch hier keine Duplikate durch das Resampling entstehen
        port_monthly = port_monthly[~port_monthly.index.duplicated(keep='last')]
        
        # 4. Zusammenführen (Dank .to_period('M') jetzt absolut sicher gegen Reindexing-Errors)
        combined = pd.concat([port_monthly, ff_data], axis=1).dropna()
        
        if len(combined) < 5:
            return f"Fehler: Zu wenig Datenüberschneidung ({len(combined)} Monate)."
            
        # Zurückkonvertieren für statsmodels
        combined.index = combined.index.to_timestamp()
            
        # 5. Regression (OLS)
        Y = combined.iloc[:, 0] - combined['RF']
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        X = combined[factors]
        X = sm.add_constant(X)
        
        model = sm.OLS(Y, X).fit()
        
        # --- AB HIER VISUALISIERUNG (wie zuvor) ---
        def get_stars(p):
            if p < 0.01: return "***"
            if p < 0.05: return "**"
            if p < 0.1: return "*"
            return ""

        monthly_alpha = model.params['const']
        annualized_alpha = (1 + monthly_alpha)**12 - 1
        
        fig, ax = plt.subplots(figsize=(12, 7))
        coeffs = [model.params[f] for f in factors]
        p_stars = [get_stars(model.pvalues[f]) for f in factors]
        colors = ['#2c3e50' if c >= 0 else '#e74c3c' for c in coeffs]
        
        bars = ax.bar(factors, coeffs, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=1, alpha=0.7)
        
        for bar, star in zip(bars, p_stars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.04),
                    star, ha='center', va='bottom' if height > 0 else 'top', fontsize=14)

        ax.set_title("Fama-French 5-Factor Analysis", fontsize=15, fontweight='bold')
        info_box = f"Annualized Alpha: {annualized_alpha:.2%}{get_stars(model.pvalues['const'])}\nR-Squared: {model.rsquared:.4f}"
        ax.text(0.02, 0.95, info_box, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8), family='monospace')

        plt.tight_layout()
        plt.show()

        return {"Alpha": f"{annualized_alpha:.2%}", "R2": f"{model.rsquared:.4f}"}

    except Exception as e:
        return f"Technischer Fehler in der Analyse: {str(e)}"

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# --- 1. SETUP ---
st.sidebar.header("Portfolio Zusammenstellung")
if 'widget_eingabe' not in st.session_state:
    st.session_state.widget_eingabe = ""
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
def clear_custom_input():
    if "custom_bench_input" in st.session_state:
        st.session_state["custom_bench_input"] = ""
modus = st.sidebar.radio(
    "Benchmark",
    ["Standardauswahl", "Individuelle Auswahl"],
    key="bench_mode",
    on_change=clear_custom_input
)
if modus == "Standardauswahl":
    bench_optionen = {
        "100/0 (MSCI World)": "EUNL.DE",
        "80/20 (LifeStrategy 80% Equity)": "V80A.DE",
        "60/40 (LifeStrategy 60% Equity)": "V60A.DE",
        "40/60 (LifeStrategy 40% Equity)": "V40A.DE",
        "20/80 (LifeStrategy 20% Equity)": "V20A.DE",
        "0/100 (Global Bonds)": "EUNA.DE"
    }
    auswahl = st.sidebar.selectbox(
        "Index wählen",
        options=list(bench_optionen.keys()),
        key="standard_bench_select"
    )
    benchmark = bench_optionen[auswahl]
else:
    custom_ticker = st.sidebar.text_input(
        "Ticker eingeben:",
        value="",
        placeholder="Ticker hier tippen...",
        key="custom_bench_input"
    ).strip().upper()
    benchmark = custom_ticker if custom_ticker else "EUNL.DE"

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

st.sidebar.header("Rebalancing")
rebalance_active = st.sidebar.checkbox("Jährliches Rebalancing aktivieren", value=False)

zuordnung = dict(zip(ticker_liste, anteile_orig))
st.title("Portfolio Backtest Dashboard")
with st.expander("Portfolio-Zusammensetzung (Name & Gewichtung)"):
    st.markdown("""
    <style>
        .port-container { background-color: rgba(100,100,100,0.05); border-radius: 12px; padding: 5px; margin-bottom: 15px; }
        .port-row { display: flex; align-items: center; padding: 10px 15px; border-bottom: 1px solid rgba(255,255,255,0.05); position: relative; overflow: hidden; }
        .port-row:last-child { border-bottom: none; }
        .port-bar { position: absolute; left: 0; top: 0; bottom: 0; background-color: rgba(74, 144, 226, 0.15); z-index: 0; }
        .port-ticker { background-color: rgba(0,0,0,0.3); color: #4A90E2; padding: 2px 8px; border-radius: 5px; font-family: monospace; font-size: 12px; margin-right: 15px; z-index: 1; min-width: 80px; text-align: center; }
        .port-name { flex: 1; color: white; font-size: 14px; z-index: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .port-weight { font-weight: bold; color: white; margin-left: 10px; z-index: 1; min-width: 60px; text-align: right; }
    </style>
    """, unsafe_allow_html=True)
    rows_html = ""
    ticker_namen = {t: t for t in ticker_liste} # Erstellt eine Basis-Liste
    analysis_active = st.session_state.get("run_analysis", False)
    for t in ticker_liste:
        name = get_ticker_name(t) if analysis_active else t
        ticker_namen[t] = name
        anteil_val = zuordnung.get(t, 0)
        anteil_pct = anteil_val * 100
        rows_html += f"""
        <div class="port-row">
            <div class="port-bar" style="width: {anteil_pct}%;"></div>
            <div class="port-ticker">{t}</div>
            <div class="port-name">{name}</div>
            <div class="port-weight">{anteil_pct:.1f}%</div>
        </div>"""
    st.markdown(f'<div class="port-container">{rows_html}</div>', unsafe_allow_html=True)
    summe_anteile = sum(anteile_orig)
    if abs(summe_anteile - 1.0) > 0.001:
        st.warning(f"⚠️ Die Summe der Anteile liegt bei {summe_anteile*100:.1f}%.")
    else:
        st.success("✅ Die Anteile ergeben 100%.")
if not ticker_liste:
    st.info("Das Portfolio ist gerade noch leer. Starte mit der Zusammenstellung.")
    st.stop()
if not st.session_state.get("run_analysis", False):
    st.warning("👈 Gewichtung einstellen und auf 'Go' klicken, um die Analyse zu starten.")
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
            fx_prices = fx_df['Close'].iloc[:, 0] if isinstance(fx_df.columns, pd.MultiIndex) else fx_df['Close']
            combined = pd.concat([price, fx_prices], axis=1)
            combined = combined.ffill().dropna()
            price = combined.iloc[:, 0] * combined.iloc[:, 1]
            
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

if not rebalance_active:
    port_rendite = (renditen[verfuegbare] * anteile).sum(axis=1)
else:
    port_rendite = calculate_annual_rebalancing(renditen[verfuegbare], anteile)
bench_rendite = renditen.loc[port_rendite.index, benchmark]
diff_rendite = port_rendite - bench_rendite
rf_daily = (1 + risk_free_rate)**(1/252) - 1
downside_returns = np.minimum(port_rendite - rf_daily, 0)
downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(252)
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
bench_cagr = ((1 + bench_rendite).prod())**(1/jahre) - 1
arith_mittel = port_rendite.mean() * 252
bench_arith = bench_rendite.mean() * 252
vola = port_rendite.std() * np.sqrt(252)
max_drawdown = (((1 + port_rendite).cumprod() / (1 + port_rendite).cumprod().cummax()) - 1).min()
tracking_error = diff_rendite.std() * np.sqrt(252)

sharpe_ratio = (arith_mittel - risk_free_rate) / vola
sortino_ratio = (arith_mittel - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan
beta = port_rendite.cov(bench_rendite) / bench_rendite.var()
treynor_ratio = (arith_mittel - risk_free_rate) / beta if beta != 0 else np.nan
capm_erwartung_pa = risk_free_rate + beta * (bench_arith - risk_free_rate)
alpha = arith_mittel - capm_erwartung_pa
active_return = arith_mittel - bench_arith
information_ratio = active_return / tracking_error if tracking_error != 0 else np.nan
upside_mask = bench_rendite > 0
downside_mask = bench_rendite < 0
if upside_mask.any():
    upside_ratio = (port_rendite[upside_mask].mean() / bench_rendite[upside_mask].mean()) * 100
else:
    upside_ratio = np.nan
if downside_mask.any():
    downside_ratio = (port_rendite[downside_mask].mean() / bench_rendite[downside_mask].mean()) * 100
else:
    downside_ratio = np.nan

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
            try:
                actual_fx_at_date = fx_prices.asof(date) 
                euro_zahlung_avg = (amount * actual_fx_at_date * stueckzahl_start) / anzahl_jahre
            except:
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
            ticker_yoc = last_year_divs_eur / price_at_start_eur if price_at_start_eur > 0 else 0
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

st.markdown(f"""
<div class="kpi-grid-5-row2">
    <div class="kpi-card"><p class="kpi-label">Sortino Ratio</p><p class="kpi-value">{sortino_ratio:.2f}</p></div>
    <div class="kpi-card"><p class="kpi-label">Treynor Ratio</p><p class="kpi-value">{treynor_ratio:.2f}</p></div>
    <div class="kpi-card"><p class="kpi-label">Information Ratio</p><p class="kpi-value">{information_ratio:.2f}</p></div>
    <div class="kpi-card"><p class="kpi-label">Upside Capture</p><p class="kpi-value">{upside_ratio:.1f}%</p></div>
    <div class="kpi-card"><p class="kpi-label">Downside Capture</p><p class="kpi-value">{downside_ratio:.1f}%</p></div>
</div>
""", unsafe_allow_html=True)

tab_allg, tab_rend, tab_risk, tab_sim = st.tabs([
    "🏠 Allgemein", 
    "📈 Rendite", 
    "🚨 Risiko", 
    "🎲 Simulationen"
])

# 1. Performance-Chart (Full Width oben)
with tab_allg:
    st.subheader("📈 Performance & Trends")
    port_kum = ((1 + port_rendite).cumprod() - 1) * 100
    bench_kum = ((1 + bench_rendite).cumprod() - 1) * 100
    sma100 = port_kum.rolling(window=100).mean()
    sma200 = port_kum.rolling(window=200).mean()
    df_perf_plot = pd.DataFrame({
        'Datum': port_kum.index,
        'Portfolio': port_kum.values,
        'Benchmark': bench_kum.values,
        '100-Tage-Linie': sma100.values,
        '200-Tage-Linie': sma200.values
    })
    fig_perf = px.line(
        df_perf_plot,
        x='Datum', 
        y=['Portfolio', 'Benchmark', '100-Tage-Linie', '200-Tage-Linie'],
        labels={'value': 'Entwicklung (%)', 'variable': 'Linie'}
    )
    fig_perf.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=""),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', ticksuffix='%', title=""),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    colors = {'Portfolio': '#FFD700', 'Benchmark': '#4A90E2', '100-Tage-Linie': '#F59E0B', '200-Tage-Linie': '#EF4444'}
    for name, color in colors.items():
        fig_perf.update_traces(line=dict(color=color, width=2 if 'Linie' not in name else 1), selector=dict(name=name))
    st.plotly_chart(fig_perf, use_container_width=True)
    st.markdown("---")

# 2. Spalten für Rendite-Check und Regionen-Verteilung
with tab_allg:
    st.subheader("🌎 Regionale Verteilung", help="Geografische Gewichtung des Portfolios.")
    reg_labels = ['Nordamerika', 'Südamerika', 'Europa', 'Asien-Pazifik', 'Afrika']
    reg_values = [total_na, total_sa, total_eu, total_ap, total_af]
    reg_colors = ['#00E6FF', '#CC00FF', '#FF9900', '#39FF14', '#FF3131']
    labels_f = [l for l, v in zip(reg_labels, reg_values) if v > 0]
    values_f = [v for v in reg_values if v > 0]
    colors_f = [c for c, v in zip(reg_colors, reg_values) if v > 0]
    
    if sum(values_f) > 0:
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels_f, 
            values=values_f, 
            hole=.6,
            marker=dict(colors=colors_f, line=dict(color='#1E1E1E', width=2)),
            textinfo='percent',
            hoverinfo='label+percent',
            insidetextorientation='radial',
            opacity=1
        )])
        fig_donut.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            hoverlabel=dict(bgcolor="#222", font_size=14, font_family="monospace"),
            height=450
        )
        fig_donut.update_traces(
            hoverinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Anteil: %{percent}<extra></extra>"
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Daten in Sidebar eintragen.")
    st.markdown("---")

with tab_rend:
    st.subheader("💰 Rendite-Analyse", help="Jährliche Renditen und die kumulierte Rendite über feste Zeiträume.")
    yearly_ret = port_rendite.groupby(port_rendite.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    periods = {"1Y": 252, "3Y": 756, "5Y": 1260, "10Y": 2520, "20Y": 5040}
    period_rets = {label: ((1 + port_rendite.iloc[-days:]).prod() - 1) * 100 
                   for label, days in periods.items() if len(port_rendite) >= days}
    colors_y = ['#39FF14' if x > 0 else '#FF3131' for x in yearly_ret.values]
    colors_p = ['#39FF14' if x > 0 else '#FF3131' for x in period_rets.values()]
    fig_yearly = px.bar(
        x=yearly_ret.index.astype(str), 
        y=yearly_ret.values,
        labels={'x': 'Jahr', 'y': 'Rendite (%)'}
    )
    fig_yearly.update_traces(marker_color=colors_y)
    fig_periods = px.bar(
        x=list(period_rets.keys()), 
        y=list(period_rets.values()),
        labels={'x': 'Zeitraum', 'y': 'Kumuliert (%)'}
    )
    fig_periods.update_traces(marker_color=colors_p)
    for fig in [fig_yearly, fig_periods]:
        fig.update_layout(
            template='plotly_dark', 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

# Rolling Returns & Rendite Verteilung
with tab_rend:
    att_col1, att_col2 = st.columns(2)
    with att_col1:
        st.subheader("🔄 Rolling Returns (12 Monate)", help="Zeigt die Rendite eines Zeitpunktes im Vergleich zum Vorjahr.")
        rolling_1y = port_rendite.rolling(window=252).apply(lambda x: (1 + x).prod() - 1) * 100
        rolling_mean = rolling_1y.mean()
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=rolling_1y.index, y=rolling_1y,
            mode='lines',
            line=dict(color='#00E6FF', width=2),
            name='Rolling Return',
            fill='tozeroy',
            fillcolor='rgba(0, 230, 255, 0.1)'
        ))
        fig_roll.add_trace(go.Scatter(
            x=rolling_1y.index, y=[rolling_mean]*len(rolling_1y),
            mode='lines',
            line=dict(color='#FF3131', width=1.5, dash='dash'),
            name=f'Schnitt ({rolling_mean:.1f}%)'
        ))
        fig_roll.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            hovermode='x unified',
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', ticksuffix='%', title=""),
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig_roll, use_container_width=True)
    with att_col2:
        st.subheader("📊 Rendite-Verteilung", help="Beitrag jeder Position zur Gesamtrendite.")
        beitraege = []
        for t in verfuegbare:
            einzel_ret = (1 + renditen[t]).prod() - 1
            gewicht = anteile[verfuegbare.index(t)]
            beitraege.append(einzel_ret * gewicht * 100)
        df_att = pd.DataFrame({
            'Ticker': verfuegbare,
            'Name': [ticker_namen.get(t, t) for t in verfuegbare],
            'Beitrag': beitraege
        }).sort_values('Beitrag', ascending=True)
        colors_att = ['#39FF14' if x > 0 else '#FF3131' for x in df_att['Beitrag']]
        fig_att = px.bar(
            df_att, 
            x='Beitrag', 
            y='Ticker',
            orientation='h',
            hover_data={'Name': True, 'Ticker': False, 'Beitrag': ':.2f'},
            text='Beitrag'
        )
        fig_att.update_traces(
            marker_color=colors_att,
            texttemplate='%{text:.1f}%', 
            textposition='outside',
            cliponaxis=False 
        )
        fig_att.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=40, t=20, b=0), 
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', ticksuffix='%', title=""),
            yaxis=dict(showgrid=False, title=""),
            height=350
        )
        st.plotly_chart(fig_att, use_container_width=True)

# Dividendenkalender
with tab_allg:
    st.subheader("📅 Dividenden-Kalender", help="Zeigt die monatlich erwarteten Ausschüttungen, gestapelt nach Positionen.")
    if cal_data:
        df_cal = pd.DataFrame(cal_data)
        monats_namen = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
        df_div_plot = df_cal.groupby(['Monat_Nr', 'Ticker'])['Ausschüttung'].sum().reset_index()
        df_div_plot['Monat'] = df_div_plot['Monat_Nr'].apply(lambda x: monats_namen[x-1])
        fig_div = px.bar(
            df_div_plot,
            x='Monat',
            y='Ausschüttung',
            color='Ticker',
            category_orders={"Monat": monats_namen},
            color_discrete_sequence=px.colors.qualitative.G10, 
            text_auto='.2f', 
        )
        fig_div.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            height=350,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title=None
            ),
            xaxis=dict(
                title=None, 
                showgrid=False, 
                fixedrange=True
            ),
            yaxis=dict(
                title="Ausschüttung in €",
                gridcolor='rgba(255,255,255,0.05)',
                fixedrange=True,
                ticksuffix=" €"
            ),
            hovermode="x unified",
            bargap=0.3 
        )
        fig_div.update_traces(
            marker_line_width=1, 
            marker_line_color="rgba(0,0,0,0.5)",
            hovertemplate="<b>%{fullData.name}</b>: %{y:.2f} €<extra></extra>"
        )
        st.plotly_chart(fig_div, use_container_width=True, config={'displayModeBar': False})
        total_div = df_cal['Ausschüttung'].sum()
        st.caption(f"💸 Erwartete Gesamt-Dividende (p.a.): **{total_div:.2f} €**")
    
    else:
        st.info("Keine historischen Dividenden im gewählten Zeitraum gefunden.")

# Mean-Variance-Optimization
with tab_sim:
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
            df_sim = pd.DataFrame({
                'Volatilität': results[1, :],
                'Rendite': results[0, :],
                'Sharpe Ratio': results[2, :]
            })
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=df_sim['Volatilität'],
                y=df_sim['Rendite'],
                mode='markers',
                marker=dict(
                    color=df_sim['Sharpe Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Sharpe",
                        thickness=15,
                        len=0.6,
                        yanchor="top",
                        y=1
                    ),
                    opacity=0.3,
                    size=5
                ),
                name='Simulationen',
                hoverinfo='skip' 
            ))
            fig_ef.add_trace(go.Scatter(
                x=[vola], 
                y=[capm_erwartung_pa], 
                mode='markers',
                marker=dict(
                    color='white', 
                    size=20, 
                    symbol='circle',
                    line=dict(color='black', width=3) 
                ),
                name='Aktuelles Portfolio',
                hovertemplate="<b>Aktuelles Portfolio</b><br>Vola: %{x:.2%}<br>Rendite: %{y:.2%}<extra></extra>"
            ))
            fig_ef.add_trace(go.Scatter(
                x=[opt_vol], 
                y=[opt_ret],
                mode='markers',
                marker=dict(
                    color='#4A90E2', 
                    size=22, 
                    symbol='star', 
                    line=dict(color='white', width=2)
                ),
                name='Optimiertes Portfolio',
                hovertemplate="<b>Max Sharpe Portfolio</b><br>Vola: %{x:.2%}<br>Rendite: %{y:.2%}<extra></extra>"
            ))
            fig_ef.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.05)',
                    tickformat='.0%', 
                    title="Risiko (Volatilität p.a.)",
                    autorange=True
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.05)',
                    tickformat='.0%', 
                    title="Erwartete Rendite p.a.",
                    autorange=True
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="right", 
                    x=1,
                    bgcolor="rgba(0,0,0,0)"
                ),
                height=500 
            )
            st.plotly_chart(fig_ef, use_container_width=True)
        with opt_col2:
            st.markdown("""
            <style>
                .opt-container { background: rgba(100,100,100,0.05); border-radius: 12px; padding: 10px; margin-bottom: 12px; }
                .opt-row { margin-bottom: 10px; position: relative; }
                .opt-ticker-label { color: #4A90E2; font-family: monospace; font-weight: bold; font-size: 13px; margin-bottom: 3px; }
                .opt-name-label { color: #9CA3AF; font-size: 11px; margin-bottom: 4px; display: block; }
                .bar-bg-stacked { 
                    background: rgba(255,255,255,0.05); height: 10px; border-radius: 5px; 
                    width: 100%; display: flex; overflow: hidden; margin-bottom: 4px;
                }
                .bar-segment-act { background: #6B7280; }
                .bar-segment-opt { background: #4A90E2; }
                .values-row { display: flex; justify-content: space-between; font-size: 11px; color: #9CA3AF; }
                .val-act { color: #D1D5DB; }
                .val-opt { color: #4A90E2; font-weight: bold; }
            </style>
            """, unsafe_allow_html=True)  
            opt_html = '<div class="opt-container">'
            for t, a, w in zip(verfuegbare, anteile, best_w):
                full_name = ticker_namen.get(t, t) 
                opt_html += f"""
                <div class="opt-row">
                    <div class="opt-ticker-label">{t}</div>
                    <div class="opt-name-label">{full_name}</div>
                    <div class="bar-bg-stacked">
                        <div class="bar-segment-act" style="width: {a*100}%;"></div>
                        <div class="bar-segment-opt" style="width: {w*100}%;"></div>
                    </div>
                    <div class="values-row">
                        <span>Aktuell <span class="val-act">{a:.1%}</span></span>
                        <span><span class="val-opt">{w:.1%}</span> Optimiert</span>
                    </div>
                </div>"""
            opt_html += "</div>"
            st.markdown(opt_html, unsafe_allow_html=True)
    else: 
        st.info("Die Portfolio-Optimierung steht erst ab zwei Positionen zur Verfügung.")
    st.markdown("---")

# Korrelationsmatrix und Risikoverteilung
with tab_risk:
    g_col1, g_col2 = st.columns(2)   
    with g_col1:
        @st.fragment
        def render_correlation_section():
            if "corr_mode_state" not in st.session_state:
                st.session_state.corr_mode_state = "Matrix (Statisch)"
            if st.session_state.corr_mode_state == "Matrix (Statisch)":
                st.subheader("⛓️ Korrelationsmatrix", help="Zeigt, wie stark sich Assets gemeinsam bewegen.")
                corr_matrix = renditen[verfuegbare].corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdYlGn', 
                    reversescale=True,
                    zmin=-1, zmax=1,
                    xgap=2, ygap=2,
                    hovertemplate="Ticker A: %{x}<br>Ticker B: %{y}<br>Korrelation: <b>%{z:.2f}</b><extra></extra>",
                    showscale=True
                ))
                for i, row in enumerate(corr_matrix.values):
                    for j, value in enumerate(row):
                        fig_corr.add_annotation(
                            x=corr_matrix.columns[j], y=corr_matrix.columns[i],
                            text=f"{value:.2f}", showarrow=False,
                            font=dict(color="white" if abs(value) > 0.7 else "black")
                        )
                fig_corr.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=450,
                    xaxis=dict(fixedrange=True, side="bottom"),
                    yaxis=dict(fixedrange=True, autorange="reversed")
                )
                st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False}, key="mx_plot")   
            else:
                st.subheader("🔁 Rollierende Korrelation", help="Zeigt die Korrelation des Portfolios zur Benchmark über ein 126-Tage-Fenster (6 Monate).")
                window = 126
                rolling_corr = port_rendite.rolling(window=window).corr(bench_rendite).dropna()
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    line=dict(color='#27AE60', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(39, 174, 96, 0.1)',
                    name="Roll. Korrelation"
                ))
                fig_roll.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[-1, 1], title="Korrelation", gridcolor='rgba(255,255,255,0.05)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    height=450,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_roll, use_container_width=True, config={'displayModeBar': False}, key="roll_plot")
            c1, c2 = st.columns(2)
            if c1.button("⛓️ Matrix", use_container_width=True):
                st.session_state.corr_mode_state = "Matrix (Statisch)"
                st.rerun(scope="fragment")
            if c2.button("🔁 Rollierend", use_container_width=True):
                st.session_state.corr_mode_state = "Zeitverlauf (Rollierend)"
                st.rerun(scope="fragment")
        render_correlation_section()
    with g_col2:
        st.subheader("🚨 Risiko-Verteilung", help="Gibt an, welche Position wie stark zur Gesamtvolatilität beiträgt")
        risk_data = pd.DataFrame({
            'Ticker': verfuegbare,
            'Beitrag': rel_risk_contrib * 100
        }).sort_values('Beitrag', ascending=True)
        risk_data['Farbe'] = risk_data['Beitrag'].apply(lambda x: '#9B51E0' if x > 0 else '#22a884')
        fig_bar = px.bar(
            risk_data, x='Beitrag', y='Ticker',
            orientation='h',
            text=risk_data['Beitrag'].apply(lambda x: f'{x:.1f}%'),
            template='plotly_dark'
        )
        fig_bar.update_traces(
            marker_color=risk_data['Farbe'],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Risiko-Beitrag: %{x:.2f}%<extra></extra>",
            width=0.7
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Beitrag zur Volatilität (%)", showgrid=True, gridcolor='rgba(255,255,255,0.05)', zerolinecolor='white'),
            yaxis=dict(title=None),
            margin=dict(l=0, r=50, t=10, b=0),
            height=450,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
    st.markdown("---")

# Risiko-Tabelle
with tab_risk:
    st.subheader("🔎 Risiko-Analyse (NTM)")
    r_labels = ["Parametrisch", "Historisch", "Monte-Carlo"]
    r_vars = [var_95_para, var_95_hist, mc_var_95_jahr]
    r_ess = [es_95_para, es_95_hist, mc_es_95_jahr]
    st.markdown("""
    <style>
    .r-grid { display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
    .r-card { 
        background-color: rgba(100,100,100,0.1); 
        padding: 18px; 
        border-radius: 12px; 
        border-left: 5px solid #4A90E2; 
        flex: 1; 
        min-width: 200px; 
    }
    .r-title { color: #9CA3AF; font-size: 13px; margin-bottom: 12px; text-transform: uppercase; font-weight: bold; }
    .r-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
    .r-lbl { color: #9CA3AF; font-size: 13px; }
    .r-val { color: white; font-size: 18px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    cards_html = ""
    for m, v, e in zip(r_labels, r_vars, r_ess):
        cards_html += f'<div class="r-card"><div class="r-title">{m}</div>'
        cards_html += f'<div class="r-row"><span class="r-lbl">VaR 95%</span><span class="r-val">{v:.2%}</span></div>'
        cards_html += f'<div class="r-row"><span class="r-lbl">Exp. Shortfall</span><span class="r-val">{e:.2%}</span></div></div>'
    st.markdown(f'<div class="r-grid">{cards_html}</div>', unsafe_allow_html=True)

# Monte Carlo Pfadsimulation (10 Jahre)
with tab_sim:
    st.subheader("🎲 Monte-Carlo-Simulation", help="Simuliert 100 mögliche Pfade der Vermögensentwicklung basierend auf der historischen Volatilität und Rendite.")
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
    median_pfad = np.percentile(mc_pfade_daten, 50, axis=1)
    top_pfad = np.percentile(mc_pfade_daten, 95, axis=1)
    bottom_pfad = np.percentile(mc_pfade_daten, 5, axis=1)
    mc_cagr_median = (median_pfad[-1] / mc_startkapital)**(1/mc_jahre) - 1
    mc_cagr_pessimist = (bottom_pfad[-1] / mc_startkapital)**(1/mc_jahre) - 1
    mc_cagr_optimist = (top_pfad[-1] / mc_startkapital)**(1/mc_jahre) - 1
    zeit_achse = np.linspace(aktuelles_jahr, aktuelles_jahr + mc_jahre, mc_tage)
    fig_mc_path = go.Figure()
    for i in range(mc_pfade):
        fig_mc_path.add_trace(go.Scatter(
            x=zeit_achse, y=mc_pfade_daten[:, i],
            mode='lines', line=dict(color='rgba(135, 206, 250, 0.1)', width=1),
            showlegend=False, hoverinfo='skip' 
        ))
    fig_mc_path.add_trace(go.Scatter(x=zeit_achse, y=top_pfad, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_mc_path.add_trace(go.Scatter(x=zeit_achse, y=bottom_pfad, mode='lines', fill='tonexty', 
                                     fillcolor='rgba(255, 255, 255, 0.03)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_mc_path.add_trace(go.Scatter(x=zeit_achse, y=bottom_pfad, mode='lines', name='Pessimistisch (5%)',
                                     line=dict(color='#FF3131', width=2, dash='dash'),
                                     hovertemplate="Jahr: %{x:.1f}<br>Wert: <b>%{y:,.0f} €</b><extra></extra>"))
    fig_mc_path.add_trace(go.Scatter(x=zeit_achse, y=top_pfad, mode='lines', name='Optimistisch (95%)',
                                     line=dict(color='#39FF14', width=2, dash='dash'),
                                     hovertemplate="Jahr: %{x:.1f}<br>Wert: <b>%{y:,.0f} €</b><extra></extra>"))
    fig_mc_path.add_trace(go.Scatter(x=zeit_achse, y=median_pfad, mode='lines', name='Median (50%)',
                                     line=dict(color='#4A90E2', width=4),
                                     hovertemplate="Jahr: %{x:.1f}<br>Wert: <b>%{y:,.0f} €</b><extra></extra>"))
    fig_mc_path.update_layout(
        title=dict(text=f"Simulation von {mc_pfade} Pfaden bei {mc_startkapital:,.0f}€ Startwert", font=dict(color='white', size=16), x=0, y=0.95),
        template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0), height=500,
        xaxis=dict(title="Jahr", gridcolor='rgba(255,255,255,0.05)', tickformat='.0f', dtick=2),
        yaxis=dict(title="Portfoliowert (€)", gridcolor='rgba(255,255,255,0.05)', ticksuffix=" €", tickformat=',.0f'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_mc_path, use_container_width=True, config={'displayModeBar': False})
    st.markdown("""
    <style>
        .mc-container { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
        .mc-card { background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; border-left: 5px solid #4A90E2; display: flex; justify-content: space-between; align-items: center; }
        .mc-label { color: #9CA3AF; font-size: 14px; font-weight: 500; }
        .mc-amount { display: block; color: white; font-size: 18px; font-weight: bold; }
        .mc-cagr { color: #4A90E2; font-size: 13px; font-weight: bold; }
        .pessimist { border-left-color: #E74C3C; }
        .median { border-left-color: #4A90E2; }
        .optimist { border-left-color: #27AE60; }
    </style>
    """, unsafe_allow_html=True)
    mc_html = f"""
    <div style="margin-bottom: 10px; font-weight: bold; color: #4A90E2;">⏳ Ergebnis nach {mc_jahre} Jahren (Projektion):</div>
    <div class="mc-container">
        <div class="mc-card optimist"><span class="mc-label">🚀 Optimistisch (95%)</span><div class="mc-value"><span class="mc-amount">{top_pfad[-1]:,.2f} €</span><span class="mc-cagr" style="color: #27AE60;">{mc_cagr_optimist:.2%} p.a.</span></div></div>
        <div class="mc-card median"><span class="mc-label">📈 Median (50%)</span><div class="mc-value"><span class="mc-amount">{median_pfad[-1]:,.2f} €</span><span class="mc-cagr">{mc_cagr_median:.2%} p.a.</span></div></div>
        <div class="mc-card pessimist"><span class="mc-label">📉 Pessimistisch (5%)</span><div class="mc-value"><span class="mc-amount">{bottom_pfad[-1]:,.2f} €</span><span class="mc-cagr" style="color: #E74C3C;">{mc_cagr_pessimist:.2%} p.a.</span></div></div>
    </div>
    """
    st.markdown(mc_html, unsafe_allow_html=True)
    st.markdown("---")

with tab_allg:
    st.markdown("---")
    st.subheader("🧬 Faktorenanalyse", help="Diese Analyse zeigt, welche wissenschaftlichen Faktoren (Betas) dein Portfolio antreiben.")
    try:
        with st.spinner("Berechne Faktor-Exposures..."):
            factors = get_factor_loadings(port_rendite)
            if isinstance(factors, pd.Series):
                factor_names = [
                    'Market', 
                    'Size', 
                    'Value', 
                    'Quality I: Profitability', 
                    'Quality II: Investmentbehavior'
                ]
                colors_fac = ['#4A90E2', '#27AE60', '#F2994A', '#9B51E0', '#D488FF']
                fig_fac = go.Figure(go.Bar(
                    x=factors.values,
                    y=factor_names,
                    orientation='h',
                    marker_color=colors_fac, 
                    hovertemplate="Faktor: <b>%{y}</b><br>Beta: <b>%{x:.3f}</b><extra></extra>",
                    width=0.6
                ))
                fig_fac.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=20, t=10, b=10),
                    height=350, 
                    xaxis=dict(
                        title="Beta-Wert", 
                        gridcolor='rgba(255,255,255,0.05)', 
                        zerolinecolor='white',
                        zerolinewidth=1
                    ),
                    yaxis=dict(
                        autorange="reversed", 
                        fixedrange=True
                    )
                )
                st.plotly_chart(fig_fac, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info(f"💡 {factors}")
    except Exception as e:
        st.error(f"Faktoranalyse konnte nicht geladen werden: {e}")

# --- INHALT FÜR TAB: SIMULATIONEN ---
with tab_sim:
    @st.fragment
    def render_simulation_area(factors, beta, endsumme):
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
                "Szenario wählen:", 
                list(szenarien.keys()), 
                help="Simuliert verschiedene Marktszenarien basierend auf dem Fama-French Fünf-Faktoren-Modell."
            )
            schocks = szenarien[auswahl]
            verlust_pro_faktor = factors.values * schocks
            gesamt_sz_ret = np.sum(verlust_pro_faktor)
            verlust_sz_euro = endsumme * gesamt_sz_ret
            farbe_sz = "#EB5757" if gesamt_sz_ret < 0 else "#27AE60"
            st.markdown(f"""
                <div style="background-color: rgba(100,100,100,0.1); padding: 15px; border-radius: 10px; border-left: 5px solid {farbe_sz};">
                    <p style="margin:0; font-size:14px; color:gray;">Erwartete Portfolio-Reaktion:</p>
                    <h2 style="margin:0; color:{farbe_sz};">{gesamt_sz_ret:.2%}</h2>
                    <p style="margin:0; font-weight:bold;">{verlust_sz_euro:,.2f} €</p>
                </div>
            """, unsafe_allow_html=True)    
        with sim_col2:
            st.markdown("### 🕹️ Benchmark-Sensitivität")
            eigener_schock = st.slider(
                f"Bewegung Benchmark (%):", 
                -50.0, 50.0, 0.0, 1.0, 
                help="Simuliert die Auswirkung einer Benchmark-Bewegung basierend auf dem Portfolio-Beta."
            )
            gesamt_sens_ret = (beta * eigener_schock / 100)
            verlust_sens_euro = endsumme * gesamt_sens_ret
            farbe_sens = "#EB5757" if gesamt_sens_ret < 0 else "#27AE60"
            st.markdown(f"""
                <div style="background-color: rgba(100,100,100,0.1); padding: 15px; border-radius: 10px; border-left: 5px solid {farbe_sens};">
                    <p style="margin:0; font-size:14px; color:gray;">Auswirkung auf Portfolio:</p>
                    <h2 style="margin:0; color:{farbe_sens};">{gesamt_sens_ret:.2%}</h2>
                    <p style="margin:0; font-weight:bold;">{verlust_sens_euro:,.2f} €</p>
                </div>
            """, unsafe_allow_html=True)
    render_simulation_area(factors, beta, endsumme)
st.markdown("---")
st.caption(f"Datenzeitraum: {daten.index[0].strftime('%d.%m.%Y')} bis {daten.index[-1].strftime('%d.%m.%Y')}")
