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
    Verhindert Datenverlust an Jahresübergängen und berücksichtigt Weight Drift.
    """
    n_assets = len(target_weights)
    target_weights = np.array(target_weights)
    portfolio_returns = pd.Series(index=returns_df.index, dtype=float)
    current_weights = target_weights.copy()
    for i in range(len(returns_df)):
        daily_asset_returns = returns_df.iloc[i].values
        day_return = np.sum(current_weights * daily_asset_returns)
        portfolio_returns.iloc[i] = day_return
        drifted_weights = current_weights * (1 + daily_asset_returns)
        current_weights = drifted_weights / np.sum(drifted_weights)
        if i + 1 < len(returns_df):
            if returns_df.index[i+1].year > returns_df.index[i].year:
                current_weights = target_weights.copy()              
    return portfolio_returns

def calculate_buy_and_hold(returns_df, start_weights):
    """
    Berechnet Portfolio-Rendite bei reinem Buy & Hold.
    Startgewichte gelten nur am ersten Tag — danach driften die Positionen
    frei je nach ihrer individuellen Kursentwicklung, ohne Rebalancing.
    Gibt die täglichen Portfolio-Renditen und die finalen gedrifteten Gewichte zurück.
    """
    weights = np.array(start_weights)
    cum_returns = (1 + returns_df).cumprod()
    position_values = cum_returns.multiply(weights, axis=1)
    port_value = position_values.sum(axis=1)
    port_returns = pd.Series(index=returns_df.index, dtype=float)
    port_returns.iloc[0] = (returns_df.iloc[0].values * weights).sum()
    if len(returns_df) > 1:
        port_returns.iloc[1:] = (port_value.iloc[1:].values / port_value.iloc[:-1].values) - 1
    final_weights = (position_values.iloc[-1] / port_value.iloc[-1]).values
    return port_returns, final_weights

def get_factor_loadings(portfolio_returns):
    try:
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return "Fehler: Webseite von Kenneth French nicht erreichbar."           
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(z.namelist()[0]) as f:
                ff_data = pd.read_csv(f, skiprows=3, index_col=0)   
        ff_data.columns = ff_data.columns.str.strip()
        ff_data.index = ff_data.index.astype(str).str.strip()
        ff_data = ff_data[ff_data.index.str.len() == 6]       
        ff_data = ff_data.apply(pd.to_numeric, errors='coerce')
        ff_data = ff_data.dropna()
        ff_data.index = pd.to_datetime(ff_data.index, format='%Y%m', errors='coerce')
        ff_data = ff_data.dropna()
        ff_data = ff_data / 100    
        port_returns = portfolio_returns.copy()
        if hasattr(port_returns.index, 'tz'):
            port_returns.index = port_returns.index.tz_localize(None)      
        port_monthly = port_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        ff_data.index = ff_data.index.strftime('%Y-%m')
        port_monthly.index = port_monthly.index.strftime('%Y-%m')   
        ff_data = ff_data.groupby(level=0).last()
        port_monthly = port_monthly.groupby(level=0).last() 
        combined = pd.concat([port_monthly, ff_data], axis=1).dropna()    
        if len(combined) < 5:
            return f"Fehler: Zu wenig Datenüberschneidung ({len(combined)} Monate)."          
        Y = combined.iloc[:, 0] - combined['RF']
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        X = combined[factors]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        alpha_monthly = model.params['const']
        alpha_annualized = (1 + alpha_monthly)**12 - 1   
        return {
            "loadings": model.params[factors],
            "r_squared": model.rsquared,
            "p_values": model.pvalues[factors], 
            "annualized_alpha": alpha_annualized, 
            "alpha_p_value": model.pvalues['const'] 
        }
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
if 'asset_typen' not in st.session_state:
    st.session_state.asset_typen = {}
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
            st.markdown("**Asset-Typ (für Szenario-Analyse)**")
            _atyp_optionen = ["Aktie", "Anleihe", "Rohstoff / Edelmetall", "Kryptowährung"]
            _atyp_default  = st.session_state.asset_typen.get(t, {}).get("typ", "Aktie")
            _atyp_idx      = _atyp_optionen.index(_atyp_default) if _atyp_default in _atyp_optionen else 0
            _asset_typ     = st.selectbox("Typ", _atyp_optionen, index=_atyp_idx, key=f"atyp_{t}")
            _asset_info    = {"typ": _asset_typ}
            if _asset_typ == "Anleihe":
                _dur_default  = st.session_state.asset_typen.get(t, {}).get("duration", 5.0)
                _duration     = st.number_input("Duration (Jahre)", 0.0, 30.0, _dur_default, step=0.5, key=f"dur_{t}",
                                                help="Modifizierte Duration der Anleihe / des Anleihen-ETFs.")
                _btype_opts   = ["Staatsanleihe", "Unternehmensanleihe (IG)", "Unternehmensanleihe (HY)", "Emerging Markets"]
                _btype_def    = st.session_state.asset_typen.get(t, {}).get("bond_type", "Staatsanleihe")
                _btype_idx    = _btype_opts.index(_btype_def) if _btype_def in _btype_opts else 0
                _bond_type    = st.selectbox("Anleihen-Typ", _btype_opts, index=_btype_idx, key=f"btype_{t}")
                _asset_info["duration"]  = _duration
                _asset_info["bond_type"] = _bond_type
            elif _asset_typ == "Rohstoff / Edelmetall":
                _rohstoff_opts = ["Gold", "Silber", "Rohöl (WTI)", "Rohöl (Brent)", "Erdgas", "Weizen", "Kupfer"]
                _rohstoff_def  = st.session_state.asset_typen.get(t, {}).get("rohstoff", "Gold")
                _rohstoff_idx  = _rohstoff_opts.index(_rohstoff_def) if _rohstoff_def in _rohstoff_opts else 0
                _rohstoff      = st.selectbox("Rohstoff-Typ", _rohstoff_opts, index=_rohstoff_idx, key=f"rohst_{t}")
                _asset_info["rohstoff"] = _rohstoff
            elif _asset_typ == "Kryptowährung":
                _crypto_opts = ["Bitcoin", "Ethereum", "Altcoin (Large Cap)", "Altcoin (Small Cap)"]
                _crypto_def  = st.session_state.asset_typen.get(t, {}).get("crypto", "Bitcoin")
                _crypto_idx  = _crypto_opts.index(_crypto_def) if _crypto_def in _crypto_opts else 0
                _crypto      = st.selectbox("Krypto-Typ", _crypto_opts, index=_crypto_idx, key=f"crypto_{t}",
                                            help="Large Cap Altcoins: z.B. SOL, ADA, XRP. Small Cap: höheres Beta.")
                _asset_info["crypto"] = _crypto
            st.session_state.asset_typen[t] = _asset_info
            st.markdown("**Kosten (TER)**")
            _ter_default = st.session_state.get(f"ter_val_{t}", 0.0)
            _ter = st.number_input("TER p.a. (%)", 0.0, 10.0, _ter_default, step=0.01, format="%.2f", key=f"ter_{t}",
                                   help="Total Expense Ratio des ETFs/Fonds. Für Einzelaktien, Anleihen oder Krypto: 0 lassen.")
            st.session_state[f"ter_val_{t}"] = _ter

st.sidebar.markdown("---")
go_button = st.sidebar.button("Go", use_container_width=True)
if go_button:
    st.session_state.run_analysis = True

def get_rf_rate():
    try:
        ticker = yf.Ticker("^ESTRON")
        current_val = ticker.history(period="1d")['Close'].iloc[-1]
        return current_val / 100
    except:
        return 0.035
        
risk_free_rate = get_rf_rate()

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
rebalance_active = st.sidebar.checkbox("Jährliches Rebalancing", value=False)

zuordnung = dict(zip(ticker_liste, anteile_orig))
st.title("Portfolio Backtest Dashboard")
st.markdown("""
<style>
    :root {
        --tv: var(--text-color, inherit);
        --ts: var(--text-color, inherit);
        --tm: var(--text-color, inherit);
        --bg-badge: rgba(128,128,128,0.15);
        --border-subtle: rgba(128,128,128,0.15);
        --bg-card: rgba(128,128,128,0.08);
    }
    .ts-opacity { opacity: 0.55; }
    .tm-opacity { opacity: 0.7; }
</style>
""", unsafe_allow_html=True)
with st.expander("Portfoliozusammensetzung — Positionen & Startgewicht"):
    st.caption("Die hier gezeigten Gewichte sind die eingegebenen Startwerte. Die aktuelle Gewichtung (Rebalancing ja/nein) findet sich weiter unten.")
    st.markdown("""
    <style>
        .port-container { background-color: rgba(100,100,100,0.05); border-radius: 12px; padding: 5px; margin-bottom: 15px; }
        .port-row { display: flex; align-items: center; padding: 10px 15px; border-bottom: 1px solid var(--border-subtle); position: relative; overflow: hidden; }
        .port-row:last-child { border-bottom: none; }
        .port-bar { position: absolute; left: 0; top: 0; bottom: 0; background-color: rgba(74, 144, 226, 0.15); z-index: 0; }
        .port-ticker { background-color: var(--bg-badge); color: #4A90E2; padding: 2px 8px; border-radius: 5px; font-family: monospace; font-size: 12px; margin-right: 15px; z-index: 1; min-width: 80px; text-align: center; }
        .port-name { flex: 1; color: var(--tv); font-size: 14px; z-index: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .port-weight { font-weight: bold; color: var(--tv); margin-left: 10px; z-index: 1; min-width: 60px; text-align: right; }
    </style>
    """, unsafe_allow_html=True)
    rows_html = ""
    ticker_namen = {t: t for t in ticker_liste} 
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
fx_prices_map = {}
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
            fx_prices_map[t] = fx_prices
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
    port_rendite, aktuelle_gewichte = calculate_buy_and_hold(renditen[verfuegbare], anteile)
else:
    port_rendite = calculate_annual_rebalancing(renditen[verfuegbare], anteile)
    aktuelle_gewichte = list(anteile)
port_ter = sum(st.session_state.get(f"ter_val_{t}", 0.0) * aktuelle_gewichte[i] for i, t in enumerate(verfuegbare))
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
port_excess = port_rendite - rf_daily
bench_excess = bench_rendite - rf_daily
sharpe_ratio = (arith_mittel - risk_free_rate) / vola
sortino_ratio = (arith_mittel - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan
beta = port_excess.cov(bench_excess) / bench_excess.var()
treynor_ratio = (arith_mittel - risk_free_rate) / beta if beta != 0 else np.nan
capm_erwartung_pa = risk_free_rate + beta * (bench_arith - risk_free_rate)
alpha = arith_mittel - capm_erwartung_pa
active_return = arith_mittel - bench_arith
information_ratio = active_return / tracking_error if tracking_error != 0 else np.nan
port_monthly = port_rendite.resample('ME').apply(lambda x: (1 + x).prod() - 1)
bench_monthly = bench_rendite.resample('ME').apply(lambda x: (1 + x).prod() - 1)
upside_mask = bench_monthly > 0
downside_mask = bench_monthly < 0
if upside_mask.any():
    upside_ratio = (port_monthly[upside_mask].mean() / bench_monthly[upside_mask].mean()) * 100
else:
    upside_ratio = np.nan
if downside_mask.any():
    downside_ratio = (port_monthly[downside_mask].mean() / bench_monthly[downside_mask].mean()) * 100
else:
    downside_ratio = np.nan

# Risikokennzahlen
var_95_para = -(arith_mittel - 1.645 * vola)
rolling_1y_rets = port_rendite.rolling(252).apply(lambda x: (1 + x).prod() - 1)
var_95_hist = abs(rolling_1y_rets.dropna().quantile(0.05))
es_95_para = -(arith_mittel - 2.063 * vola)
es_95_tag_hist = rolling_1y_rets.dropna().quantile(0.05)
es_95_hist = abs(rolling_1y_rets.dropna()[rolling_1y_rets.dropna() <= es_95_tag_hist].mean())

# Monte Carlo
simulations = 10000
tage = 252
daily_rets = port_rendite.dropna()
sim_returns = np.random.choice(daily_rets, size=(tage, simulations), replace=True)
paths = np.prod(1 + sim_returns, axis=0) - 1
schwellenwert = np.percentile(paths, 5)
mc_var_95_jahr = schwellenwert * -1
mc_es_95_jahr = paths[paths <= schwellenwert].mean() * -1

# Risikoverteilung
cov_matrix = renditen[verfuegbare].cov() * 252
port_vola = np.sqrt(np.dot(aktuelle_gewichte, np.dot(cov_matrix, aktuelle_gewichte)))
marginal_contrib = np.dot(cov_matrix, aktuelle_gewichte) / port_vola
abs_risk_contrib = aktuelle_gewichte * marginal_contrib
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
        div_dates = div_history.index
        if hasattr(div_dates, 'tz') and div_dates.tz is not None:
            div_dates = div_dates.tz_localize(None)
        div_history = div_history.copy()
        div_history.index = div_dates
        period_start = daten.index[0]
        period_end = daten.index[-1]
        divs_in_period = div_history[(div_dates >= period_start) & (div_dates <= period_end)]
        anzahl_jahre = max(jahre, 1.0)
        try:
            last_date = daten.index[-1]
            price_raw = ticker_obj.history(start=last_date - pd.Timedelta(days=5), end=last_date + pd.Timedelta(days=1))['Close'].iloc[-1]
            price_eur = daten[t].iloc[-1]
            fx_faktor = price_eur / price_raw
        except:
            fx_faktor = 1.0
        _ticker_fx = fx_prices_map.get(t, None)
        for date, amount in divs_in_period.items():
            stueckzahl_start = (startkapital * gewicht) / daten[t].iloc[0]
            stueckzahl_ende = (endsumme * gewicht) / daten[t].iloc[-1]
            stueckzahl_avg = (stueckzahl_start + stueckzahl_ende) / 2
            if _ticker_fx is not None:
                try:
                    _lookup_date = date.tz_localize(None) if hasattr(date, 'tz') and date.tz is not None else date
                    actual_fx_at_date = _ticker_fx.asof(_lookup_date)
                    if pd.isna(actual_fx_at_date):
                        actual_fx_at_date = fx_faktor
                    euro_zahlung_avg = (amount * actual_fx_at_date * stueckzahl_avg) / anzahl_jahre
                except:
                    euro_zahlung_avg = (amount * fx_faktor * stueckzahl_avg) / anzahl_jahre
            else:
                euro_zahlung_avg = (amount * fx_faktor * stueckzahl_avg) / anzahl_jahre
            if euro_zahlung_avg > 0:
                cal_data.append({
                    "Monat": date.strftime("%B"), "Monat_Nr": date.month,
                    "Ticker": t, "Name": ticker_namen.get(t, t), "Ausschüttung": euro_zahlung_avg
                })
        one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
        last_year_divs = div_history[div_history.index > one_year_ago]
        if not last_year_divs.empty and _ticker_fx is not None:
            last_year_divs_eur = 0.0
            for _div_date, _div_amt in last_year_divs.items():
                try:
                    _lookup = _div_date.tz_localize(None) if hasattr(_div_date, 'tz') and _div_date.tz is not None else _div_date
                    _fx_at_div = _ticker_fx.asof(_lookup)
                    if pd.isna(_fx_at_div):
                        _fx_at_div = fx_faktor
                except:
                    _fx_at_div = fx_faktor
                last_year_divs_eur += _div_amt * _fx_at_div
        elif not last_year_divs.empty:
            last_year_divs_eur = last_year_divs.sum() * fx_faktor
        else:
            last_year_divs_eur = 0.0
        stueckzahl_heute = (endsumme * gewicht) / daten[t].iloc[-1]
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
ter_anzeige = f"{port_ter:.2f}%"
st.markdown("""
<style>
    .header-grid {
        display: grid;
        grid-template-columns: 1.5fr 1.2fr 1fr 1.2fr 1fr;
        gap: 15px;
        margin-bottom: 25px;
    }
    .header-card {
        background-color: rgba(100,100,100,0.1);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4A90E2;
    }
    .header-label { color: var(--ts) !important; opacity: 0.55 !important; font-size: 14px !important; margin: 0 !important; }
    .header-value { color: var(--tv) !important; font-size: 28px !important; font-weight: bold !important; margin: 5px 0 0 0 !important; }
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
<div class="header-card">
<p class="header-label">Portfoliokosten p.a. (TER)</p>
<p class="header-value">{ter_anzeige}</p>
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
        opacity: 0.55;
        color: var(--ts);
        font-size: 13px;
        margin: 0;
    }
    .kpi-value {
        color: var(--tv);
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

st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
_drift_label = "Portfoliozusammensetzung - Positionen & Aktuelle Gewichtung (ohne Rebalancing)" if not rebalance_active else "Portfoliozusammensetzung - Positionen & Aktuelle Gewichtung (mit Rebalancing)"
with st.expander(_drift_label):
    if rebalance_active:
        st.caption("Es findet ein jährliches Rebalancing auf die ursprüngliche Gewichtung (Startgewichtung) statt.")
    else:
        st.caption("Es findet kein jährliches Rebalancing statt, die Gewichtungen haben sich je nach Performance verändert (Buy & Hold).")
    _drift_rows = ""
    for i, t in enumerate(verfuegbare):
        _name        = ticker_namen.get(t, t)
        _ziel_pct    = anteile[i] * 100
        _aktuell_pct = aktuelle_gewichte[i] * 100
        _drift_pct   = _aktuell_pct - _ziel_pct
        _drift_farbe = "#27AE60" if _drift_pct >= 0 else "#EB5757"
        _drift_pfeil = "▲" if _drift_pct > 0.05 else ("▼" if _drift_pct < -0.05 else "●")
        _drift_rows += f"""
        <div class="port-row">
            <div class="port-bar" style="width: {_aktuell_pct}%;"></div>
            <div class="port-ticker">{t}</div>
            <div class="port-name">{_name}</div>
            <div style="color:var(--ts); opacity:0.55; font-size:12px; margin-left:10px; z-index:1; min-width:55px; text-align:right;">Ziel: {_ziel_pct:.1f}%</div>
            <div style="font-weight:bold; color:var(--tv); margin-left:10px; z-index:1; min-width:65px; text-align:right;">{_aktuell_pct:.1f}%</div>
            <div style="color:{_drift_farbe}; font-size:12px; margin-left:8px; z-index:1; min-width:55px; text-align:right;">{_drift_pfeil} {_drift_pct:+.1f}%</div>
        </div>"""
    st.markdown(f'<div class="port-container">{_drift_rows}</div>', unsafe_allow_html=True)
    if not rebalance_active and len(verfuegbare) > 1:
        _max_over  = max(range(len(verfuegbare)), key=lambda i: aktuelle_gewichte[i] - anteile[i])
        _max_under = min(range(len(verfuegbare)), key=lambda i: aktuelle_gewichte[i] - anteile[i])
        _over_d  = (aktuelle_gewichte[_max_over]  - anteile[_max_over])  * 100
        _under_d = (aktuelle_gewichte[_max_under] - anteile[_max_under]) * 100
        if abs(_over_d) > 0.5 or abs(_under_d) > 0.5:
            st.caption(f"Stärkster Drift: **{verfuegbare[_max_over]}** {_over_d:+.1f}% · **{verfuegbare[_max_under]}** {_under_d:+.1f}%")

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
    sma50 = port_kum.rolling(window=50).mean()
    sma200 = port_kum.rolling(window=200).mean()
    df_perf_plot = pd.DataFrame({
        'Datum': port_kum.index,
        'Portfolio': port_kum.values,
        'Benchmark': bench_kum.values,
        '50-Tage-Linie': sma50.values,
        '200-Tage-Linie': sma200.values
    })
    fig_perf = px.line(
        df_perf_plot,
        x='Datum', 
        y=['Portfolio', 'Benchmark', '50-Tage-Linie', '200-Tage-Linie'],
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
    colors = {'Portfolio': '#FFD700', 'Benchmark': '#4A90E2', '50-Tage-Linie': '#F59E0B', '200-Tage-Linie': '#EF4444'}
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
        if not rebalance_active:
            for t in verfuegbare:
                einzel_ret = (1 + renditen[t]).prod() - 1
                gewicht = anteile[verfuegbare.index(t)]
                beitraege.append(einzel_ret * gewicht * 100)
        else:
            target_w = np.array(anteile)
            current_w = target_w.copy()
            asset_contribs = np.zeros(len(verfuegbare))
            rets_df = renditen[verfuegbare]
            for i in range(len(rets_df)):
                daily_r = rets_df.iloc[i].values
                asset_contribs += current_w * daily_r
                drifted = current_w * (1 + daily_r)
                current_w = drifted / np.sum(drifted)
                if i + 1 < len(rets_df):
                    if rets_df.index[i+1].year > rets_df.index[i].year:
                        current_w = target_w.copy()
            beitraege = (asset_contribs * 100).tolist()
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
        df_div_plot = df_cal.groupby(['Monat_Nr', 'Ticker', 'Name'])['Ausschüttung'].sum().reset_index()
        df_div_plot['Monat'] = df_div_plot['Monat_Nr'].apply(lambda x: monats_namen[x-1])
        fig_div = px.bar(
            df_div_plot,
            x='Monat',
            y='Ausschüttung',
            color='Name',
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
        st.caption(f"💸 Summe: **{total_div:.2f} €**")
    
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
            expected_ret = risk_free_rate + asset_beta * (bench_arith - risk_free_rate)
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
                .opt-name-label { color: var(--ts); opacity: 0.55; font-size: 11px; margin-bottom: 4px; display: block; }
                .bar-bg-stacked { 
                    background: rgba(255,255,255,0.05); height: 10px; border-radius: 5px; 
                    width: 100%; display: flex; overflow: hidden; margin-bottom: 4px;
                }
                .bar-segment-act { background: #6B7280; }
                .bar-segment-opt { background: #4A90E2; }
                .values-row { display: flex; justify-content: space-between; font-size: 11px; color: var(--ts); opacity: 0.55; }
                .val-act { color: var(--tm); opacity: 0.7; }
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
    .r-title { color: var(--ts); opacity: 0.55; font-size: 13px; margin-bottom: 12px; text-transform: uppercase; font-weight: bold; }
    .r-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
    .r-lbl { color: var(--ts); opacity: 0.55; font-size: 13px; }
    .r-val { color: var(--tv); font-size: 18px; font-weight: bold; }
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
        .mc-card { background: var(--bg-card); border-radius: 10px; padding: 15px; border-left: 5px solid #4A90E2; display: flex; justify-content: space-between; align-items: center; }
        .mc-label { color: var(--ts); opacity: 0.55; font-size: 14px; font-weight: 500; }
        .mc-amount { display: block; color: var(--tv); font-size: 18px; font-weight: bold; }
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
    _asset_typen_fa = st.session_state.get("asset_typen", {})
    _equity_tickers = [t for t in verfuegbare if _asset_typen_fa.get(t, {}).get("typ", "Aktie") == "Aktie"]
    if not _equity_tickers:
        st.info("Die Fama-French-Faktorenanalyse wird angezeigt, sobald mindestens eine Aktienposition im Portfolio enthalten ist.")
        factors = pd.Series(0.0, index=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
    else:
      try:
        _equity_weights_fa = [anteile[verfuegbare.index(t)] for t in _equity_tickers]
        _equity_weights_fa = [w / sum(_equity_weights_fa) for w in _equity_weights_fa]
        if not rebalance_active:
            _equity_port_ret, _ = calculate_buy_and_hold(renditen[_equity_tickers], _equity_weights_fa)
        else:
            _equity_port_ret = calculate_annual_rebalancing(renditen[_equity_tickers], _equity_weights_fa)
        with st.spinner("Berechne Faktor-Exposures..."):
            result = get_factor_loadings(_equity_port_ret)          
            if isinstance(result, dict):
                loadings = result['loadings']
                factors = loadings
                p_values = result['p_values']
                r_sq = result['r_squared']
                alpha_val = result['annualized_alpha']
                alpha_p = result['alpha_p_value']
                def get_stars(p):
                    if p < 0.01: return "***"
                    if p < 0.05: return "**"
                    if p < 0.1: return "*"
                    return ""
                display_map = {
                    'Mkt-RF': 'Market',
                    'SMB': 'Size',
                    'HML': 'Value',
                    'RMW': 'Quality I: Profitability',
                    'CMA': 'Quality II: Investmentbehavior'
                }          
                factor_names_with_stars = [display_map[idx] for idx in loadings.index]
                bar_labels = [f"{val:.3f} {get_stars(p_values[idx])}" for idx, val in loadings.items()]
                colors_fac = ['#4A90E2', '#27AE60', '#F2994A', '#9B51E0', '#D488FF']         
                fig_fac = go.Figure(go.Bar(
                    x=loadings.values,
                    y=factor_names_with_stars, 
                    orientation='h',
                    marker_color=colors_fac, 
                    hovertemplate="Faktor: <b>%{y}</b><br>Beta: <b>%{x:.3f}</b><extra></extra>",
                    width=0.6,
                    text=bar_labels,        
                    textposition='outside',   
                    cliponaxis=False
                ))             
                fig_fac.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=20, t=10, b=10),
                    height=350, 
                    xaxis=dict(
                        title="Factor-Betas", 
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
                alpha_stars = get_stars(alpha_p)
                st.markdown(f"**Alpha:** {alpha_val:.2%} {get_stars(alpha_p)} | **R²:** {r_sq:.2%}")
                st.caption("p-Wert: * < 0.1, ** < 0.05, *** < 0.01")            
            else:
                st.info(f"💡 {result}")
                factors = pd.Series(0.0, index=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])          
      except Exception as e:
        st.error(f"Faktoranalyse konnte nicht geladen werden: {e}")
with tab_sim:
    def berechne_nicht_aktien_beitrag(verfuegbare, anteile, asset_typen, szenario_schocks):
        """
        Berechnet den Rendite-Beitrag aller Nicht-Aktien-Positionen (Anleihen,
        Rohstoffe) für ein gegebenes Szenario-Schock-Dict.
        Aktien werden hier bewusst übersprungen – ihr Beitrag kommt weiterhin
        aus dem FF5-Modell in der bestehenden Logik.
        """
        _BOND_BETAS = {
            "Staatsanleihe":             {"credit_spread": 0.0,  "liquidity": 0.0 },
            "Unternehmensanleihe (IG)":  {"credit_spread": 0.5,  "liquidity": 0.10},
            "Unternehmensanleihe (HY)":  {"credit_spread": 1.5,  "liquidity": 0.30},
            "Emerging Markets":          {"credit_spread": 1.2,  "liquidity": 0.40},
        }
        _COMMODITY_BETAS = {
            "Gold":           {"spot": 1.0, "usd": -0.85, "roll":  0.00},
            "Silber":         {"spot": 1.0, "usd": -0.70, "roll":  0.00},
            "Rohöl (WTI)":   {"spot": 1.0, "usd": -0.65, "roll": -0.30},
            "Rohöl (Brent)": {"spot": 1.0, "usd": -0.60, "roll": -0.25},
            "Erdgas":         {"spot": 1.0, "usd": -0.30, "roll": -0.55},
            "Weizen":         {"spot": 1.0, "usd": -0.45, "roll": -0.40},
            "Kupfer":         {"spot": 1.0, "usd": -0.60, "roll": -0.20},
        }
        _CRYPTO_BETAS = {
            "Bitcoin":              {"market_sentiment": 2.0, "usd": -1.0,  "liquidity": -0.5},
            "Ethereum":             {"market_sentiment": 2.5, "usd": -1.0,  "liquidity": -0.7},
            "Altcoin (Large Cap)":  {"market_sentiment": 3.0, "usd": -1.2,  "liquidity": -1.0},
            "Altcoin (Small Cap)":  {"market_sentiment": 4.0, "usd": -1.2,  "liquidity": -1.5},
        }
        gesamt_beitrag  = 0.0
        detail_liste    = []
        for i, t in enumerate(verfuegbare):
            info    = asset_typen.get(t, {"typ": "Aktie"})
            typ     = info.get("typ", "Aktie")
            gewicht = anteile[i]
            if typ == "Anleihe":
                duration   = info.get("duration", 5.0)
                bond_type  = info.get("bond_type", "Staatsanleihe")
                cb         = _BOND_BETAS.get(bond_type, _BOND_BETAS["Staatsanleihe"])
                yield_ret  = -duration * szenario_schocks.get("yield_change", 0.0)
                spread_ret = -cb["credit_spread"] * szenario_schocks.get("credit_spread", 0.0)
                liq_ret    = -cb["liquidity"]     * szenario_schocks.get("liquidity",     0.0)
                pos_ret    = yield_ret + spread_ret + liq_ret
                beitrag    = gewicht * pos_ret
                gesamt_beitrag += beitrag
                detail_liste.append({"ticker": t, "typ": typ,
                                     "rendite": pos_ret, "beitrag": beitrag,
                                     "details": f"Duration {duration:.1f}J | {bond_type}"})
            elif typ == "Rohstoff / Edelmetall":
                rohstoff = info.get("rohstoff", "Gold")
                cb       = _COMMODITY_BETAS.get(rohstoff, _COMMODITY_BETAS["Gold"])
                spot_ret = cb["spot"] * szenario_schocks.get("spot_return", 0.0)
                usd_ret  = cb["usd"]  * szenario_schocks.get("usd_index",   0.0)
                roll_ret = cb["roll"] * szenario_schocks.get("roll_yield",   0.0)
                pos_ret  = spot_ret + usd_ret + roll_ret
                beitrag  = gewicht * pos_ret
                gesamt_beitrag += beitrag
                detail_liste.append({"ticker": t, "typ": typ,
                                     "rendite": pos_ret, "beitrag": beitrag,
                                     "details": rohstoff})
            elif typ == "Kryptowährung":
                crypto = info.get("crypto", "Bitcoin")
                cb     = _CRYPTO_BETAS.get(crypto, _CRYPTO_BETAS["Bitcoin"])
                sent_ret = cb["market_sentiment"] * szenario_schocks.get("crypto_sentiment", 0.0)
                usd_ret  = cb["usd"]              * szenario_schocks.get("usd_index",        0.0)
                liq_ret  = cb["liquidity"]         * szenario_schocks.get("liquidity",        0.0)
                pos_ret  = sent_ret + usd_ret + liq_ret
                beitrag  = gewicht * pos_ret
                gesamt_beitrag += beitrag
                detail_liste.append({"ticker": t, "typ": typ,
                                     "rendite": pos_ret, "beitrag": beitrag,
                                     "details": crypto})
        return gesamt_beitrag, detail_liste
    @st.fragment
    def render_simulation_area(factors, beta, endsumme, szenario_gewichte):
        szenarien = {
            "Stagflation": [-0.35, -0.05, +0.15, +0.08, +0.05], 
            "Inflation": [-0.20, 0.00, 0.10, 0.12, 0.05],
            "Schwere Rezession": [-0.45, -0.10, +0.05, +0.15, +0.10],
            "Tech-Blase": [-0.40, -0.05, +0.35, +0.15, +0.00],
            "Small Cap Rallye": [+0.15, +0.20, +0.05, -0.05, -0.05]
        }
        szenarien_multi = {
            "Stagflation": {
                "yield_change":  +0.015,  
                "credit_spread": +0.010,   
                "liquidity":     +0.005,
                "spot_return":   +0.15,    
                "usd_index":     +0.02,
                "roll_yield":    -0.01,
                "crypto_sentiment": -0.25,
            },
            "Inflation": {
                "yield_change":  +0.020,   
                "credit_spread": +0.008,
                "liquidity":     +0.002,
                "spot_return":   +0.20,    
                "usd_index":     +0.03,
                "roll_yield":    -0.005,
                "crypto_sentiment": -0.10,
            },
            "Schwere Rezession": {
                "yield_change":  -0.020,   
                "credit_spread": +0.040,  
                "liquidity":     +0.025,
                "spot_return":   -0.22,    
                "usd_index":     +0.05,    
                "roll_yield":    -0.02,
                "crypto_sentiment": -0.35,
            },
            "Tech-Blase": {
                "yield_change":  -0.005,
                "credit_spread": +0.010,
                "liquidity":     +0.005,
                "spot_return":   -0.05,
                "usd_index":     +0.01,
                "roll_yield":     0.00,
                "crypto_sentiment": -0.30,
            },
            "Small Cap Rallye": {
                "yield_change":  +0.003,
                "credit_spread": -0.002,
                "liquidity":      0.000,
                "spot_return":   +0.02,
                "usd_index":     -0.005,
                "roll_yield":    +0.005,
                "crypto_sentiment": +0.15,
            },
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
            gesamt_aktien_ret = np.sum(verlust_pro_faktor)
            _asset_typen = st.session_state.get("asset_typen", {})
            _multi_schocks = szenarien_multi[auswahl]
            _equity_weight = sum(
                szenario_gewichte[i] for i, t in enumerate(verfuegbare)
                if _asset_typen.get(t, {}).get("typ", "Aktie") == "Aktie"
            )
            if _equity_weight == 0:
                gesamt_aktien_ret = 0.0
            else:
                gesamt_aktien_ret = gesamt_aktien_ret * _equity_weight
            nicht_aktien_ret, nicht_aktien_details = berechne_nicht_aktien_beitrag(
                verfuegbare, szenario_gewichte, _asset_typen, _multi_schocks
            )
            gesamt_sz_ret  = gesamt_aktien_ret + nicht_aktien_ret
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
    render_simulation_area(factors, beta, endsumme, aktuelle_gewichte if not rebalance_active else anteile)
st.markdown("---")
st.caption(f"Datenzeitraum: {daten.index[0].strftime('%d.%m.%Y')} bis {daten.index[-1].strftime('%d.%m.%Y')}")
