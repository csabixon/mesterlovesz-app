import json
import os
import streamlit as st
from PIL import Image
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Mesterlövész Chart Analyzer",
    page_icon="🎯",
    layout="centered"
)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_chart_image(ticker, interval, period, filename="current_chart.png"):
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    
    # --- XAUUSD=X FALLBACK JAVÍTÁS ---
    # Ha a spot arany nem ad adatot, automatikusan határidős (GC=F) aranyra váltunk
    if df.empty and ticker == "XAUUSD=X":
        ticker = "GC=F"
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        
    if df.empty:
        raise ValueError(f"Nem sikerült adatot letölteni ehhez a tickerhez: {ticker}")
    
    # --- YFINANCE TÁBLÁZAT JAVÍTÁS ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna(subset=['Open', 'Close'])
    # ------------------------------

    if df.empty:
        raise ValueError(f"Az adatok tisztítása után nem maradt érvényes adat: {ticker}")

    df = df.tail(80)
    df['RSI'] = calculate_rsi(df['Close'])
    
    mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)
    
    ap = [
        mpf.make_addplot(df['RSI'], panel=1, ylabel='RSI (14)', color='purple', width=1.2)
    ]
    
    mpf.plot(
        df, 
        type='candle', 
        style=s, 
        addplot=ap,
        panel_ratios=(3, 1),
        savefig=filename, 
        tight_layout=True, 
        figsize=(10, 7)
    )
    return filename

def analyze_chart(image_path, api_key, pair_name, style_name):
    client = genai.Client(api_key=api_key)
    image = Image.open(image_path)
    
    prompt = f"""
    Ez egy {pair_name} instrumentum chartja, {style_name} idősíkon. 
    A felső panelen a gyertyadiagram, az alsó panelen az RSI (14) indikátor látható.

    Kereskedési feladatod: Eljárni mint egy MESTERLÖVÉSZ (Sniper) kereskedő.
    KIZÁRÓLAG akkor adj BUY vagy SELL jelet, ha legalább 3 AZ ALÁBBI 4 CONFLUENCE (egybeesés) KÖZÜL TELJESÜL:

    1. **SMC / ICT Struktúra:**
       - Liquidity Sweep (Sell-side / Buy-side likviditás kisöprése) megtörtént?
       - Szerkezeti törés (ChoCh vagy BoS) megerősítette a fordulatot?
       - Van kitöltetlen Fair Value Gap (FVG) vagy érvényes Order Block (OB)?

    2. **Price Action (Gyertya alakzatok):**
       - Látható-e erős elutasító gyertya (Pin bar, Bullish/Bearish Engulfing, hosszú kanóc) a kulcsszintnél?

    3. **Fibonacci / OTE (Optimal Trade Entry):**
       - Az ármozgás a prémium vagy diszkont zónába (kb. 61.8% - 79% visszahúzódáshoz) érkezett?

    4. **RSI Indikátor Megerősítés (Alsó panel):**
       - Látható RSI Divergencia (pl. árfolyam új alacsonyabb mélypontot üt, de az RSI magasabb mélypontot mutat)?
       - Az RSI túladott (<30) vagy túlvett (>70) zónából fordul?

    SZIGORÚ SZABÁLY: Ha bizonytalanság van, vagy nincs meg a többszörös megerősítés (confluence), a válasz irányának KIZÁRÓLAG "NEUTRAL"-nak kell lennie!

    Válaszolj KIZÁRÓLAG egy érvényes JSON objektumban. 
    A szintek (entry, sl, tp) TISZTA SZÁMOK legyenek (float vagy null), mértékegység nélkül!

    JSON struktúra:
    {{
      "direction": "BUY" vagy "SELL" vagy "NEUTRAL",
      "entry": 2650.50,
      "sl": 2640.00,
      "tp": 2680.00,
      "confluences_found": ["SMC Sweep + ChoCh", "Engulfing candle", "RSI Divergence"],
      "reasoning": "Részletes magyar nyelvű indoklás a Mesterlövész szempontok alapján..."
    }}
    """

    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text)

def calculate_risk_reward(direction, entry, sl, tp):
    try:
        entry, sl, tp = float(entry), float(sl), float(tp)
    except (ValueError, TypeError):
        return None, "A szintek nem konvertálhatóak számmá."

    direction = str(direction).upper()
    if direction == "BUY":
        risk, reward = entry - sl, tp - entry
    elif direction == "SELL":
        risk, reward = sl - entry, entry - tp
    else:
        return None, "Semleges (NEUTRAL) pozíció."

    if risk <= 0 or reward <= 0:
        return None, "Érvénytelen Stop Loss vagy Take Profit."
    return reward / risk, None

def calculate_position_size(balance, risk_pct, entry, sl, ticker):
    try:
        balance, risk_pct = float(balance), float(risk_pct)
        entry, sl = float(entry), float(sl)
    except (ValueError, TypeError):
        return None, None, "Kérlek adj meg érvényes számokat!"

    risk_usd = balance * (risk_pct / 100.0)
    price_delta = abs(entry - sl)

    if price_delta <= 0:
        return None, None, "Entry és SL megegyezik."

    if ticker == "GC=F" or ticker == "XAUUSD=X":
        lots = risk_usd / (price_delta * 100.0)
        lot_str = f"{lots:.2f} Lot (Arany)"
    elif "=X" in ticker:
        units = risk_usd / price_delta
        standard_lots = units / 100000.0
        lot_str = f"{standard_lots:.2f} Lot (Forex)"
    elif "BTC" in ticker:
        units = risk_usd / price_delta
        lot_str = f"{units:.4f} BTC"
    else:
        multiplier = 20.0 if "NQ" in ticker else (50.0 if "ES" in ticker else 1000.0)
        contracts = risk_usd / (price_delta * multiplier)
        lot_str = f"{contracts:.2f} Kontraktus"

    return risk_usd, lot_str, None

# --- STREAMLIT FELÜLET ---
st.title("🎯 Mesterlövész Chart Analyzer")
st.markdown("SMC + Price Action + Fibonacci OTE + RSI Divergencia AI-alapú elemzés.")

st.sidebar.header("⚙️ Beállítások")

api_key_input = st.sidebar.text_input(
    "Gemini API Kulcs", 
    value=os.environ.get("GEMINI_API_KEY", ""), 
    type="password"
)

assets = {
    "🥇 Arany Forex (XAU/USD)": "XAUUSD=X",
    "🥇 Arany Futures (GC=F)": "GC=F",
    "📊 EUR/USD (Forex)": "EURUSD=X",
    "💷 GBP/USD (Forex)": "GBPUSD=X",
    "📈 Nasdaq Futures (NQ=F)": "NQ=F",
    "🇺🇸 S&P 500 Futures (ES=F)": "ES=F",
    "🛢️ Kőolaj Futures (CL=F)": "CL=F",
    "₿ Bitcoin (BTC-USD)": "BTC-USD"
}

styles = {
    "⚡ Skalp (15 perces chart)": {"interval": "15m", "period": "5d"},
    "🌊 Swing (Napi chart)": {"interval": "1d", "period": "6mo"}
}

selected_asset_name = st.sidebar.selectbox("Instrumentum", list(assets.keys()))
selected_style_name = st.sidebar.selectbox("Kereskedési Stílus", list(styles.keys()))

balance_input = st.sidebar.text_input("Számla Tőke ($)", value="10000")
risk_input = st.sidebar.text_input("Kockázat (%)", value="1.0")

analyze_btn = st.sidebar.button("🚀 Mesterlövész Elemzés Indítása", use_container_width=True)

if analyze_btn:
    if not api_key_input:
        st.error("Kérlek add meg a Gemini API kulcsot az oldalsávban!")
    else:
        ticker = assets[selected_asset_name]
        timeframe_data = styles[selected_style_name]
        
        with st.spinner(f"Adatok letöltése & Elemzés folyamatban ({selected_asset_name})..."):
            try:
                img_path = generate_chart_image(ticker, timeframe_data["interval"], timeframe_data["period"])
                data = analyze_chart(img_path, api_key_input, selected_asset_name, selected_style_name)
                
                dir_val = data.get('direction', 'NEUTRAL')
                confluences = data.get('confluences_found', [])
                entry = data.get('entry')
                sl = data.get('sl')
                tp = data.get('tp')
                
                st.markdown("---")
                st.subheader("📊 Elemzési Eredmények")
                
                if dir_val == "BUY":
                    st.success(irany_text := f"Irány: {dir_val} 🟢")
                elif dir_val == "SELL":
                    st.error(irany_text := f"Irány: {dir_val} 🔴")
                else:
                    st.warning(irany_text := f"Irány: {dir_val} 🟠 (Nincs elég confluence)")
                
                st.markdown(f"**Megtalált Confluence-ek:** {', '.join(confluences) if confluences else 'Nincs'}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Belépő (Entry)", str(entry) if entry else "-")
                col2.metric("Stop Loss (SL)", str(sl) if sl else "-")
                col3.metric("Take Profit (TP)", str(tp) if tp else "-")
                
                if dir_val != "NEUTRAL" and entry and sl and tp:
                    rr, error = calculate_risk_reward(dir_val, entry, sl, tp)
                    if rr is not None:
                        rr_str = f"1 : {rr:.2f}"
                        if rr < 2:
                            st.warning(f"Kockázat / Hozam (R:R): {rr_str} (⚠️ 1:2 alatt!)")
                        else:
                            st.info(f"Kockázat / Hozam (R:R): {rr_str} (✅ Kiváló)")
                    
                    risk_usd, lot_size, _ = calculate_position_size(balance_input, risk_input, entry, sl, ticker)
                    if risk_usd is not None:
                        col_a, col_b = st.columns(2)
                        col_a.metric("Kockáztatott Összeg", f"${risk_usd:.2f} USD")
                        col_b.metric("Javasolt Méret", str(lot_size))
                
                st.markdown("### 📝 Részletes Indoklás")
                st.write(data.get('reasoning', 'Nincs magyarázat.'))
                
                st.markdown("### 📈 Vizsgált Chart & RSI")
                st.image(img_path, caption=f"{selected_asset_name} - {selected_style_name}")
                
            except Exception as e:
                st.error(f"Hiba történt a folyamat során: {e}")
