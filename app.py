import json
import os
import requests
import streamlit as st
from PIL import Image
import mplfinance as mpf
import pandas as pd
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Mesterlövész Chart Analyzer (Élő)",
    page_icon="🎯",
    layout="centered"
)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_binance_chart(symbol, interval):
    # Binance API idősík mapping
    interval_map = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "1day": "1d"
    }
    binance_interval = interval_map.get(interval, "1m")
    
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={binance_interval}&limit=100"
    response = requests.get(url)
    data = response.json()
    
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Nem érkezett adat a {symbol} szimbólumhoz a Binance-ról.")
        
    df = pd.DataFrame(data, columns=[
        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    
    # Adatok konvertálása számmá
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col])
        
    df['datetime'] = pd.to_datetime(df['Open_time'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def generate_chart_image(ticker, interval, filename="current_chart.png"):
    df = get_binance_chart(ticker, interval)
    
    if len(df) < 15:
        raise ValueError("Túl kevés gyertya érkezett az elemzéshez.")
        
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

    1. **SMC / ICT Struktúra:** Liquidity Sweep megtörtént? Szerkezeti törés (ChoCh/BoS) látható? Van FVG vagy OB?
    2. **Price Action:** Erős elutasító gyertya a kulcsszintnél?
    3. **Fibonacci OTE:** Az ár a 61.8% - 79% prémium/diszkont zónába érkezett?
    4. **RSI Megerősítés:** Látható RSI Divergencia vagy extrém túladott/túlvett zónából fordulás?

    Válaszolj KIZÁRÓLAG egy érvényes JSON objektumban. Számok (entry, sl, tp) tiszta float értékek legyenek!

    JSON struktúra:
    {{
      "direction": "BUY" vagy "SELL" vagy "NEUTRAL",
      "entry": 2650.50,
      "sl": 2640.00,
      "tp": 2680.00,
      "confluences_found": ["SMC Sweep", "RSI Divergence"],
      "reasoning": "Részletes indoklás..."
    }}
    """
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[image, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text)

def calculate_risk_reward(direction, entry, sl, tp):
    try:
        entry, sl, tp = float(entry), float(sl), float(tp)
        if direction == "BUY":
            risk, reward = entry - sl, tp - entry
        elif direction == "SELL":
            risk, reward = sl - entry, entry - tp
        else:
            return None, "NEUTRAL"
        return reward / risk if risk > 0 else None, None
    except:
        return None, "Hiba"

def calculate_position_size(balance, risk_pct, entry, sl, ticker):
    try:
        risk_usd = float(balance) * (float(risk_pct) / 100.0)
        price_delta = abs(float(entry) - float(sl))
        if price_delta <= 0: return None, None, "Hiba"
        
        if "PAXG" in ticker:
            return risk_usd, f"{(risk_usd / (price_delta * 100.0)):.2f} Lot", None
        elif "BTC" in ticker:
            return risk_usd, f"{(risk_usd / price_delta):.4f} BTC", None
        else:
            return risk_usd, f"{(risk_usd / (price_delta * 10)):.2f} Kontraktus", None
    except:
        return None, None, "Hiba"

st.title("🎯 Mesterlövész Chart Analyzer (Élő)")
st.markdown("Valós idejű, másodpercre pontos perces adatok.")

st.sidebar.header("⚙️ Konfiguráció")
gemini_api_input = st.sidebar.text_input("Gemini API Kulcs (AI)", value=os.environ.get("GEMINI_API_KEY", ""), type="password")

assets = {
    "🥇 Arany Spot (PAXG / USD)": "PAXGUSDT",
    "₿ Bitcoin (BTC / USDT)": "BTCUSDT",
    "イー Ethereum (ETH / USDT)": "ETHUSDT"
}

styles = {
    "🚀 Mikro-Skalp (1 perces)": "1min",
    "⚡ Skalp (5 perces)": "5min",
    "⚡ Skalp (15 perces)": "15min",
    "🌊 Swing (Napi)": "1day"
}

st.sidebar.header("📊 Kereskedés")
selected_asset_name = st.sidebar.selectbox("Instrumentum", list(assets.keys()))
selected_style_name = st.sidebar.selectbox("Idősík", list(styles.keys()))
balance_input = st.sidebar.text_input("Számla Tőke ($)", value="10000")
risk_input = st.sidebar.text_input("Kockázat (%)", value="1.0")

if st.sidebar.button("🚀 Elemzés Indítása", use_container_width=True):
    if not gemini_api_input:
        st.error("Kérlek add meg a Gemini API kulcsot az oldalsávban!")
    else:
        ticker = assets[selected_asset_name]
        interval = styles[selected_style_name]
        
        with st.spinner(f"Valós idejű adatok betöltése & AI Elemzés ({ticker})..."):
            try:
                img_path = generate_chart_image(ticker, interval)
                data = analyze_chart(img_path, gemini_api_input, selected_asset_name, selected_style_name)
                
                dir_val, entry, sl, tp = data.get('direction', 'NEUTRAL'), data.get('entry'), data.get('sl'), data.get('tp')
                
                st.markdown("---")
                if dir_val == "BUY": st.success(f"Irány: {dir_val} 🟢")
                elif dir_val == "SELL": st.error(f"Irány: {dir_val} 🔴")
                else: st.warning(f"Irány: {dir_val} 🟠 (Kivárás)")
                
                st.markdown(f"**Confluence-ek:** {', '.join(data.get('confluences_found', []))}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Belépő", str(entry) if entry else "-")
                c2.metric("Stop Loss", str(sl) if sl else "-")
                c3.metric("Take Profit", str(tp) if tp else "-")
                
                if dir_val != "NEUTRAL" and entry and sl and tp:
                    rr, _ = calculate_risk_reward(dir_val, entry, sl, tp)
                    if rr: st.info(f"R:R Arány -> 1 : {rr:.2f}")
                    r_usd, size, _ = calculate_position_size(balance_input, risk_input, entry, sl, ticker)
                    if r_usd:
                        ca, cb = st.columns(2)
                        ca.metric("Kockázat", f"${r_usd:.2f}")
                        cb.metric("Méret", str(size))
                
                st.markdown("### 📝 Indoklás")
                st.write(data.get('reasoning', ''))
                st.image(img_path)
                
            except Exception as e:
                st.error(f"Hiba: {e}")
