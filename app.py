import json
import os
import time
from datetime import datetime

import mplfinance as mpf
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from google import genai
from google.genai import types
from PIL import Image

st.set_page_config(
    page_title="Mesterlövész Chart Analyzer",
    page_icon="🎯",
    layout="centered",
)

# ----------------------------------------------------------------------
# Segédfüggvények
# ----------------------------------------------------------------------

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def get_twelvedata_data(symbol: str, interval: str, api_key: str, outputsize: int = 400) -> pd.DataFrame:
    twelve_intervals = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
    }
    iv = twelve_intervals.get(interval, "5min")
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={iv}&outputsize={outputsize}&apikey={api_key}"
    )
    response = requests.get(url, timeout=20)
    data = response.json()

    if "values" not in data:
        msg = data.get("message") or data.get("status") or str(data)
        raise ValueError(f"Twelve Data hiba: {msg}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
        inplace=True,
    )
    df.dropna(subset=["Close"], inplace=True)

    if len(df) < 40:
        raise ValueError(f"Túl kevés gyertya Twelve Data-ból ({len(df)} db)")

    return df


def get_yfinance_data(symbol: str = "GC=F", interval: str = "5m") -> pd.DataFrame:
    yf_intervals = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "60m",
        "4h": "60m",
        "1d": "1d",
    }
    yf_interval = yf_intervals.get(interval, "5m")
    period_map = {
        "1m": "7d",
        "5m": "60d",
        "15m": "60d",
        "1h": "730d",
        "4h": "730d",
        "1d": "2y",
    }
    period = period_map.get(interval, "60d")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=yf_interval, auto_adjust=True)

    if df.empty:
        raise ValueError("yfinance nem adott vissza adatot")

    df = df[["Open", "High", "Low", "Close"]].copy()
    df.dropna(inplace=True)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if len(df) < 40:
        raise ValueError(f"Túl kevés gyertya yfinance-ból ({len(df)} db)")

    return df


def get_data_with_fallback(ticker: str, interval: str, twelvedata_key: str) -> tuple[pd.DataFrame, str]:
    # Gold Futures: először Twelve Data GC, majd yfinance GC=F
    if ticker.upper() in ["GC", "GC=F"]:
        try:
            if twelvedata_key:
                df = get_twelvedata_data("GC", interval, twelvedata_key, outputsize=400)
                return df, "Twelve Data (GC)"
        except Exception:
            pass
        df = get_yfinance_data("GC=F", interval)
        return df, "yfinance (GC=F) – fallback"

    # Spot és forex: Twelve Data
    if not twelvedata_key:
        raise ValueError("Twelve Data API kulcs szükséges ehhez az instrumentumhoz")
    df = get_twelvedata_data(ticker, interval, twelvedata_key, outputsize=400)
    return df, f"Twelve Data ({ticker})"


def generate_chart_image(
    ticker: str,
    interval: str,
    api_key: str,
    filename: str = "current_chart.png",
) -> tuple[str, pd.DataFrame, str]:
    df, source = get_data_with_fallback(ticker, interval, api_key)
    df["RSI"] = calculate_rsi(df["Close"])
    plot_df = df.tail(180).copy()

    mc = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume="in",
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        y_on_right=True,
        facecolor="#0e1117",
        edgecolor="#333333",
        figcolor="#0e1117",
        rc={
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        },
    )
    apds = [
        mpf.make_addplot(
            plot_df["RSI"],
            panel=1,
            ylabel="RSI (14)",
            color="#ab47bc",
            width=1.4,
        )
    ]
    mpf.plot(
        plot_df,
        type="candle",
        style=style,
        addplot=apds,
        panel_ratios=(4, 1),
        savefig=dict(fname=filename, dpi=160, bbox_inches="tight"),
        tight_layout=True,
        figsize=(12, 8),
        volume=False,
        warn_too_much_data=1000,
    )
    return filename, df, source


def analyze_chart(
    image_path: str,
    gemini_api_key: str,
    pair_name: str,
    style_name: str,
    last_close: float,
) -> dict:
    client = genai.Client(api_key=gemini_api_key)
    image = Image.open(image_path)

    prompt = f"""
Te egy extrém fegyelmezett, profi SMC / ICT "Mesterlövész" (Sniper) kereskedő vagy.
Kizárólag magas valószínűségű setupokra adsz jelet. A tőke védelme az elsődleges szempont.

Instrumentum: {pair_name}
Idősík: {style_name}
Utolsó ismert záróár: {last_close}

A képen a gyertyadiagram (felső panel) és az RSI(14) (alsó panel) látható.

==================================================
ALAP CONFLUENCE SZABÁLYOK (legalább 3 a 4-ből kötelező):
==================================================
1. **SMC / ICT Struktúra**: Liquidity Sweep megtörtént + ChoCh vagy BoS látható + érvényes Fair Value Gap (FVG) vagy Order Block
2. **Price Action**: Erős, egyértelmű elutasító gyertya (Pin Bar, Bullish/Bearish Engulfing, vagy nagyon hosszú kanóc) pontosan a kulcsszinten / zónán belül
3. **Fibonacci OTE**: Az ár a 61.8% – 79% prémium/diszkont zónában van
4. **RSI megerősítés**: Látható RSI divergencia VAGY extrém túladott (<30) / túlvett (>70) zónából történő fordulás

==================================================
SKALP SPECIÁLIS EXTRA SZABÁLYOK (1m és 5m idősíkon kötelezően figyeld):
==================================================
- A Liquidity Sweep maximum az utolsó 6–8 gyertyán belül történt meg (friss sweep kell)
- Az elutasító gyertyának erősnek és egyértelműnek kell lennie (nem gyenge, kis testű gyertya)
- Kerüld az oldalazó, alacsony momentumú időszakokat (az utolsó 3–5 gyertyának legyen tiszta irányú momentuma)
- Csak akkor adj BUY vagy SELL jelet, ha a várható Risk:Reward arány legalább 1:1.8
- Ha a fenti skalp extra feltételek közül több is hiányzik → kötelezően NEUTRAL

==================================================
ÁLTALÁNOS SZIGORÚ SZABÁLYOK:
==================================================
- Ha a 4 alap confluence-ből kevesebb mint 3 teljesül → NEUTRAL
- Ha bármilyen kétséged van a setup minőségével kapcsolatban → NEUTRAL
- Soha ne erőltesd a jelet. A "nincs trade" is érvényes és okos döntés.
- Az entry, stop loss és take profit szinteknek logikusnak és a struktúrához igazodónak kell lenniük.

Válaszolj KIZÁRÓLAG érvényes JSON formátumban. Semmilyen más szöveget ne írj a JSON-on kívül!

{{
  "direction": "BUY" | "SELL" | "NEUTRAL",
  "entry": float vagy null,
  "sl": float vagy null,
  "tp": float vagy null,
  "confluences_found": ["lista a talált erős confluences-ekről"],
  "missing": ["lista a hiányzó vagy gyenge feltételekről"],
  "reasoning": "Rövid, szakmai, magyar nyelvű indoklás. Magyarázd el miért adsz jelet vagy miért NEUTRAL."
}}
"""

    models = [
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    last_error = None

    for model_name in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.12,
                    ),
                )
                data = json.loads(response.text)
                data["_model_used"] = model_name
                return data
            except Exception as e:
                last_error = e
                err = str(e).upper()
                if any(
                    x in err
                    for x in (
                        "503",
                        "UNAVAILABLE",
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "HIGH DEMAND",
                        "QUOTA",
                    )
                ):
                    time.sleep(1.5 * (attempt + 1))
                    break  # következő modell
                raise

    raise RuntimeError(f"Minden Gemini modell sikertelen. Utolsó hiba: {last_error}")


def calculate_risk_reward(direction: str, entry, sl, tp):
    try:
        entry, sl, tp = float(entry), float(sl), float(tp)
        if direction == "BUY":
            risk = entry - sl
            reward = tp - entry
        elif direction == "SELL":
            risk = sl - entry
            reward = entry - tp
        else:
            return None
        return round(reward / risk, 2) if risk > 0 else None
    except Exception:
        return None


def calculate_position_size(balance, risk_pct, entry, sl, ticker: str):
    try:
        risk_usd = float(balance) * (float(risk_pct) / 100.0)
        price_delta = abs(float(entry) - float(sl))
        if price_delta <= 0:
            return None, None

        if "XAU" in ticker.upper() or "GC" in ticker.upper():
            lots = risk_usd / (price_delta * 100)
            return risk_usd, f"{lots:.2f} Lot"
        if "BTC" in ticker.upper():
            return risk_usd, f"{(risk_usd / price_delta):.4f} BTC"
        lots = risk_usd / (price_delta * 100000)
        return risk_usd, f"{lots:.2f} Lot"
    except Exception:
        return None, None


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

st.title("🎯 Mesterlövész Chart Analyzer")
st.caption("SMC + Price Action + Fibonacci OTE + RSI | Szigorú confluence + skalp optimalizálás")

st.sidebar.header("⚙️ Konfiguráció")

twelvedata_api_input = st.sidebar.text_input(
    "Twelve Data API Kulcs",
    value=os.environ.get("TWELVEDATA_API_KEY", ""),
    type="password",
)
gemini_api_input = st.sidebar.text_input(
    "Gemini API Kulcs",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password",
)

assets = {
    "🥇 XAU/USD (Arany Spot) – Ajánlott": "XAU/USD",
    "🥇 Gold Futures (GC)": "GC",
    "📊 EUR/USD": "EUR/USD",
    "💷 GBP/USD": "GBP/USD",
    "₿ BTC/USD": "BTC/USD",
}

styles = {
    "🚀 Mikro-Skalp (1 perc)": "1m",
    "⚡ Skalp (5 perc)": "5m",
    "⚡ Skalp (15 perc)": "15m",
    "📈 Intraday (1 óra)": "1h",
    "🌊 Swing (4 óra)": "4h",
    "🌊 Swing (Napi)": "1d",
}

st.sidebar.header("📊 Kereskedés")
selected_asset_name = st.sidebar.selectbox("Instrumentum", list(assets.keys()))
selected_style_name = st.sidebar.selectbox("Idősík", list(styles.keys()))
balance_input = st.sidebar.number_input("Számla tőke ($)", value=10000.0, step=100.0)
risk_input = st.sidebar.number_input(
    "Kockázat (%)", value=1.0, step=0.1, min_value=0.1, max_value=5.0
)

if st.sidebar.button("🚀 Elemzés Indítása", use_container_width=True, type="primary"):
    if not gemini_api_input:
        st.error("Add meg a Gemini API kulcsot!")
    elif not twelvedata_api_input and "GC" not in selected_asset_name:
        st.error("Add meg a Twelve Data API kulcsot!")
    else:
        ticker = assets[selected_asset_name]
        interval = styles[selected_style_name]

        with st.spinner(f"Adatlekérés + AI elemzés ({ticker} • {interval})..."):
            try:
                img_path, df, source = generate_chart_image(
                    ticker, interval, twelvedata_api_input or "dummy"
                )
                last_close = float(df["Close"].iloc[-1])
                last_time = df.index[-1].strftime("%Y-%m-%d %H:%M")

                data = analyze_chart(
                    img_path,
                    gemini_api_input,
                    selected_asset_name,
                    selected_style_name,
                    last_close,
                )

                direction = data.get("direction", "NEUTRAL")
                entry = data.get("entry")
                sl = data.get("sl")
                tp = data.get("tp")
                confluences = data.get("confluences_found", [])
                missing = data.get("missing", [])
                reasoning = data.get("reasoning", "")
                model_used = data.get("_model_used", "—")

                st.markdown("---")
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    if direction == "BUY":
                        st.success("**Irány: BUY 🟢**")
                    elif direction == "SELL":
                        st.error("**Irány: SELL 🔴**")
                    else:
                        st.warning("**Irány: NEUTRAL 🟠**")
                with col_b:
                    st.metric("Utolsó ár", f"{last_close:.2f}")
                with col_c:
                    st.caption(f"Forrás: {source}")
                    st.caption(f"{last_time}")
                    st.caption(f"AI: {model_used}")

                st.markdown(
                    f"**Talált confluence-ek:** {', '.join(confluences) if confluences else '—'}"
                )
                if missing:
                    st.markdown(f"**Hiányzó feltételek:** {', '.join(missing)}")

                if direction != "NEUTRAL" and entry and sl and tp:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Belépő", f"{float(entry):.2f}")
                    c2.metric("Stop Loss", f"{float(sl):.2f}")
                    c3.metric("Take Profit", f"{float(tp):.2f}")
                    rr = calculate_risk_reward(direction, entry, sl, tp)
                    if rr:
                        c4.metric("R:R", f"1 : {rr}")

                    risk_usd, size = calculate_position_size(
                        balance_input, risk_input, entry, sl, ticker
                    )
                    if risk_usd and size:
                        st.info(
                            f"Kockázat: **${risk_usd:.2f}**  |  Ajánlott méret: **{size}**"
                        )

                st.markdown("### 📝 Indoklás")
                st.write(reasoning)

                st.image(img_path, use_container_width=True)
                st.caption(
                    f"Gyertyák: {len(df)} | Generálva: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

            except Exception as e:
                st.error(f"**Hiba:** {e}")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Mesterlövész v2.4 • Modell fallback: 3.6 → 2.5 → 2.0 • yfinance GC fallback"
)
