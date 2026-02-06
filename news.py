import streamlit as st
import requests
import trafilatura
import yfinance as yf
from ddgs import DDGS
import os
import time
from urllib.parse import urlparse
import html as _html

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )
}

BLOCKED_DOMAINS = (
    "globeandmail.com",
    "seekingalpha.com",
    "benzinga.com",
    "wsj.com",
    "ft.com",
)

def search_news(query, max_results=40):
    links = []
    with DDGS() as ddgs:
        for r in ddgs.news(query, max_results=max_results):
            if r.get("url"):
                links.append(r["url"])
    return list(dict.fromkeys(links))

def is_blocked_domain(url):
    domain = urlparse(url).netloc.lower()
    return any(bad in domain for bad in BLOCKED_DOMAINS)

def crawl_article(url):
    try:
        if is_blocked_domain(url):
            return None
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200 or len(r.text) < 2000:
            return None
        txt = trafilatura.extract(r.text, include_comments=False, include_tables=False)
        return txt.strip() if txt and len(txt) > 300 else None
    except Exception:
        return None

def get_stock_price(ticker):
    try:
        h = yf.Ticker(ticker).history(period="1d")
        return round(float(h["Close"].iloc[-1]), 2) if not h.empty else None
    except Exception:
        return None

def analyze_with_groq(news_text, price, ticker):
    prompt = f"""
Stock: {ticker}
Current Price: {price}

News:
{news_text[:3500]}

Tasks:
1. Sentiment
2. Does news justify price?
3. Actionable insight (3 lines max)
"""
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    r = requests.post(
        GROQ_ENDPOINT,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    return r.json()["choices"][0]["message"]["content"]

def chat_with_groq(ticker, price, eval_text, history, user_msg):
    system = f"""
You are a stock assistant.
Use evaluation context.
Scenario based. No guarantees.

Stock: {ticker}
Price: {price}

Context:
{eval_text[:2500]}
"""
    msgs = [{"role": "system", "content": system}] + history[-20:] + [{"role": "user", "content": user_msg}]
    payload = {"model": GROQ_MODEL, "messages": msgs, "temperature": 0.3}
    r = requests.post(
        GROQ_ENDPOINT,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    return r.json()["choices"][0]["message"]["content"]

def bubbles(msgs):
    out = []
    for m in msgs:
        c = _html.escape(m["content"]).replace("\n", "<br>")
        cls = "bubble user" if m["role"] == "user" else "bubble bot"
        out.append(f'<div class="{cls}">{c}</div>')
    return "".join(out)

st.set_page_config(page_title="News vs Stock Analyzer", layout="wide")

for k, v in {
    "chat_open": False,
    "chat_msgs": [],
    "chat_draft": "",
    "last_eval": "",
    "last_price": None,
    "last_ticker": ""
}.items():
    st.session_state.setdefault(k, v)

st.markdown("""
<style>
#chat_anchor{display:none}
div:has(#chat_anchor){
 position:fixed;bottom:18px;right:18px;width:360px;z-index:99999;
}
.chat{
 border-radius:22px;overflow:hidden;
 box-shadow:0 20px 60px rgba(0,0,0,.6);
 background:rgba(14,14,18,.9)
}
.header{
 background:linear-gradient(90deg,#6D28D9,#5B21B6);
 color:white;padding:12px;display:flex;justify-content:space-between
}
.body{height:360px;overflow:auto;padding:12px}
.bubble{max-width:85%;padding:10px 12px;border-radius:16px;margin:6px 0}
.user{margin-left:auto;background:#6D28D933}
.bot{margin-right:auto;background:#ffffff1a}
.input{border-top:1px solid #ffffff22;padding:10px}
</style>
""", unsafe_allow_html=True)

st.title("News vs Stock Price Analyzer")

ticker = st.text_input("Ticker")
if st.button("Analyze") and ticker:
    ticker = ticker.upper()
    price = get_stock_price(ticker)
    links = search_news(ticker)
    articles = [crawl_article(l) for l in links if crawl_article(l)]
    result = analyze_with_groq("\n".join(articles[:5]), price, ticker)
    st.session_state.last_eval = result
    st.session_state.last_price = price
    st.session_state.last_ticker = ticker
    st.markdown(result)

st.subheader("Market Snapshot")
stocks = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AMD","INTC"]
data = {s: yf.Ticker(s).history(period="6mo") for s in stocks}
c1,c2 = st.columns(2)
with c1: st.line_chart({k:v["Close"] for k,v in data.items()})
with c2: st.area_chart({k:v["Volume"] for k,v in data.items()})

if st.session_state.last_eval:
    with st.container():
        st.markdown('<div id="chat_anchor"></div>', unsafe_allow_html=True)

        if not st.session_state.chat_open:
            st.markdown(f"""
<div class="chat">
 <div class="header">
  <b>Assistant</b>
  <button onclick="window.location.reload()">Open</button>
 </div>
</div>
""", unsafe_allow_html=True)
            if st.button("Open"):
                st.session_state.chat_open = True
                st.rerun()
        else:
            st.markdown(f"""
<div class="chat">
 <div class="header">
  <b>Assistant</b>
  <button onclick="window.location.reload()">–</button>
 </div>
 <div class="body">{bubbles(st.session_state.chat_msgs)}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown('<div class="input">', unsafe_allow_html=True)
            msg = st.text_input(" ", key="chat_input")
            if st.button("Send"):
                st.session_state.chat_msgs.append({"role":"user","content":msg})
                reply = chat_with_groq(
                    st.session_state.last_ticker,
                    st.session_state.last_price,
                    st.session_state.last_eval,
                    st.session_state.chat_msgs,
                    msg
                )
                st.session_state.chat_msgs.append({"role":"assistant","content":reply})
                st.session_state.chat_draft=""
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
