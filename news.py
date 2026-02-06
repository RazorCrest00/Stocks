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
            url = r.get("url")
            if url and url not in links:
                links.append(url)
    return links

def is_blocked_domain(url):
    domain = urlparse(url).netloc.lower()
    return any(bad in domain for bad in BLOCKED_DOMAINS)

def crawl_article(url):
    try:
        if is_blocked_domain(url):
            return None

        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200 or len(resp.text) < 2000:
            return None

        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if text and len(text.strip()) > 300:
            return text.strip()
        return None
    except Exception:
        return None

def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            return None
        return round(float(data["Close"].iloc[-1]), 2)
    except Exception:
        return None

def analyze_with_groq(news_text, price, ticker):
    prompt = f"""
You are a financial analyst.

Stock: {ticker}
Current Price: {price}

News Articles:
{news_text[:3500]}

Tasks:
1. Sentiment (Positive / Negative / Neutral)
2. Does news justify the price?
3. Actionable insight (max 3 lines)

Respond in plain text.
"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
        data = response.json()
        if "choices" not in data:
            return "Groq API did not return a valid completion.\n\n" + f"Response:\n{data}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq API call failed: {str(e)}"

def chat_with_groq(ticker, price, evaluation_text, chat_history, user_message):
    if not GROQ_API_KEY:
        return "GROQ_API_KEY not set."

    eval_context = (evaluation_text or "")[:2500]

    system_prompt = f"""
You are a stock research assistant inside a Streamlit app.

Use the provided evaluation context as your primary reference.
Discuss plausible bullish/bearish scenarios and what would need to happen for price to rise/fall.
Do not claim certainty or guarantee future price movements.
If asked "will it go up", respond with a scenario-based answer and key risks.

Stock: {ticker}
Current Price: {price}

Evaluation Context:
{eval_context}

Answer format rules:
- Be concise.
- Plain text.
- If you reference up/down, clarify it is a hypothesis, not a guarantee.
"""

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history[-20:])
    messages.append({"role": "user", "content": user_message})

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": messages, "temperature": 0.3}

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
        data = response.json()
        if "choices" not in data:
            return f"Groq API did not return a valid completion.\n\nResponse:\n{data}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq API call failed: {str(e)}"

def _render_bubbles(messages):
    parts = []
    for m in messages:
        role = m.get("role", "assistant")
        txt = _html.escape(m.get("content", ""))
        txt = txt.replace("\n", "<br/>")
        cls = "chat-bubble user" if role == "user" else "chat-bubble bot"
        parts.append(f'<div class="{cls}">{txt}</div>')
    return "\n".join(parts)

def _chat_header_html(ticker, price, collapsed=False):
    sub = f"{ticker} • {price}" if ticker and price is not None else ""
    cls = "chat-header collapsed" if collapsed else "chat-header"
    return f"""
<div class="{cls}">
  <div class="chat-h-left">
    <div class="chat-h-icon" aria-hidden="true">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <path d="M7.5 10.5h9" stroke="white" stroke-width="2" stroke-linecap="round"/>
        <path d="M9 14.5h6" stroke="white" stroke-width="2" stroke-linecap="round"/>
        <path d="M12 3c4.97 0 9 3.58 9 8s-4.03 8-9 8c-1.06 0-2.08-.16-3.02-.46L4 20l1.6-3.8C4.6 14.98 3 12.59 3 11c0-4.42 4.03-8 9-8z" fill="rgba(255,255,255,0.14)"/>
      </svg>
    </div>
    <div class="chat-h-text">
      <div class="chat-h-title">Assistant</div>
      <div class="chat-h-sub">{_html.escape(sub)}</div>
    </div>
  </div>
  <div class="chat-h-right">
    <div class="chat-h-square">C</div>
  </div>
</div>
"""

st.set_page_config(page_title="News vs Stock Analyzer", layout="wide")

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""
if "last_price" not in st.session_state:
    st.session_state.last_price = None
if "last_evaluation" not in st.session_state:
    st.session_state.last_evaluation = ""
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_draft" not in st.session_state:
    st.session_state.chat_draft = ""

st.markdown(
    """
<style>
#chat_anchor { display:none; }

div:has(> div > div > div > #chat_anchor){
  position: fixed !important;
  right: 18px !important;
  bottom: 18px !important;
  z-index: 999999 !important;
  height: 0 !important;
  overflow: visible !important;
  width: 360px !important;
  max-width: calc(100vw - 36px) !important;
}

.chat-shell{
  border-radius: 22px;
  overflow: hidden;
  box-shadow: 0 18px 60px rgba(0,0,0,0.55);
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(14,14,18,0.70);
  backdrop-filter: blur(10px);
}

.chat-header{
  height: 62px;
  display:flex;
  align-items:center;
  justify-content: space-between;
  padding: 12px 12px;
  background: linear-gradient(90deg, #6D28D9, #5B21B6);
  color: white;
}
.chat-header.collapsed{
  height: 56px;
}

.chat-h-left{
  display:flex;
  align-items:center;
  gap: 10px;
}
.chat-h-icon{
  width: 38px;
  height: 38px;
  border-radius: 16px;
  background: rgba(255,255,255,0.16);
  display:grid;
  place-items:center;
}
.chat-h-text{
  display:flex;
  flex-direction: column;
  gap: 2px;
}
.chat-h-title{
  font-weight: 800;
  font-size: 18px;
  letter-spacing: 0.2px;
  line-height: 1.1;
}
.chat-h-sub{
  font-size: 11px;
  opacity: 0.92;
}

.chat-h-right{
  display:flex;
  align-items:center;
  gap: 8px;
}
.chat-h-square{
  width: 44px;
  height: 44px;
  border-radius: 14px;
  background: rgba(9,9,20,0.82);
  display:grid;
  place-items:center;
  font-weight: 900;
  font-size: 20px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.35);
}

.chat-body{
  height: 380px;
  max-height: calc(100vh - 220px);
  overflow-y: auto;
  padding: 12px 12px 10px 12px;
  background: rgba(255,255,255,0.04);
}

.chat-bubble{
  max-width: 86%;
  padding: 10px 12px;
  border-radius: 18px;
  margin: 8px 0;
  font-size: 14px;
  line-height: 1.25rem;
  border: 1px solid rgba(255,255,255,0.10);
}
.chat-bubble.user{
  margin-left: auto;
  background: rgba(109,40,217,0.22);
  border-top-right-radius: 10px;
}
.chat-bubble.bot{
  margin-right: auto;
  background: rgba(255,255,255,0.10);
  border-top-left-radius: 10px;
}

.chat-input-wrap{
  padding: 10px 12px 12px 12px;
  border-top: 1px solid rgba(255,255,255,0.10);
  background: rgba(14,14,18,0.78);
}

div:has(> div > div > div > #chat_anchor) .stButton{ margin:0 !important; }
div:has(> div > div > div > #chat_anchor) button{
  border-radius: 14px !important;
  height: 42px !important;
}
</style>
""",
    unsafe_allow_html=True
)

st.title("News vs Stock Price Analyzer")
st.caption("Adaptive crawling • Free stack • Real-world safe")

ticker = st.text_input(
    "Enter Stock Ticker (e.g. AAPL, TSLA, INFY)",
    placeholder="AAPL"
)

analyze_clicked = st.button("Analyze")

if analyze_clicked and ticker:
    ticker = ticker.strip().upper()

    with st.spinner("Fetching live stock price..."):
        price = get_stock_price(ticker)

    if not price:
        st.error("Could not fetch stock price. Check ticker.")
        st.stop()

    st.success(f"Current Price: {price}")

    with st.spinner("Searching news sources..."):
        links = search_news(ticker)

    if not links:
        st.error("No news links found.")
        st.stop()

    st.subheader("Discovered News Links")
    for l in links[:10]:
        st.markdown(f"- {l}")

    st.subheader("Crawled Articles")

    successful_articles = []
    attempted = 0

    for link in links:
        if len(successful_articles) >= 5:
            break

        attempted += 1
        with st.spinner(f"Trying source {attempted}..."):
            text = crawl_article(link)

        if text:
            successful_articles.append(text)
            st.success(f"Source {attempted}: extracted ✔")
        else:
            st.warning(f"Source {attempted}: blocked / failed ")

        time.sleep(0.8)

    if not successful_articles:
        st.error("Could not extract content from any source.")
        st.stop()

    combined_text = "\n\n".join(successful_articles)

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set.")
        st.stop()

    st.subheader("AI Evaluation")
    with st.spinner("Analyzing news vs price..."):
        result = analyze_with_groq(combined_text, price, ticker)

    if st.session_state.last_ticker and st.session_state.last_ticker != ticker:
        st.session_state.chat_messages = []
        st.session_state.chat_open = False
        st.session_state.chat_draft = ""

    st.session_state.last_ticker = ticker
    st.session_state.last_price = price
    st.session_state.last_evaluation = result

    st.markdown(result)
    st.divider()

st.subheader("Market Snapshot: Top Stocks")

TOP_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC"]

@st.cache_data(ttl=900)
def load_stock_history(tickers):
    data = {}
    for t in tickers:
        hist = yf.Ticker(t).history(period="6mo")
        if not hist.empty:
            data[t] = hist
    return data

stock_data = load_stock_history(TOP_STOCKS)

c1, c2 = st.columns(2)

with c1:
    st.markdown("**1. Price Trends (6 months)**")
    st.line_chart({k: v["Close"] for k, v in stock_data.items()})

with c2:
    st.markdown("**2. Trading Volume (6 months)**")
    st.area_chart({k: v["Volume"] for k, v in stock_data.items()})

st.markdown("**3. Latest Closing Prices**")
latest_prices = {k: float(v["Close"].iloc[-1]) for k, v in stock_data.items()}
st.bar_chart(latest_prices)

st.markdown("**4. 7-Day % Change**")
pct_change = {
    k: round(((v["Close"].iloc[-1] / v["Close"].iloc[-7]) - 1) * 100, 2)
    for k, v in stock_data.items() if len(v) >= 7
}
st.bar_chart(pct_change)

st.markdown("**5. AAPL vs MSFT Price Comparison**")
compare_df = {"AAPL": stock_data["AAPL"]["Close"], "MSFT": stock_data["MSFT"]["Close"]}
st.line_chart(compare_df)

st.markdown("**6. NVDA Momentum (Close Price)**")
st.area_chart(stock_data["NVDA"]["Close"])

st.markdown("**7. TSLA Volatility**")
st.line_chart(stock_data["TSLA"]["High"] - stock_data["TSLA"]["Low"])

st.markdown("**8. Average Daily Volume**")
avg_volume = {k: int(v["Volume"].mean()) for k, v in stock_data.items()}
st.bar_chart(avg_volume)

st.markdown("**9. META Growth Curve**")
st.line_chart(stock_data["META"]["Close"])

st.markdown("**10. Semiconductor Performance (AMD vs INTC)**")
semi_df = {"AMD": stock_data["AMD"]["Close"], "INTC": stock_data["INTC"]["Close"]}
st.line_chart(semi_df)

show_chat = bool(st.session_state.last_evaluation) and bool(st.session_state.last_ticker)

if show_chat:
    st.markdown('<div id="chat_anchor"></div>', unsafe_allow_html=True)

    tkr = st.session_state.last_ticker
    prc = st.session_state.last_price

    if not st.session_state.chat_open:
        st.markdown(
            f"""
<div class="chat-shell">
  {_chat_header_html(tkr, prc, collapsed=True)}
</div>
""",
            unsafe_allow_html=True
        )
        if st.button("Open", type="primary", key="chat_open_btn"):
            st.session_state.chat_open = True
            st.rerun()

    else:
        st.markdown(
            f"""
<div class="chat-shell">
  {_chat_header_html(tkr, prc, collapsed=False)}
</div>
""",
            unsafe_allow_html=True
        )

        bubbles_html = _render_bubbles(st.session_state.chat_messages[-60:])
        st.markdown(f'<div class="chat-body">{bubbles_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="chat-input-wrap">', unsafe_allow_html=True)
        in_col, send_col, min_col = st.columns([6, 2, 1])
        with in_col:
            st.session_state.chat_draft = st.text_input(
                "",
                value=st.session_state.chat_draft,
                placeholder="Type a message…",
                key="chat_text_input"
            )
        with send_col:
            send = st.button("Send", type="primary", key="chat_send_btn")
        with min_col:
            if st.button("–", type="primary", key="chat_min_btn"):
                st.session_state.chat_open = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if send and st.session_state.chat_draft.strip():
            user_msg = st.session_state.chat_draft.strip()
            st.session_state.chat_draft = ""
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})

            assistant_reply = chat_with_groq(
                ticker=st.session_state.last_ticker,
                price=st.session_state.last_price,
                evaluation_text=st.session_state.last_evaluation,
                chat_history=st.session_state.chat_messages,
                user_message=user_msg
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})
            st.rerun()
