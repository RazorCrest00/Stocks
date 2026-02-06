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

        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False
        )

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
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        data = response.json()

        if "choices" not in data:
            return (
                "Groq API did not return a valid completion.\n\n"
                f"Response:\n{data}"
            )

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

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3
    }

    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
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
        cls = "bubble user" if role == "user" else "bubble bot"
        parts.append(f'<div class="{cls}">{txt}</div>')
    return "\n".join(parts)

st.set_page_config(page_title="News vs Stock Analyzer", layout="wide")

st.markdown(
    """
<style>
.chat-mini {
  position: fixed;
  right: 18px;
  bottom: 18px;
  z-index: 10000;
  width: 280px;
}
.chat-panel {
  position: fixed;
  right: 18px;
  bottom: 18px;
  z-index: 10000;
  width: 360px;
  max-width: calc(100vw - 36px);
  height: 520px;
  max-height: calc(100vh - 36px);
  background: rgba(18, 18, 22, 0.92);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.55);
  overflow: hidden;
  backdrop-filter: blur(10px);
}
.chat-topbar {
  height: 54px;
  background: linear-gradient(90deg, rgba(109,40,217,0.95), rgba(88,28,135,0.95));
  padding: 10px 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.chat-topbar-left {
  display: flex;
  align-items: center;
  gap: 10px;
  color: rgba(255,255,255,0.95);
  font-weight: 700;
}
.chat-badge {
  width: 30px;
  height: 30px;
  border-radius: 10px;
  background: rgba(255,255,255,0.16);
  display: grid;
  place-items: center;
  font-size: 16px;
}
.chat-sub {
  font-size: 12px;
  font-weight: 600;
  color: rgba(255,255,255,0.85);
  margin-top: 2px;
}
.chat-body {
  height: 394px;
  padding: 12px 12px 8px 12px;
  overflow-y: auto;
  background: rgba(255,255,255,0.03);
}
.bubble {
  max-width: 86%;
  padding: 10px 12px;
  border-radius: 16px;
  margin: 8px 0;
  font-size: 14px;
  line-height: 1.25rem;
  border: 1px solid rgba(255,255,255,0.10);
}
.bubble.user {
  margin-left: auto;
  background: rgba(109,40,217,0.22);
  border-top-right-radius: 8px;
}
.bubble.bot {
  margin-right: auto;
  background: rgba(255,255,255,0.08);
  border-top-left-radius: 8px;
}
.chat-input-wrap {
  padding: 10px 12px 12px 12px;
  border-top: 1px solid rgba(255,255,255,0.08);
  background: rgba(18,18,22,0.92);
}
.chat-note {
  font-size: 11px;
  color: rgba(255,255,255,0.55);
  margin-top: 6px;
}
.chat-mini-card {
  background: linear-gradient(90deg, rgba(109,40,217,0.95), rgba(88,28,135,0.95));
  border-radius: 14px;
  padding: 10px 12px;
  color: rgba(255,255,255,0.95);
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 14px 40px rgba(0,0,0,0.45);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.chat-mini-title {
  font-weight: 800;
  font-size: 13px;
}
.chat-mini-sub {
  font-size: 11px;
  opacity: 0.9;
  margin-top: 2px;
}
button[kind="primary"] {
  border-radius: 999px !important;
}
div[data-testid="stButton"] > button {
  padding-top: 0.45rem !important;
  padding-bottom: 0.45rem !important;
}
</style>
""",
    unsafe_allow_html=True
)

if "view" not in st.session_state:
    st.session_state.view = "home"

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "last_evaluation" not in st.session_state:
    st.session_state.last_evaluation = ""

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""

if "last_price" not in st.session_state:
    st.session_state.last_price = None

if "chat_draft" not in st.session_state:
    st.session_state.chat_draft = ""

st.title("News vs Stock Price Analyzer")
st.caption("Adaptive crawling • Free stack • Real-world safe")

ticker = st.text_input(
    "Enter Stock Ticker (e.g. AAPL, TSLA, INFY)",
    placeholder="AAPL"
)

analyze_clicked = st.button("Analyze")

if analyze_clicked:
    if not ticker or not ticker.strip():
        st.session_state.view = "home"
        st.session_state.chat_open = False
    else:
        ticker = ticker.strip().upper()
        st.session_state.view = "analysis"

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

        st.session_state.last_evaluation = result
        st.session_state.last_ticker = ticker
        st.session_state.last_price = price

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
compare_df = {
    "AAPL": stock_data["AAPL"]["Close"],
    "MSFT": stock_data["MSFT"]["Close"]
}
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
semi_df = {
    "AMD": stock_data["AMD"]["Close"],
    "INTC": stock_data["INTC"]["Close"]
}
st.line_chart(semi_df)

show_chat = (st.session_state.view == "analysis") and bool(st.session_state.last_evaluation) and bool(st.session_state.last_ticker)

if show_chat:
    if not st.session_state.chat_open:
        st.markdown('<div class="chat-mini">', unsafe_allow_html=True)
        left, right = st.columns([4, 1])
        with left:
            st.markdown(
                f"""
<div class="chat-mini-card">
  <div>
    <div class="chat-mini-title">Assistant</div>
    <div class="chat-mini-sub">{st.session_state.last_ticker} • {st.session_state.last_price}</div>
  </div>
</div>
""",
                unsafe_allow_html=True
            )
        with right:
            if st.button("💬", type="primary", key="open_chat_btn"):
                st.session_state.chat_open = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.chat_open:
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)

        top_left, top_right = st.columns([5, 1])
        with top_left:
            st.markdown(
                f"""
<div class="chat-topbar">
  <div class="chat-topbar-left">
    <div class="chat-badge">🤖</div>
    <div>
      <div>Assistant</div>
      <div class="chat-sub">{st.session_state.last_ticker} • {st.session_state.last_price}</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True
            )
        with top_right:
            if st.button("–", type="primary", key="min_chat_btn"):
                st.session_state.chat_open = False
                st.rerun()

        bubbles_html = _render_bubbles(st.session_state.chat_messages[-50:])
        st.markdown(f'<div class="chat-body">{bubbles_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="chat-input-wrap">', unsafe_allow_html=True)
        c_in, c_send = st.columns([5, 1])
        with c_in:
            st.session_state.chat_draft = st.text_input(
                "",
                value=st.session_state.chat_draft,
                placeholder="Type a message…",
                key="chat_text_input"
            )
        with c_send:
            send = st.button("Send", type="primary", key="chat_send_btn")

        st.markdown('<div class="chat-note">Not financial advice.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if send and st.session_state.chat_draft.strip():
            user_msg = st.session_state.chat_draft.strip()
            st.session_state.chat_draft = ""
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})

            history_for_model = st.session_state.chat_messages[-20:]
            assistant_reply = chat_with_groq(
                ticker=st.session_state.last_ticker,
                price=st.session_state.last_price,
                evaluation_text=st.session_state.last_evaluation,
                chat_history=history_for_model,
                user_message=user_msg
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.session_state.chat_open = False
