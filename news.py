import streamlit as st
import requests
import trafilatura
import yfinance as yf
from ddgs import DDGS
import os
import time
from urllib.parse import urlparse

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

You MUST use the provided evaluation context as your primary reference.
You may discuss possible bullish/bearish scenarios and what would need to happen for price to rise/fall.
You MUST NOT claim certainty or guarantee future price movements.
If asked "will it go up", respond with a probability-style, scenario-based answer and key risks.

Stock: {ticker}
Current Price: {price}

Evaluation Context (from earlier LLM analysis):
{eval_context}

Answer format rules:
- Be concise.
- Use plain text.
- If you reference "up/down", clarify it is a hypothesis, not a guarantee.
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

st.set_page_config(page_title="News vs Stock Analyzer", layout="wide")

st.title("News vs Stock Price Analyzer")
st.caption("Adaptive crawling • Free stack • Real-world safe")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "last_evaluation" not in st.session_state:
    st.session_state.last_evaluation = ""

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""

if "last_price" not in st.session_state:
    st.session_state.last_price = None

ticker = st.text_input(
    "Enter Stock Ticker (e.g. AAPL, TSLA, INFY)",
    placeholder="AAPL"
)

if st.button("Analyze") and ticker:
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

    st.session_state.last_evaluation = result
    st.session_state.last_ticker = ticker
    st.session_state.last_price = price

    st.markdown(result)
    st.divider()

st.subheader("Chat about this stock (context-aware)")

if not st.session_state.last_evaluation:
    st.info("Run Analyze first so the chatbot has context from the Groq evaluation.")
else:
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Ask a question about the stock (e.g., 'Bullish or bearish?' or 'What would make it go up?')")

    if user_msg:
        with st.chat_message("user"):
            st.markdown(user_msg)

        history_for_model = st.session_state.chat_messages[-20:]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                assistant_reply = chat_with_groq(
                    ticker=st.session_state.last_ticker,
                    price=st.session_state.last_price,
                    evaluation_text=st.session_state.last_evaluation,
                    chat_history=history_for_model,
                    user_message=user_msg
                )
                st.markdown(assistant_reply)

        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})

    st.caption("Not financial advice. This is an automated analysis based on recent scraped news and may be incomplete or wrong.")

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
