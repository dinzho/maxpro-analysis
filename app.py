import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
from functools import wraps
import plotly.graph_objects as go

# === 1. 頁面配置 ===
st.set_page_config(page_title="股票深度分析報告", page_icon="📊", layout="wide")

# === 2. 全局樣式配置 ===
st.markdown("""
    <style>
    .report-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: #1f2937; line-height: 1.6; }
    .section-card { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .section-title { font-size: 1.25rem; font-weight: 600; color: #111827; margin-bottom: 12px; border-bottom: 2px solid #f3f4f6; padding-bottom: 8px; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 12px; }
    .metric-box { background: #f9fafb; padding: 10px; border-radius: 6px; text-align: center; }
    .metric-label { font-size: 0.85rem; color: #6b7280; }
    .metric-value { font-size: 1.1rem; font-weight: 600; color: #111827; }
    .tag-green { color: #059669; font-weight: 600; }
    .tag-red { color: #dc2626; font-weight: 600; }
    .tag-orange { color: #d97706; font-weight: 600; }
    hr { border-top: 1px solid #e5e7eb; margin: 20px 0; }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# === 3. 緩存與數據獲取 ===
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1d")
        if df.empty or len(df) < 30:
            return None, None, None, None, None, "數據不足或代碼錯誤，請檢查股票代碼。"
        
        # 清理欄位名 (兼容 yfinance 多索引或後綴)
        df.columns = [str(c).lower().split('_')[0] for c in df.columns]
        if 'close' not in df.columns:
            return None, None, None, None, None, "無法解析收盤價欄位。"
            
        # 獲取基本面與新聞 (帶重試)
        info = {}
        for _ in range(3):
            try:
                info = stock.info
                if info and 'sector' in info: break
                time.sleep(1)
            except: time.sleep(1)
                
        news_data = getattr(stock, 'news', []) or []
        recommendations = getattr(stock, 'recommendations', None)
        return stock, df, info, news_data, recommendations, None
    except Exception as e:
        return None, None, None, None, None, f"數據獲取失敗：{str(e)}"

# === 4. 技術指標計算 ===
@st.cache_data
def calc_indicators(df):
    df = df.copy()
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    
    df['SMA20'] = df['close'].rolling(20, min_periods=1).mean()
    df['SMA50'] = df['close'].rolling(50, min_periods=1).mean()
    df['SMA200'] = df['close'].rolling(200, min_periods=1).mean()
    return df

# === 5. 圖表生成 ===
def generate_chart(df, fib_levels):
    plot_df = df.tail(252).copy()
    if plot_df.empty: return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df['open'], high=plot_df['high'],
        low=plot_df['low'], close=plot_df['close'], name="K線",
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='#10b981', decreasing_fillcolor='#ef4444'
    ))
    for ratio, y in fib_levels.items():
        fig.add_hline(y=y, line_dash="dot", line_color="#f59e0b", line_width=1, opacity=0.6)
        fig.add_annotation(y=y, x=1.01, text=f"{ratio}", showarrow=False,
                           font=dict(size=10, color="#f59e0b"), xref="paper", yref="y",
                           xanchor="left", yanchor="middle")
    fig.update_layout(title="K線與斐波那契回撤", yaxis_title="價格", xaxis_rangeslider_visible=False,
                      height=480, margin=dict(l=0, r=40, t=30, b=0), plot_bgcolor='white',
                      paper_bgcolor='white', hovermode='x unified')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb', gridwidth=0.5)
    return fig

# === 6. 市場情緒與行業分析 ===
def get_sentiment(df, current_price, sma20, recent_low):
    tech = "🟢 多頭" if current_price > sma20.iloc[-1] else "🔴 空頭"
    vol = df['volume'].iloc[-1]
    avg_vol = df['volume'].rolling(10, min_periods=1).mean().iloc[-1]
    vol_chg = ((vol / avg_vol - 1) * 100) if avg_vol > 0 else 0
    vol_str = f"📈 放量 (+{vol_chg:.0f}%)" if vol_chg > 20 else f"📉 縮量 ({vol_chg:.0f}%)" if vol_chg < -20 else "➡️ 平量"
    
    try:
        vix = yf.Ticker("^VIX").history(period="1d", interval="1d")
        vix_val = vix['Close'].iloc[-1] if not vix.empty else 20
        spy = yf.Ticker("^GSPC").history(period="5d", interval="1d")
        spy_chg = ((spy['Close'].iloc[-1] - spy['Close'].iloc[-2]) / spy['Close'].iloc[-2]) * 100 if len(spy) >= 2 else 0
    except:
        vix_val, spy_chg = 20, 0
        
    vix_mood = "😰 恐慌" if vix_val > 30 else "😟 謹慎" if vix_val > 25 else "😌 樂觀" if vix_val < 18 else "😐 中性"
    stock_chg = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
    rel = "🟢 強於大盤" if stock_chg > spy_chg + 2 else "🔴 弱於大盤" if stock_chg < spy_chg - 2 else "🟡 同步大盤"
    
    score = 50
    if current_price > sma20.iloc[-1]: score += 15
    if vol_chg > 0: score += 10
    if stock_chg > spy_chg: score += 15
    if vix_val < 20: score += 10
    score = max(0, min(100, score))
    rec = "✅ 積極操作" if score >= 70 else "⚖️ 控制倉位" if score >= 55 else "⚠️ 謹慎觀望" if score >= 40 else "❌ 防守為主"
    return {'tech': tech, 'vol': vol_str, 'vix': vix_val, 'vix_mood': vix_mood, 'spy_chg': spy_chg, 'rel': rel, 'score': score, 'rec': rec, 'stock_chg': stock_chg}

def analyze_industry(sector, industry, gross_margin, roe):
    s = str(sector).lower() if sector else ""
    i = str(industry).lower() if industry else ""
    if any(x in s for x in ['technology', 'communication']) or any(x in i for x in ['software', 'internet']):
        return "科技成長型", "中上游 (核心技術/平台)", "中等 (依賴人才/算力)", "強勢 (高轉換成本/訂閱制)"
    elif 'consumer' in s:
        return "消費驅動型", "下游 (品牌終端)", "強勢 (規模壓價)", "弱勢 (價格敏感)"
    elif any(x in s for x in ['industrials', 'materials', 'energy']):
        return "週期製造型", "中上游 (原材料/設備)", "弱勢 (受制大宗)", "中等"
    elif 'financial' in s:
        return "資金密集型", "服務中介", "弱勢 (資金成本)", "中等"
    return "一般行業", "中游", "中等", "中等"

# === 7. 主程序 ===
if __name__ == "__main__":
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🔍 查詢設置")
        ticker_input = st.text_input("股票代碼", value="AAPL", placeholder="例如：AAPL, 09988.HK, TSM")
        analyze_btn = st.button("🚀 開始深度分析", type="primary", use_container_width=True)
        st.markdown("---")
        st.info("💡 數據來源：Yahoo Finance | 延遲約15分鐘")

    if analyze_btn and ticker_input.strip():
        ticker = ticker_input.strip().upper()
        with st.spinner("🔍 正在生成深度報告..."):
            stock, df, info, news_data, recommendations, error = fetch_stock_data(ticker)
            
            if error:
                st.error(f"❌ {error}")
            else:
                df = calc_indicators(df)
                current_price = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-2]
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                recent_high = df['high'].max()
                recent_low = df['low'].min()
                high_52w = df['high'].rolling(252, min_periods=1).max().iloc[-1]
                low_52w = df['low'].rolling(252, min_periods=1).min().iloc[-1]
                last_date = df.index[-1].strftime('%Y-%m-%d')
                
                drop_range = recent_high - recent_low if recent_high != recent_low else 1
                fib_levels = {
                    '100%': recent_high, '78.6%': recent_low + drop_range * 0.214,
                    '61.8%': recent_low + drop_range * 0.618, '50%': recent_low + drop_range * 0.5,
                    '38.2%': recent_low + drop_range * 0.382, '23.6%': recent_low + drop_range * 0.236, '0%': recent_low
                }
                
                macd_val, sig_val = df['MACD'].iloc[-1], df['Signal'].iloc[-1]
                rsi_val = df['RSI'].iloc[-1]
                sma20, sma50, sma200 = df['SMA20'].iloc[-1], df['SMA50'].iloc[-1], df['SMA200'].iloc[-1]
                fib_pos = ((current_price - recent_low) / drop_range) * 100
                
                sentiment = get_sentiment(df, current_price, df['SMA20'], recent_low)
                
                def safe_get(k, d="N/A"):
                    try:
                        v = info.get(k)
                        return v if v is not None and not pd.isna(v) else d
                    except: return d
                
                sector, industry = safe_get('sector'), safe_get('industry')
                gross_margin = safe_get('grossMargins')
                roe = safe_get('returnOnEquity')
                pe = safe_get('trailingPE')
                rev_growth = safe_get('revenueGrowth')
                mcap = safe_get('marketCap')
                
                ind_type, ind_pos, ind_up, ind_down = analyze_industry(sector, industry, gross_margin, roe)
                
                fib_names = {'23.6%': '23.6%', '38.2%': '38.2%', '50%': '50.0%', '61.8%': '61.8%', '78.6%': '78.6%'}
                all_lvls = [{'n': v, 'p': fib_levels[k]} for k, v in fib_names.items()]
                all_lvls += [{'n': '近期低點', 'p': recent_low}, {'n': '近期高點', 'p': recent_high}]
                
                res = sorted([x for x in all_lvls if x['p'] > current_price * 1.005], key=lambda y: y['p'])
                sup = sorted([x for x in all_lvls if x['p'] < current_price * 0.995], key=lambda y: y['p'], reverse=True)
                
                s1 = sup[0]['p'] if sup else recent_low
                s2 = sup[1]['p'] if len(sup) > 1 else s1 * 0.95
                s3 = sup[2]['p'] if len(sup) > 2 else s1 * 0.85
                r1 = res[0]['p'] if res else recent_high
                r2 = res[1]['p'] if len(res) > 1 else r1 * 1.1
                r3 = res[2]['p'] if len(res) > 2 else r1 * 1.2
                
                entry_short = s1 * 1.01
                entry_long = s2 * 1.02
                sl_short = s1 * 0.97
                sl_long = s3 * 0.95
                tp1, tp2, tp3 = r1 * 0.99, r2 * 0.98, r3 * 0.97
                
                rating_sc = 0
                if roe and roe > 0.15: rating_sc += 2
                if rev_growth and rev_growth > 0.15: rating_sc += 2
                if pe and 10 < pe < 30: rating_sc += 1
                if current_price > sma200: rating_sc += 1
                if sentiment['score'] >= 60: rating_sc += 1
                rating_txt = "🟢 強烈買入" if rating_sc >= 6 else "🟡 買入/增持" if rating_sc >= 4 else "🟠 持有/觀望" if rating_sc >= 2 else "🔴 減持/規避"
                
                risk_sc = 50
                risk_reasons = []
                if macd_val < 0: risk_sc += 15; risk_reasons.append("MACD空頭")
                if current_price < sma200: risk_sc += 15; risk_reasons.append("低於年線")
                if rsi_val > 70: risk_sc += 10; risk_reasons.append("RSI超買")
                elif rsi_val < 30: risk_sc -= 10
                risk_sc = max(0, min(100, risk_sc))
                risk_txt = "； ".join(risk_reasons) if risk_reasons else "指標中性"

                # === 渲染區塊 1~10 ===
                st.markdown(f"""
                 <div class="section-card">
                     <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                         <div>
                             <h2 style="margin:0; font-size:1.5rem;">{safe_get('longName', ticker)} ({ticker})</h2>
                             <span style="color:#6b7280; font-size:0.9rem;">數據基準：{last_date} 收盤</span>
                         </div>
                         <div style="text-align:right;">
                             <div style="font-size:1.8rem; font-weight:700; color:{'#10b981' if change_pct >=0 else '#ef4444'};">${current_price:.2f}</div>
                             <div style="color:{'#10b981' if change_pct >=0 else '#ef4444'}; font-weight:600;">{change_pct:+.2f}%</div>
                         </div>
                     </div>
                     <div style="margin-top:8px; color:#4b5563; font-size:0.95rem;">
                        52周範圍：${low_52w:.2f} - ${high_52w:.2f} &nbsp;|&nbsp; 市值：${mcap/1e9:.2f}B
                     </div>
                 </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.plotly_chart(generate_chart(df, fib_levels), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                macd_st = '<span class="tag-green">多頭</span>' if macd_val > sig_val else '<span class="tag-red">空頭</span>'
                rsi_st = '<span class="tag-red">超買</span>' if rsi_val > 70 else '<span class="tag-green">超賣</span>' if rsi_val < 30 else '中性'
                ma_st = '<span class="tag-green">✅</span>' if current_price > sma20 else '<span class="tag-red">⚠️</span>'
                ma200_st = '<span class="tag-green">✅</span>' if current_price > sma200 else '<span class="tag-red">🔴</span>'
                
                st.markdown('<div class="section-card"><div class="section-title">3. 技術指標</div>', unsafe_allow_html=True)
                st.markdown(f"""
                 <div class="metric-grid">
                     <div class="metric-box"><div class="metric-label">MACD</div><div class="metric-value">{macd_st}</div></div>
                     <div class="metric-box"><div class="metric-label">RSI (14)</div><div class="metric-value">{rsi_val:.1f} {rsi_st}</div></div>
                     <div class="metric-box"><div class="metric-label">均線狀態</div><div class="metric-value">SMA20 {ma_st} | SMA200 {ma200_st}</div></div>
                     <div class="metric-box"><div class="metric-label">FIB 位置</div><div class="metric-value">{fib_pos:.1f}% 回撤位</div></div>
                 </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                trend_txt = "MACD多頭排列，短線動能偏強" if macd_val > sig_val and macd_val > 0 else "MACD空頭排列，短線動能偏弱" if macd_val < sig_val and macd_val < 0 else "MACD糾結，方向不明"
                wave_txt = f"當前位於本輪波段 {fib_pos:.1f}% 回撤區間，屬於{'上升浪後的深度回測' if fib_pos < 50 else '高位震盪整理'}，若守住關鍵支撐有望開啟反彈，跌破則趨勢轉空。"
                
                st.markdown('<div class="section-card"><div class="section-title">4. 技術結構</div>', unsafe_allow_html=True)
                st.markdown(f"""
                 <ul style="margin:0; padding-left:20px; color:#374151;">
                     <li><b>趨勢判斷：</b>{trend_txt}</li>
                     <li><b>波浪位置：</b>{wave_txt}</li>
                     <li><b>均線結論：</b>股價{"站上" if current_price > sma20 else "受壓於"}SMA20，{"處於長期多頭格局" if current_price > sma200 else "長期趨勢仍偏弱"}。</li>
                 </ul>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-card"><div class="section-title">5. 基本面分析</div>', unsafe_allow_html=True)
                st.markdown(f"""
                 <ul style="margin:0; padding-left:20px; color:#374151;">
                     <li><b>行業定位：</b>{ind_type} | {ind_pos}</li>
                     <li><b>所屬板塊：</b>{sector} / {industry}</li>
                     <li><b>議價能力：</b>對上游 {ind_up} | 對下游 {ind_down}</li>
                     <li><b>護城河評估：</b>{"✅ 具備定價權與轉換成本優勢" if ind_down == "強勢 (高轉換成本/訂閱制)" else "⚖️ 行業競爭激烈，需靠效率與規模取勝"}</li>
                 </ul>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                roe_txt = "✅ 優秀 (>15%)" if roe and roe > 0.15 else "⚠️ 一般" if roe else "N/A"
                grow_txt = "🚀 高速 (>20%)" if rev_growth and rev_growth > 0.2 else "📈 穩健" if rev_growth and rev_growth > 0.1 else "🐢 放緩" if rev_growth else "N/A"
                margin_txt = "💰 高毛利 (>50%)" if gross_margin and gross_margin > 0.5 else "🏭 標準" if gross_margin else "N/A"
                pe_txt = "💸 高估值" if pe and pe > 40 else "⚖️ 合理" if pe and 15 < pe < 40 else "💰 低估值" if pe and pe < 15 else "N/A"
                
                st.markdown('<div class="section-card"><div class="section-title">6. 財報指標與投資評級</div>', unsafe_allow_html=True)
                st.markdown(f"""
                 <table style="width:100%; border-collapse:collapse; margin-bottom:12px;">
                     <tr style="background:#f9fafb;"><th style="padding:8px; text-align:left; border-bottom:1px solid #e5e7eb;">指標</th><th style="padding:8px; text-align:left; border-bottom:1px solid #e5e7eb;">數值</th><th style="padding:8px; text-align:left; border-bottom:1px solid #e5e7eb;">評估</th></tr>
                     <tr><td style="padding:8px; border-bottom:1px solid #f3f4f6;">ROE</td><td style="padding:8px; border-bottom:1px solid #f3f4f6;">{roe*100:.1f}%</td><td style="padding:8px; border-bottom:1px solid #f3f4f6;">{roe_txt}</td></tr>
                     <tr><td style="padding:8px; border-bottom:1px solid #f3f4f6;">營收增長</td><td style="padding:8px; border-bottom:1px solid #f3f4f6;">{rev_growth*100:.1f}%</td><td style="padding:8px; border-bottom:1px solid #f3f4f6;">{grow_txt}</td></tr>
                     <tr><td style="padding:8px; border-bottom:1px solid #f3f4f6;">毛利率</td><td style="padding:8px; border-bottom:1px solid #f3f4f6;">{gross_margin*100:.1f}%</td><td style="padding:8px; border-bottom:1px solid #f3f4f6;">{margin_txt}</td></tr>
                     <tr><td style="padding:8px; border-bottom:1px solid #f3f4f6;">PE (TTM)</td><td style="padding:8px; border-bottom:1px solid #f3f4f6;">{pe:.1f}x</td><td style="padding:8px; border-bottom:1px solid #f3f4f6;">{pe_txt}</td></tr>
                 </table>
                 <div style="background:#f0fdf4; padding:10px; border-radius:6px; border-left:4px solid #10b981;">
                     <b>🏆 投資評級：{rating_txt}</b> (評分 {rating_sc}/7) <br>
                     <span style="font-size:0.9rem; color:#4b5563;">基於基本面質量、估值安全邊際與技術趨勢綜合打分。</span>
                 </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                res_html = " ".join([f"<li>{i}. {x['n']} ${x['p']:.2f}</li>" for i, x in enumerate(res[:5], 1)]) or "<li>暫無明顯壓力</li>"
                sup_html = " ".join([f"<li>{i}. {x['n']} ${x['p']:.2f}{' (緊鄰)' if i==1 else ''}</li>" for i, x in enumerate(sup[:5], 1)]) or "<li>暫無明顯支撐</li>"
                
                st.markdown('<div class="section-card"><div class="section-title">7. 關鍵位</div>', unsafe_allow_html=True)
                st.markdown(f"""
                 <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                     <div><b style="color:#ef4444;">🔺 壓力位（由近至遠）</b><ul style="margin:8px 0 0 0; padding-left:20px; color:#374151;">{res_html}</ul></div>
                     <div><b style="color:#10b981;">🔻 支撐位（由近至遠）</b><ul style="margin:8px 0 0 0; padding-left:20px; color:#374151;">{sup_html}</ul></div>
                 </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-card"><div class="section-title">8. 市場情緒</div>', unsafe_allow_html=True)
                st.markdown(f"""
                 <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:12px;">
                     <div class="metric-box"><div class="metric-label">個股技術</div><div class="metric-value">{sentiment['tech']}</div></div>
                     <div class="metric-box"><div class="metric-label">資金流向</div><div class="metric-value">{sentiment['vol']}</div></div>
                     <div class="metric-box"><div class="metric-label">標普 500</div><div class="metric-value" style="color:{'#10b981' if sentiment['spy_chg'] >=0 else '#ef4444'}">{sentiment['spy_chg']:+.2f}%</div></div>
                     <div class="metric-box"><div class="metric-label">VIX 恐慌指數</div><div class="metric-value">{sentiment['vix']:.1f} ({sentiment['vix_mood']})</div></div>
                 </div>
                 <div style="display:flex; justify-content:space-between; align-items:center; background:#f9fafb; padding:10px; border-radius:6px;">
                     <span><b>相對強弱：</b>{sentiment['rel']}</span>
                     <span><b>情緒評分：</b><span style="font-size:1.2rem; font-weight:700; color:{'#10b981' if sentiment['score'] >=60 else '#d97706' if sentiment['score'] >=40 else '#ef4444'}">{sentiment['score']}/100</span></span>
                 </div>
                 <div style="margin-top:8px; color:#4b5563;">💡 綜合建議：{sentiment['rec']}</div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                rr_short = ((tp1 - entry_short) / (entry_short - sl_short)) if entry_short > sl_short else 0
                rr_long = ((tp2 - entry_long) / (entry_long - sl_long)) if entry_long > sl_long else 0
                
                st.markdown('<div class="section-card"><div class="section-title">9. 入場與離場策略</div>', unsafe_allow_html=True)
                st.markdown(f"""
                 <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                     <div style="background:#f0fdf4; padding:12px; border-radius:6px; border:1px solid #bbf7d0;">
                         <b style="color:#15803d;">🟢 短線交易 (1-4週)</b><br>
                         <span style="font-size:0.9rem;">
                            • <b>回調入場</b>：${entry_short:.2f} (靠近第一支撐)<br>
                            • <b>離場/止盈</b>：T1 ${tp1:.2f} | T2 ${tp2:.2f}<br>
                            • <b>止損</b>：${sl_short:.2f} (跌破支撐 3%)<br>
                            • <b>風報比</b>：1:{rr_short:.1f}
                         </span>
                     </div>
                     <div style="background:#eff6ff; padding:12px; border-radius:6px; border:1px solid #bfdbfe;">
                         <b style="color:#1d4ed8;">🔵 長線佈局 (3-6月+)</b><br>
                         <span style="font-size:0.9rem;">
                            • <b>回調入場</b>：${entry_long:.2f} (靠近第二支撐)<br>
                            • <b>離場/止盈</b>：T2 ${tp2:.2f} | T3 ${tp3:.2f}<br>
                            • <b>止損</b>：${sl_long:.2f} (跌破第三支撐 5%)<br>
                            • <b>風報比</b>：1:{rr_long:.1f}
                         </span>
                     </div>
                 </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-card"><div class="section-title">10. 風險評分</div>', unsafe_allow_html=True)
                st.markdown(f"""
                 <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
                     <div style="width:60px; height:60px; border-radius:50%; background:{'#10b981' if risk_sc <=40 else '#f59e0b' if risk_sc <=70 else '#ef4444'}; color:white; display:flex; align-items:center; justify-content:center; font-size:1.5rem; font-weight:700;">{risk_sc}</div>
                     <div>
                         <div style="font-weight:600; font-size:1.1rem;">綜合風險等級：{'低' if risk_sc <=40 else '中' if risk_sc <=70 else '高'}</div>
                         <div style="color:#4b5563; font-size:0.95rem;">理由：{risk_txt}。當前價格貼近關鍵支撐，若跌破則下行空間打開；若守住則有反彈修復機會，多空博弈明顯。</div>
                     </div>
                 </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<hr><div style="text-align:center; color:#6b7280; font-size:0.85rem;">⚠️ 免責聲明：本報告僅供參考，不構成投資建議。市場有風險，決策需謹慎。</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    elif not analyze_btn:
        st.info("👈 請在左側輸入股票代碼並點擊「開始深度分析」")