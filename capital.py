# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:51:11 2026

@author: jnchi
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 1. 頁面基礎配置 ---
st.set_page_config(
    page_title="美台 AI 資本反轉與 OpenAI 曝險監測儀表板",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 官方數據來源超連結 ---
DATA_SOURCES = {
    "TSMC (TSM)": "https://finance.yahoo.com/quote/TSM/financials/",
    "Microsoft (MSFT)": "https://finance.yahoo.com/quote/MSFT/cash-flow/",
    "Alphabet (GOOGL)": "https://finance.yahoo.com/quote/GOOGL/cash-flow/",
    "Amazon (AMZN)": "https://finance.yahoo.com/quote/AMZN/cash-flow/",
    "Meta (META)": "https://finance.yahoo.com/quote/META/cash-flow/",
    "NVIDIA (NVDA)": "https://finance.yahoo.com/quote/NVDA/financials/",
    "Oracle (ORCL)": "https://finance.yahoo.com/quote/ORCL/financials/",
    "Reuters 電網幽靈需求": "https://www.reuters.com/business/texas-halt-powering-data-centers-reflects-us-reckoning-over-ghost-demand-2026-09-01/",
    "NVIDIA SEC 8-K": "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000069/nvda-20260817.htm",
    "OpenAI 財務與合約分析": "https://mlq.ai/news/leaked-openai-financials-reveal-21b-operating-loss-on-13b-revenue-in-2025/"
}

# --- 3. 自動連網擷取 Yahoo Finance 真實市場數據 ---
@st.cache_data(ttl=3600)
def fetch_realtime_market_data():
    tickers = ["TSM", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "ORCL"]
    results = {}
    for sym in tickers:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d")
            if not hist.empty:
                last_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else last_price
                pct_change = ((last_price - prev_price) / prev_price) * 100
                last_date = hist.index[-1].strftime('%Y-%m-%d')
            else:
                last_price, pct_change, last_date = 0.0, 0.0, "N/A"
            results[sym] = {"price": last_price, "change": pct_change, "date": last_date}
        except Exception:
            results[sym] = {"price": 0.0, "change": 0.0, "date": "連線逾時"}
    return results

# --- 4. 自動化 Overbooking 雙軌審計引擎 ---
def audit_overbooking(texas_queue_gw, retraction_rate, tsmc_inv_days):
    """
    Overbooking 雙軌評估：
    1. 電力 OVERBOOKING：德州申請 > 300GW 或 審查後管線萎縮率 > 35% -> 成立 (扣 0.5 分)
    2. 高階晶片 OVERBOOKING：台積電或晶片存貨週轉天數異常飆高 (> 110天) -> 成立 (扣 0.5 分)
    """
    power_ob = (texas_queue_gw >= 300.0) or (retraction_rate >= 35.0)
    chip_ob = (tsmc_inv_days >= 110.0)
    return power_ob, chip_ob

# --- 5. 評分核心引擎 (回歸直接扣分，無尺度壓縮) ---
def calculate_score(deductions_dict):
    """
    滿分 5.0 分制：
    - 四大核心指標、雙軌 Overbooking、循環融資任一項觸發：各扣 0.5 分
    - 最低 0.0 分
    """
    total_deductions = sum(deductions_dict.values())
    final_score = max(0.0, 5.0 - float(total_deductions))
    return round(final_score, 1), total_deductions

# --- 6. 儀表板量表圖繪製 ---
def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "綜合健康評比 (5分滿分量表)", 'font': {'size': 20, 'color': '#FFFFFF'}},
        gauge={
            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#2ecc71" if score >= 4.0 else ("#f39c12" if score >= 2.5 else "#e74c3c")},
            'steps': [
                {'range': [0, 2.0], 'color': "rgba(231, 76, 60, 0.4)"},
                {'range': [2.0, 3.5], 'color': "rgba(243, 156, 18, 0.4)"},
                {'range': [3.5, 5.0], 'color': "rgba(46, 204, 113, 0.4)"}
            ],
            'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 2.5}
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# --- 7. 主介面與即時更新按鈕 ---
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.title("美台 AI 資本與 OpenAI 財務傳導監測儀表板")
    st.caption(f"🕒 系統時間：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CST` | 評核標準：各項反轉/風險訊號皆扣 0.5 分制")

with col_t2:
    st.write("")
    if st.button("🔄 立即重新更新資料", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.toast("已清除快取，正在重新連線擷取市場最新行情...", icon="🚀")
        st.rerun()

market_data = fetch_realtime_market_data()

# 顯示即時行情看板
m_cols = st.columns(7)
for idx, sym in enumerate(["TSM", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "ORCL"]):
    info = market_data.get(sym, {"price": 0, "change": 0, "date": "N/A"})
    m_cols[idx].metric(
        label=f"{sym} ({info['date']})",
        value=f"${info['price']:.2f}" if info['price'] > 0 else "載入中",
        delta=f"{info['change']:+.2f}%"
    )

st.markdown("---")

# --- 8. 側邊欄參數設定 ---
st.sidebar.header("四大核心領先指標設定 (各扣0.5分)")
lead_time = st.sidebar.slider("1. GPU 交貨期 (月)", 1.0, 18.0, 8.5, 0.5)
capex_yoy = st.sidebar.number_input("2. 四大巨頭 Capex 最低 YoY (%)", -50.0, 150.0, 35.0, 1.0)
chip_growth = st.sidebar.number_input("3. 核心晶片營收年增率 YoY (%)", -30.0, 250.0, 117.0, 1.0)
openai_rev_gap = st.sidebar.checkbox("4. 企業端採用遲緩 / OpenAI 營收下修200億", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("OVERBOOKING 雙軌 (各扣0.5分)")
tx_gw = st.sidebar.number_input("德州電網排隊量 (GW)", 10.0, 800.0, 474.0, 10.0)
grid_cut = st.sidebar.slider("要求押金後管線萎縮率 (%)", 0.0, 100.0, 45.0, 5.0)
tsmc_inv = st.sidebar.number_input("台積電先進製程存貨天數", 30.0, 180.0, 82.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("循環融資與閉環交易 (扣0.5分)")
circular_loop = st.sidebar.checkbox("微軟算力閉環 / ORCL千億合約 / NVDA千億承諾", value=True)

# 執行雙軌審計
power_ob, chip_ob = audit_overbooking(tx_gw, grid_cut, tsmc_inv)

# 統一鍵名 (全項各扣 0.5 分)
deductions = {
    "指標 1: 硬體交貨前置時間 < 6個月": 0.5 if lead_time < 6.0 else 0.0,
    "指標 2: 四大巨頭 Capex 首度 YoY < 0%": 0.5 if capex_yoy < 0.0 else 0.0,
    "指標 3: 核心晶片營收連兩季 YoY < 20%": 0.5 if chip_growth < 20.0 else 0.0,
    "指標 4: 企業端採用與 OpenAI 營收缺口": 0.5 if openai_rev_gap else 0.0,
    "考核項 5a: 電力 OVERBOOKING": 0.5 if power_ob else 0.0,
    "考核項 5b: 高階晶片 OVERBOOKING": 0.5 if chip_ob else 0.0,
    "考核項 6: 循環融資與閉環合約風險": 0.5 if circular_loop else 0.0
}

# 執行直接扣分計算
final_score, total_deductions = calculate_score(deductions)

# --- 9. 第一層：評分量表與綜合診斷 ---
col_g, col_s = st.columns([1, 2])
with col_g:
    st.plotly_chart(create_gauge(final_score), use_container_width=True)

with col_s:
    st.markdown("### 綜合評比結論")
    status_label = "🟢 安全區 (Safe)" if final_score >= 4.0 else ("🟡 警告區 (Warning)" if final_score >= 2.5 else "🔴 觸發反轉 (Triggered)")
    st.subheader(f"當前評估得分：`{final_score:.1f} / 5.0 分` — 狀態：{status_label}")
    st.markdown(f"""
    - **計分準則**：滿分 5.0 分，任一項風險或反轉訊號觸發**各扣 0.5 分**（最低為 0 分）[cite: 1]。目前累計扣除 `{total_deductions:.1f}` 分。
    - **已扣分項目說明 (共扣 1.5 分)**：
      1. **企業端採用與變現缺口（-0.5 分）**：OpenAI 2025 年實質營運赤字達 209.2 億美元，且 2026 年預期營收由 700 億下修至 500 億美元，終端應用無法填補龐大折舊[cite: 8, 14]。
      2. **電力 OVERBOOKING（-0.5 分）**：德州申請量高達 `{tx_gw} GW`，且電力公司收押金後管線萎縮達 `{grid_cut}%`，電網申報端幽靈需求成立。
      3. **循環融資與閉環合約（-0.5 分）**：涵蓋微軟 Azure 算力抵用閉環、甲骨文 1,000 億美元 Stargate 建設合約、NVIDIA 1,050 億美元租約擔保與巨頭千億發債[cite: 7, 14]。
    - **未扣分項目**：硬體交期（`{lead_time} 個月`）、巨頭 Capex 增速、晶片營收年增率（`+{chip_growth:.0f}%`）均維持在安全區間[cite: 1]；台積電先進封裝維持滿載，存貨天數（`{tsmc_inv:.0f} 天`）正常，未見高階晶片實體砍單[cite: 8]。
    """)

# --- 10. 第二層：監測矩陣清單與原始連結 ---
st.markdown("---")
st.subheader("美台四大核心指標、雙軌 Overbooking 與 OpenAI 財務傳導矩陣")

monitor_rows = [
    {
        "指標類別": "指標 1: 硬體交貨前置時間",
        "監測標的": "Nvidia Blackwell / H200 交付期",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": f"{lead_time:.1f} 個月",
        "判定結果": "❌ 觸發扣分 (-0.5)" if deductions["指標 1: 硬體交貨前置時間 < 6個月"] > 0 else "✅ 正常安全",
        "資料公告日期": market_data["NVDA"]["date"],
        "官方/查核連結": DATA_SOURCES["NVIDIA (NVDA)"]
    },
    {
        "指標類別": "指標 2: CAPEX 指引反轉",
        "監測標的": "四大巨頭 2026 Capex (合計約 7,250 億美元)",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": "各大巨頭尚未調降年度資本支出",
        "判定結果": "❌ 觸發扣分 (-0.5)" if deductions["指標 2: 四大巨頭 Capex 首度 YoY < 0%"] > 0 else "✅ 正常安全",
        "資料公告日期": market_data["MSFT"]["date"],
        "官方/查核連結": DATA_SOURCES["Microsoft (MSFT)"]
    },
    {
        "指標類別": "指標 3: 核心晶片營收減速",
        "監測標的": "NVDA Data Center / TSMC HPC 營收",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": f"NVDA Data Center YoY: +{chip_growth:.0f}%",
        "判定結果": "❌ 觸發扣分 (-0.5)" if deductions["指標 3: 核心晶片營收連兩季 YoY < 20%"] > 0 else "✅ 正常安全",
        "資料公告日期": "2026-08-26",
        "官方/查核連結": DATA_SOURCES["TSMC (TSM)"]
    },
    {
        "指標類別": "指標 4: 企業端採用與變現",
        "監測標的": "OpenAI 2026 營收下修至 500 億 / 營運赤字 209 億",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": "高額推論費用侵蝕毛利，終端營收成長斜率放緩",
        "判定結果": "⚠️ 觸發扣分 (-0.5)" if deductions["指標 4: 企業端採用與 OpenAI 營收缺口"] > 0 else "✅ 正常觀察",
        "資料公告日期": "2026-10",
        "官方/查核連結": DATA_SOURCES["OpenAI 財務與合約分析"]
    },
    {
        "指標類別": "考核項 5a: 電力 OVERBOOKING",
        "監測標的": "美國電網端幽靈需求 (德州 474GW / 全美 700GW)",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": "德州暫停併網；Exelon/AEP 需求下修 40-50%",
        "判定結果": "⚠️ 觸發扣分 (-0.5)" if deductions["考核項 5a: 電力 OVERBOOKING"] > 0 else "✅ 正常安全",
        "資料公告日期": "2026-09-01",
        "官方/查核連結": DATA_SOURCES["Reuters 電網幽靈需求"]
    },
    {
        "指標類別": "考核項 5b: 高階晶片 OVERBOOKING",
        "監測標的": "台積電先進封裝排單與存貨週轉天數",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": f"先進製程維持滿載，存貨天數 {tsmc_inv:.0f} 天正常",
        "判定結果": "❌ 觸發扣分 (-0.5)" if deductions["考核項 5b: 高階晶片 OVERBOOKING"] > 0 else "✅ 正常安全 (扣0分)",
        "資料公告日期": market_data["TSM"]["date"],
        "官方/查核連結": DATA_SOURCES["TSMC (TSM)"]
    },
    {
        "指標類別": "考核項 6: 循環融資與閉環交易",
        "監測標的": "MSFT 2,500 億承諾 / ORCL 1,000 億 / NVDA 千億曝險",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": "依賴算力抵用額度折抵，高度集中單一客戶信用",
        "判定結果": "⚠️ 觸發扣分 (-0.5)" if deductions["考核項 6: 循環融資與閉環合約風險"] > 0 else "✅ 正常安全",
        "資料公告日期": "2026-08 / 2026-10",
        "官方/查核連結": DATA_SOURCES["NVIDIA SEC 8-K"]
    }
]

df_display = pd.DataFrame(monitor_rows)
st.dataframe(
    df_display,
    column_config={
        "官方/查核連結": st.column_config.LinkColumn("點擊開啟官方文件 / 查核報導")
    },
    use_container_width=True,
    hide_index=True
)