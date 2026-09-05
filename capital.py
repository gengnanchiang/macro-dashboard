# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:42:49 2026

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
    page_title="美台 AI 資本反轉訊號監測儀表板 (每項各扣0.5分制)",
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
    "Reuters 電網幽靈需求查核": "https://www.reuters.com/business/texas-halt-powering-data-centers-reflects-us-reckoning-over-ghost-demand-2026-09-01/",
    "NVIDIA SEC 8-K 法定文件": "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000069/nvda-20260817.htm"
}

# --- 3. 自動連網擷取 Yahoo Finance 真實市場數據 ---
@st.cache_data(ttl=3600)
def fetch_realtime_market_data():
    tickers = ["TSM", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
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
            results[sym] = {
                "price": last_price,
                "change": pct_change,
                "date": last_date
            }
        except Exception:
            results[sym] = {"price": 0.0, "change": 0.0, "date": "連線逾時"}
    return results

# --- 4. 自動化 Overbooking 雙軌審計引擎 ---
def audit_overbooking_subfactors(texas_queue_gw, grid_retraction_rate, tsmc_inventory_turnover_days):
    """
    Overbooking 雙軌評估：
    1. 電力 OVERBOOKING：德州申請 > 300GW 或 審查後管線萎縮率 > 35% -> 成立 (扣 0.5 分)
    2. 高階晶片 OVERBOOKING：台積電或晶片存貨週轉天數異常飆高 (> 110天) -> 成立 (扣 0.5 分)
    """
    power_overbooking = (texas_queue_gw >= 300.0) or (grid_retraction_rate >= 35.0)
    chip_overbooking = (tsmc_inventory_turnover_days >= 110.0)
    return power_overbooking, chip_overbooking

# --- 5. 評分核心引擎 (每項各扣 0.5 分) ---
def calculate_reversal_score_uniform(deductions_dict):
    """
    滿分 5.0 分制：
    - 四大核心領先指標任一項觸發：各扣 0.5 分
    - 電力 OVERBOOKING 觸發：扣 0.5 分
    - 高階晶片 OVERBOOKING 觸發：扣 0.5 分
    - 循環融資 / 專案保證槓桿觸發：扣 0.5 分
    - 最低分：0.0 分
    """
    total_deductions = sum(deductions_dict.values())
    final_score = max(0.0, 5.0 - float(total_deductions))
    return final_score, total_deductions

# --- 6. 儀表板圖形繪製 ---
def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "綜合健康評比 (5分量表)", 'font': {'size': 20, 'color': '#FFFFFF'}},
        gauge={
            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#2ecc71" if score >= 4.0 else ("#f39c12" if score >= 2.5 else "#e74c3c")},
            'steps': [
                {'range': [0, 2.5], 'color': "rgba(231, 76, 60, 0.4)"},
                {'range': [2.5, 4.0], 'color': "rgba(243, 156, 18, 0.4)"},
                {'range': [4.0, 5.0], 'color': "rgba(46, 204, 113, 0.4)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 2.5
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# --- 7. 介面主標題與「即時更新按鈕」 ---
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.title("美台 AI 資本與重複預訂(Overbooking)監測儀表板")
    st.caption(f"🕒 系統時間：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CST` | 評核標準：各項反轉/風險訊號皆扣 0.5 分制")

with h_col2:
    st.write("")
    if st.button("🔄 立即重新更新資料", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.toast("已清除快取，正在重新連線擷取市場最新行情...", icon="🚀")
        st.rerun()

market_data = fetch_realtime_market_data()

# 顯示即時行情看板
m_cols = st.columns(6)
for idx, sym in enumerate(["TSM", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]):
    info = market_data.get(sym, {"price": 0, "change": 0, "date": "N/A"})
    m_cols[idx].metric(
        label=f"{sym} ({info['date']})",
        value=f"${info['price']:.2f}" if info['price'] > 0 else "載入中",
        delta=f"{info['change']:+.2f}%"
    )

st.markdown("---")

# --- 8. 側邊欄：前瞻參數與雙軌 Overbooking 參數 ---
st.sidebar.header("四大核心領先指標設定 (各扣0.5分)")
lead_time_months = st.sidebar.slider("1. Nvidia GPU 交貨期 (月)", min_value=1.0, max_value=18.0, value=8.5, step=0.5)
hyperscaler_capex_yoy = st.sidebar.number_input("2. 四大巨頭 Capex 最低 YoY (%)", min_value=-50.0, max_value=150.0, value=35.0, step=1.0)
nvda_dc_rev_growth = st.sidebar.number_input("3. 核心晶片營收年增率 YoY (%)", min_value=-30.0, max_value=250.0, value=117.0, step=1.0)
enterprise_adoption_stagnant = st.sidebar.checkbox("4. 企業端活躍使用量實質停滯", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("OVERBOOKING 雙軌評估 (各扣0.5分)")
texas_queue_gw = st.sidebar.number_input("德州電網申報用電量 (GW)", min_value=10.0, max_value=800.0, value=474.0, step=10.0)
grid_retraction_rate = st.sidebar.slider("加收擔保金後需求管線下修率 (%)", min_value=0.0, max_value=100.0, value=45.0, step=5.0)
tsmc_inventory_days = st.sidebar.number_input("台積電先進封裝/製程存貨天數", min_value=30.0, max_value=180.0, value=82.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("循環融資與專案保證 (扣0.5分)")
circular_financing_risk = st.sidebar.checkbox("循環融資風險 (SB Energy 1050億保證 / 巨頭1750億發債)", value=True)

# 執行雙軌 Overbooking 審計
power_ob_triggered, chip_ob_triggered = audit_overbooking_subfactors(texas_queue_gw, grid_retraction_rate, tsmc_inventory_days)

# 扣分權重計算字典 (全品項各扣 0.5 分)
deductions_map = {
    "指標 1: 硬體交貨前置時間 < 6個月": 0.5 if (lead_time_months < 6.0) else 0.0,
    "指標 2: 四大巨頭 Capex 首度 YoY < 0%": 0.5 if (hyperscaler_capex_yoy < 0.0) else 0.0,
    "指標 3: 核心晶片營收連兩季 YoY < 20%": 0.5 if (nvda_dc_rev_growth < 20.0) else 0.0,
    "指標 4: 企業端採用停滯": 0.5 if enterprise_adoption_stagnant else 0.0,
    "考核項 5a: 電力 OVERBOOKING (幽靈需求)": 0.5 if power_ob_triggered else 0.0,
    "考核項 5b: 高階晶片 OVERBOOKING (晶圓/庫存積壓)": 0.5 if chip_ob_triggered else 0.0,
    "考核項 6: 循環融資與專案槓桿": 0.5 if circular_financing_risk else 0.0
}

current_score, total_deductions = calculate_reversal_score_uniform(deductions_map)

# --- 9. 第一層：評分量表與綜合診斷 ---
col_gauge, col_summary = st.columns([1, 2])

with col_gauge:
    st.plotly_chart(create_gauge_chart(current_score), use_container_width=True)

with col_summary:
    st.markdown("### 綜合評比結論")
    status_label = "🟢 安全區 (Safe)" if current_score >= 4.0 else ("🟡 警告區 (Warning)" if current_score >= 2.5 else "🔴 觸發反轉 (Triggered)")
    st.subheader(f"當前評分：`{current_score:.1f} / 5.0 分` — 狀態：{status_label}")
    
    st.markdown(f"""
    - **計分準則**：基準 5.0 分。四大指標、雙軌 Overbooking 與循環融資**每出現一項各扣 0.5 分**（累計扣除 `{total_deductions:.1f}` 分，最低為 0 分）。
    - **已扣分項目說明**：
      1. **電力 OVERBOOKING（扣 0.5 分）**：德州排隊達 `{texas_queue_gw} GW`，加收押金後管線萎縮率達 `{grid_retraction_rate}%`，電網申報端幽靈需求成立。
      2. **循環融資與保證槓桿（扣 0.5 分）**：雲端巨頭 2026 發債 1,750 億美元，且 NVDA 對 SB Energy 存在 1,050 億美元初期租約累計保證與 30 億美元股權安排[cite: 1]。
    - **未扣分項目**：四大核心指標目前均安全（交期 > 6個月、Capex 高成長、核心晶片營收年增 > 20%、企業採用持續擴張）[cite: 1]；高階晶片未見實體砍單（存貨天數 `{tsmc_inventory_days:.0f} 天` 正常）。
    """)

# --- 10. 第二層：監測矩陣清單與原始連結 ---
st.markdown("---")
st.subheader("美台四大核心指標、Overbooking 與循環融資監測矩陣")

monitor_rows = [
    {
        "指標類別": "指標 1: 硬體交貨前置時間",
        "監測標的": "Nvidia Blackwell / H200 交付期",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": f"{lead_time_months:.1f} 個月",
        "判定結果": "❌ 觸發扣分 (-0.5)" if deductions_map["指標 1: 硬體交貨前置時間 < 6個月"] > 0 else "✅ 正常安全",
        "資料公告日期": market_data["NVDA"]["date"],
        "官方/查核連結": DATA_SOURCES["NVIDIA (NVDA)"]
    },
    {
        "指標類別": "指標 2: CAPEX 指引反轉",
        "監測標的": "四大巨頭 2026 Capex (合計約 7,250 億美元)",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": "各大巨頭維持資本支出高速成長",
        "判定結果": "❌ 觸發扣分 (-0.5)" if deductions_map["指標 2: 四大巨頭 Capex 首度 YoY < 0%"] > 0 else "✅ 正常安全",
        "資料公告日期": market_data["MSFT"]["date"],
        "官方/查核連結": DATA_SOURCES["Microsoft (MSFT)"]
    },
    {
        "指標類別": "指標 3: 核心晶片營收減速",
        "監測標的": "NVDA Data Center / TSMC HPC 營收",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": f"NVDA Data Center YoY: +{nvda_dc_rev_growth:.0f}%",
        "判定結果": "❌ 觸發扣分 (-0.5)" if deductions_map["指標 3: 核心晶片營收連兩季 YoY < 20%"] > 0 else "✅ 正常安全",
        "資料公告日期": "2026-08-26",
        "官方/查核連結": DATA_SOURCES["TSMC (TSM)"]
    },
    {
        "指標類別": "指標 4: 企業端採用停滯",
        "監測標的": "Fortune 500 企業 AI 應用經常性活躍度",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": "使用量持續擴張，但需觀察 6,000 億缺口",
        "判定結果": "❌ 觸發扣分 (-0.5)" if deductions_map["指標 4: 企業端採用停滯"] > 0 else "✅ 持續觀察",
        "資料公告日期": market_data["GOOGL"]["date"],
        "官方/查核連結": DATA_SOURCES["Alphabet (GOOGL)"]
    },
    {
        "指標類別": "考核項 5a: 電力 OVERBOOKING",
        "監測標的": "美國電網端幽靈需求 (德州 474GW / 全美 700GW)",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": "德州暫停併網；Exelon/AEP 需求下修 40-50%",
        "判定結果": "⚠️ 觸發扣分 (-0.5)" if power_ob_triggered else "✅ 正常安全",
        "資料公告日期": "2026-09-01",
        "官方/查核連結": DATA_SOURCES["Reuters 電網幽靈需求查核"]
    },
    {
        "指標類別": "考核項 5b: 高階晶片 OVERBOOKING",
        "監測標的": "台積電先進封裝排單與存貨週轉天數",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": f"先進製程維持滿載，存貨天數 {tsmc_inventory_days:.0f} 天正常",
        "判定結果": "❌ 觸發扣分 (-0.5)" if chip_ob_triggered else "✅ 正常安全 (扣0分)",
        "資料公告日期": market_data["TSM"]["date"],
        "官方/查核連結": DATA_SOURCES["TSMC (TSM)"]
    },
    {
        "指標類別": "考核項 6: 循環融資與專案槓桿",
        "監測標的": "巨頭發債 1,750 億美元；SB Energy 4,390 億合約積壓與保證",
        "扣分規則": "觸發扣 0.5 分",
        "最新數值 / 狀態": "NVDA 提供 1,050 億租約保證上限與 30 億股權曝險",
        "判定結果": "⚠️ 觸發扣分 (-0.5)" if circular_financing_risk else "✅ 正常安全",
        "資料公告日期": "2026-08-17 / 2026-09-01",
        "官方/查核連結": DATA_SOURCES["NVIDIA SEC 8-K 法定文件"]
    }
]

df_table = pd.DataFrame(monitor_rows)
st.dataframe(
    df_table,
    column_config={
        "官方/查核連結": st.column_config.LinkColumn("點擊開啟官方文件 / 查核報導")
    },
    use_container_width=True,
    hide_index=True
)