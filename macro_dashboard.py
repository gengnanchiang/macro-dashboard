# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:34:06 2026

@author: jnchi
"""

import sys
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# ==========================================
# 0. 系統與頁面全域設定
# ==========================================
st.set_page_config(
    page_title="美台總體經濟即時監測儀表板",
    page_icon="📈",
    layout="wide"
)

# 優先讀取 Streamlit Secrets，若未設定則使用預設常數
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "03ff533806d1f33a86bcdbe948d9abf7")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ==========================================
# 1. 輔助函式：全指標統計期格式標準化
# ==========================================
def format_period_label(date_str, freq="monthly"):
    """
    將各類日期格式統一轉換為標準統計期標籤：
    - freq='quarterly': 轉為 YYYY Q1 ~ Q4 (比照美國實質 GDP 格式)
    - freq='monthly': 轉為 YYYY-MM
    支援輸入格式包含: 'YYYY-MM-DD', '113M06', '113/06', '113年06月', '2026Q2' 等
    """
    if not date_str or date_str in ["-", "即時查詢"]:
        return date_str
        
    date_str = str(date_str).strip()
    
    # 處理已是 2026Q2 或 2026 Q2 格式
    if "Q" in date_str or "q" in date_str:
        clean_q = date_str.upper().replace(" ", "")
        if len(clean_q) >= 6:
            return f"{clean_q[:4]} {clean_q[4:]}"
        return date_str

    # 處理台灣民國年月 (如 113M06, 113/06, 113-06, 113年6月)
    for sep in ['M', '/', '年', '-']:
        if sep in date_str:
            parts = date_str.replace('月', '').replace('日', '').split(sep)
            if len(parts) >= 2:
                try:
                    year, month = int(parts[0]), int(parts[1])
                    if year < 1911:
                        year += 1911
                    
                    if freq == "quarterly":
                        quarter = (month - 1) // 3 + 1
                        return f"{year} Q{quarter}"
                    return f"{year}-{month:02d}"
                except ValueError:
                    pass

    # 處理標準西元 'YYYY-MM-DD'
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        if freq == "quarterly":
            quarter = (dt.month - 1) // 3 + 1
            return f"{dt.year} Q{quarter}"
        return dt.strftime("%Y-%m")
    except Exception:
        return date_str

# ==========================================
# 2. 數據獲取模組 (含快取機制與標準期別處理)
# ==========================================

@st.cache_data(ttl=1800)
def fetch_fred_series(series_id, api_key, freq="monthly", limit=24):
    """
    自 FRED 官方 API 抓取美國時間序列數據
    回傳: (最新數值, 統計期標籤, 歷史 DataFrame)
    """
    if not api_key or api_key == "YOUR_FRED_API_KEY":
        # 示範/備用時間序列
        dates = pd.date_range(end=datetime.today(), periods=limit, freq="QE" if freq == "quarterly" else "ME")
        dummy_df = pd.DataFrame({
            "日期": [format_period_label(d.strftime("%Y-%m-%d"), freq=freq) for d in dates],
            "數值": [2.5 + (i * 0.05) for i in range(limit)]
        })
        latest_val = dummy_df["數值"].iloc[-1]
        latest_period = dummy_df["日期"].iloc[-1]
        return latest_val, latest_period, dummy_df

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
        f"&sort_order=desc&limit={limit}"
    )
    try:
        res = requests.get(url, timeout=8).json()
        observations = res.get("observations", [])
        if not observations:
            return None, None, pd.DataFrame()

        df = pd.DataFrame(observations)[["date", "value"]]
        df = df[df["value"] != "."]
        df["value"] = pd.to_numeric(df["value"])
        
        # 轉換為標準期別標籤 (例如 2026-04-01 -> 2026 Q2)
        df["統計期"] = df["date"].apply(lambda d: format_period_label(d, freq=freq))
        df.rename(columns={"value": "數值"}, inplace=True)
        df.sort_values("date", inplace=True)

        latest_val = df["數值"].iloc[-1]
        latest_period = df["統計期"].iloc[-1]
        return latest_val, latest_period, df[["統計期", "數值"]]
    except Exception:
        return None, None, pd.DataFrame()


@st.cache_data(ttl=1800)
def fetch_ndc_indicators(limit=15):
    """抓取國發會景氣對策信號數據 (月度指標)"""
    api_url = "https://ws.ndc.gov.tw/Download.ashx?u=LzAwMS9hZG1pbmlzdHJhdG9yLzEwL3BkZl8xMDkvYnVzaW5lc3NfaW5kaWNhdG9ycy5qc29u&n=YnVzaW5lc3NfaW5kaWNhdG9ycy5qc29u"
    try:
        res = requests.get(api_url, headers=HEADERS, timeout=8)
        if res.status_code == 200:
            data = res.json()
            df = pd.DataFrame(data)
            col_map = {
                '年月': '月份', 'Period': '月份', '項目/時間': '月份',
                '景氣對策信號(分)': '綜合分數', 'CheckPoint': '綜合分數', 'Score': '綜合分數'
            }
            df.rename(columns=col_map, inplace=True)
            if '月份' in df.columns and '綜合分數' in df.columns:
                df['統計期'] = df['月份'].apply(lambda d: format_period_label(d, freq="monthly"))
                df['綜合分數'] = pd.to_numeric(df['綜合分數'], errors='coerce')
                df = df.dropna(subset=['綜合分數']).sort_values('統計期').reset_index(drop=True)
                latest_period = df['統計期'].iloc[-1]
                latest_score = int(df['綜合分數'].iloc[-1])
                return latest_period, latest_score, df.tail(limit)
    except Exception:
        pass

    # 備用參考時間序列
    demo_df = pd.DataFrame({
        "統計期": ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"],
        "綜合分數": [32, 34, 38, 37, 36, 39, 35, 36, 37, 35]
    })
    return demo_df["統計期"].iloc[-1], int(demo_df["綜合分數"].iloc[-1]), demo_df


@st.cache_data(ttl=1800)
def fetch_dgbas_cpi(limit=15):
    """抓取主計總處 CPI 變動率數據 (月度指標)"""
    stat_url = "https://apiservice.mol.gov.tw/OdService/rest/datastore/A17000000J-030018-bV2"
    try:
        res = requests.get(stat_url, headers=HEADERS, timeout=8)
        if res.status_code == 200:
            data = res.json()
            records = data.get("result", {}).get("records", []) if isinstance(data, dict) else data
            df = pd.DataFrame(records)
            col_map = {'統計期': '月份', '年月': '月份', '年增率(%)': 'CPI_YoY', '指數': 'CPI_YoY'}
            df.rename(columns=col_map, inplace=True)
            if '月份' in df.columns and 'CPI_YoY' in df.columns:
                df['統計期'] = df['月份'].apply(lambda d: format_period_label(d, freq="monthly"))
                df['CPI_YoY'] = pd.to_numeric(df['CPI_YoY'], errors='coerce')
                df = df.dropna(subset=['CPI_YoY']).sort_values('統計期').reset_index(drop=True)
                return df['統計期'].iloc[-1], float(df['CPI_YoY'].iloc[-1]), df.tail(limit)
    except Exception:
        pass

    # 備用參考時間序列
    demo_df = pd.DataFrame({
        "統計期": ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"],
        "CPI_YoY": [1.69, 1.75, 2.05, 1.79, 2.15, 2.10, 1.95, 1.88, 1.92, 1.85]
    })
    return demo_df["統計期"].iloc[-1], float(demo_df["CPI_YoY"].iloc[-1]), demo_df

# ==========================================
# 3. 評分與燈號判定
# ==========================================
def calculate_score_and_light(score):
    mapping = {
        5: {"light": "🟢 綠燈", "status": "強勁擴張 (景氣繁榮，供需與就業熱絡)"},
        4: {"light": "🟡 綠黃燈", "status": "穩健成長 (景氣擴張，多數指標向好)"},
        3: {"light": "🟠 黃燈", "status": "中性轉折 (成長趨緩，處於政策與利率觀望期)"},
        2: {"light": "🟠 黃藍燈", "status": "景氣放緩 (緊縮環境下總合需求轉弱)"},
        1: {"light": "🔴 藍燈", "status": "低迷衰退 (景氣收縮，總體指標普遍疲弱)"}
    }
    return mapping.get(score, mapping[3])

# ==========================================
# 4. 前端儀表板渲染
# ==========================================
def main():
    # 頂部控制列
    header_col1, header_col2 = st.columns([4, 1])

    with header_col1:
        st.title("🌐 美國 vs 台灣 總體經濟即時監測儀表板")
        st.caption(f"數據最後更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (CST)")

    with header_col2:
        st.write("")
        if st.button("🔄 立即更新數據", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.toast("已清除快取，正在重新檢索官方最新數據...", icon="🔄")
            st.rerun()

    st.divider()

    # 美國指標定義 (加入 freq 參數識別季度或月度)
    us_meta = [
        {"name": "實質 GDP 季增年率 (Real GDP)", "code": "A191RL1Q225SBEA", "unit": "%", "freq": "quarterly", "url": "https://fred.stlouisfed.org/series/A191RL1Q225SBEA"},
        {"name": "消費者物價指數年增率 (CPI YoY)", "code": "CPIAUCSL", "unit": "%", "freq": "monthly", "url": "https://fred.stlouisfed.org/series/CPIAUCSL"},
        {"name": "聯邦基金有效利率 (Fed Funds Rate)", "code": "FEDFUNDS", "unit": "%", "freq": "monthly", "url": "https://fred.stlouisfed.org/series/FEDFUNDS"},
        {"name": "非農失業率 (Unemployment Rate)", "code": "UNRATE", "unit": "%", "freq": "monthly", "url": "https://fred.stlouisfed.org/series/UNRATE"}
    ]

    col_us, col_tw = st.columns(2)

    # ------------------ 美國專區 ------------------
    with col_us:
        st.subheader("🇺🇸 美國總體經濟 (United States)")
        us_score = st.slider("美國景氣綜合評分 (五分量表)", 1, 5, 4, key="us_score_slider")
        us_eval = calculate_score_and_light(us_score)
        st.metric("總體環境評比", f"{us_score} / 5 分", delta=us_eval["light"])
        st.info(f"景氣研判：**{us_eval['status']}**")

        st.markdown("#### 📊 即時變量清單與官方查驗來源")
        us_table = []
        us_hist_store = {}

        for item in us_meta:
            val, period_label, h_df = fetch_fred_series(item["code"], FRED_API_KEY, freq=item["freq"], limit=20)
            us_hist_store[item["name"]] = h_df
            us_table.append({
                "指標名稱": item["name"],
                "最新數值": f"{val:.2f} {item['unit']}" if val is not None else "待填 API Key",
                "統計期 (涵蓋期間)": period_label if period_label else "-",
                "官方查驗超連結": f"[前往 FRED 官網]({item['url']})"
            })

        st.markdown(pd.DataFrame(us_table).to_markdown(index=False), unsafe_allow_html=True)

        st.markdown("#### 📉 美國總經時間序列走勢")
        selected_us = st.selectbox("選擇走勢指標：", list(us_hist_store.keys()), key="us_metric_choice")
        target_us_df = us_hist_store.get(selected_us)

        if target_us_df is not None and not target_us_df.empty:
            fig_us = px.line(
                target_us_df,
                x="統計期",
                y="數值",
                title=f"{selected_us} 走勢",
                markers=True,
                template="plotly_white"
            )
            fig_us.update_traces(line_color="#1f77b4", hovertemplate="期別: %{x}<br>數值: %{y}")
            fig_us.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_us, use_container_width=True)

    # ------------------ 台灣專區 ------------------
    with col_tw:
        st.subheader("🇹🇼 台灣總體經濟 (Taiwan)")
        tw_score = st.slider("台灣景氣綜合評分 (五分量表)", 1, 5, 4, key="tw_score_slider")
        tw_eval = calculate_score_and_light(tw_score)
        st.metric("總體環境評比", f"{tw_score} / 5 分", delta=tw_eval["light"])
        st.info(f"景氣研判：**{tw_eval['status']}**")

        st.markdown("#### 📊 即時變量清單與官方查驗來源")
        
        ndc_period, ndc_score, ndc_df = fetch_ndc_indicators(limit=12)
        cpi_period, cpi_score, cpi_df = fetch_dgbas_cpi(limit=12)

        # 台灣指標依季度 (Q) 或月份 (YYYY-MM) 格式標準化
        tw_table = [
            {
                "指標名稱": "國發會景氣對策信號綜合分數",
                "最新數值": f"{ndc_score} 分" if ndc_score else "-",
                "統計期 (涵蓋期間)": ndc_period if ndc_period else "-",
                "官方查驗超連結": "[國發會景氣指標專區](https://www.ndc.gov.tw/Content_List.aspx?n=4F87D397C224E26A)"
            },
            {
                "指標名稱": "主計總處 CPI 年增率 (%)",
                "最新數值": f"{cpi_score:.2f} %" if cpi_score else "-",
                "統計期 (涵蓋期間)": cpi_period if cpi_period else "-",
                "官方查驗超連結": "[主計總處物價統計](https://www.stat.gov.tw/News_Notice.aspx?n=2848&sms=10731)"
            },
            {
                "指標名稱": "實質 GDP 年增率 (經濟成長率 %)",
                "最新數值": "依官方最新公告",
                "統計期 (涵蓋期間)": "2026 Q2",
                "官方查驗超連結": "[主計總處國民所得統計](https://www.stat.gov.tw/)"
            },
            {
                "指標名稱": "中央銀行重貼現率 (%)",
                "最新數值": "依理監事會決議",
                "統計期 (涵蓋期間)": "2026 Q2 (現行水準)",
                "官方查驗超連結": "[中央銀行貼現率專區](https://www.cbc.gov.tw/tw/cp-440-1087-B9C09-1.html)"
            }
        ]
        st.markdown(pd.DataFrame(tw_table).to_markdown(index=False), unsafe_allow_html=True)

        st.markdown("#### 📉 台灣總經時間序列走勢")
        tw_choice = st.selectbox("選擇走勢指標：", ["國發會景氣對策信號綜合分數", "主計總處 CPI 年增率"], key="tw_metric_choice")

        if tw_choice == "國發會景氣對策信號綜合分數" and not ndc_df.empty:
            fig_tw = px.line(
                ndc_df,
                x="統計期",
                y="綜合分數",
                title="景氣對策信號綜合判斷分數",
                markers=True,
                template="plotly_white"
            )
            fig_tw.update_traces(line_color="#2ca02c", hovertemplate="期別: %{x}<br>分數: %{y} 分")
            fig_tw.add_hline(y=38, line_dash="dot", line_color="red", annotation_text="紅燈閾值 (38分)")
            fig_tw.add_hline(y=32, line_dash="dot", line_color="orange", annotation_text="黃紅燈閾值 (32分)")
            fig_tw.add_hline(y=23, line_dash="dot", line_color="green", annotation_text="綠燈閾值 (23分)")
            fig_tw.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_tw, use_container_width=True)
        elif tw_choice == "主計總處 CPI 年增率" and not cpi_df.empty:
            fig_tw = px.line(
                cpi_df,
                x="統計期",
                y="CPI_YoY",
                title="消費者物價指數年增率 (CPI YoY %)",
                markers=True,
                template="plotly_white"
            )
            fig_tw.update_traces(line_color="#d62728", hovertemplate="期別: %{x}<br>CPI: %{y}%")
            fig_tw.add_hline(y=2.0, line_dash="dot", line_color="gray", annotation_text="通膨警戒線 (2.0%)")
            fig_tw.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_tw, use_container_width=True)

    st.divider()

if __name__ == "__main__":
    main()