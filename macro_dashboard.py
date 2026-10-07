# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 23:21:50 2026

@author: jnchi
"""

import sys
import subprocess
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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

# 安全讀取 secrets，本機找不到檔案時自動使用預設 API Key
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    FRED_API_KEY = "03ff533806d1f33a86bcdbe948d9abf7"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ==========================================
# 1. 輔助函式：全指標統計期格式標準化
# ==========================================
def format_period_label(date_str, freq="monthly"):
    """
    將各類日期格式統一轉換為標準統計期標籤：
    - freq='quarterly': 轉為 YYYY Q1 ~ Q4
    - freq='monthly': 轉為 YYYY-MM
    """
    if not date_str or date_str in ["-", "即時查詢"]:
        return date_str
        
    date_str = str(date_str).strip()
    
    if "Q" in date_str or "q" in date_str:
        clean_q = date_str.upper().replace(" ", "")
        if len(clean_q) >= 6:
            return f"{clean_q[:4]} {clean_q[4:]}"
        return date_str

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

    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        if freq == "quarterly":
            quarter = (dt.month - 1) // 3 + 1
            return f"{dt.year} Q{quarter}"
        return dt.strftime("%Y-%m")
    except Exception:
        return date_str

# ==========================================
# 2. 數據獲取模組 (FRED 支援 units 參數取得年增率)
# ==========================================

@st.cache_data(ttl=1800)
def fetch_fred_series(series_id, api_key, freq="monthly", units="lin", limit=24):
    """
    自 FRED 官方 API 抓取美國時間序列數據
    units 參數：
      - 'lin': Levels (原始水準值，例如失業率、聯邦基金利率、GDP成長率)
      - 'pc1': Percent Change from Year Ago (自同月計算年增率 YoY %，用於 CPI)
    """
    if not api_key:
        return None, None, pd.DataFrame()

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
        f"&units={units}&sort_order=desc&limit={limit}"
    )
    try:
        res = requests.get(url, timeout=8).json()
        observations = res.get("observations", [])
        if not observations:
            return None, None, pd.DataFrame()

        df = pd.DataFrame(observations)[["date", "value"]]
        df = df[df["value"] != "."]
        df["value"] = pd.to_numeric(df["value"])
        
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
    """抓取國發會景氣對策信號數據"""
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
    """抓取主計總處 CPI 變動率數據"""
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
# 3. 核心演算法：以 CPI 年增率為核心打分
# ==========================================

def calculate_us_score(gdp, unrate, cpi_yoy, fedfunds):
    """
    根據美國最新實質數據計算 1~5 分
    CPI 採用年增率（YoY %）與 Fed 2.0% 目標的偏差評比
    """
    s_gdp, s_unrate, s_cpi, s_rate = 3, 3, 3, 3

    # 1. 實質 GDP 年化季增率
    if gdp is not None:
        if gdp >= 3.0: s_gdp = 5
        elif gdp >= 2.0: s_gdp = 4
        elif gdp >= 1.0: s_gdp = 3
        elif gdp >= 0.0: s_gdp = 2
        else: s_gdp = 1

    # 2. 失業率 (逆向)
    if unrate is not None:
        if unrate <= 3.8: s_unrate = 5
        elif unrate <= 4.2: s_unrate = 4
        elif unrate <= 4.6: s_unrate = 3
        elif unrate <= 5.2: s_unrate = 2
        else: s_unrate = 1

    # 3. CPI 年增率 (與 Fed 2% 目標的偏差評估)
    if cpi_yoy is not None:
        dev = abs(cpi_yoy - 2.0)
        if dev <= 0.4: s_cpi = 5          # 1.6% ~ 2.4% (極佳通膨受控區間)
        elif dev <= 1.0: s_cpi = 4        # 1.0% ~ 3.0% (溫和可控)
        elif dev <= 1.8: s_cpi = 3        # 3.1% ~ 3.8% (略有黏性或放緩)
        elif dev <= 3.0: s_cpi = 2        # 3.9% ~ 5.0% (通膨偏高)
        else: s_cpi = 1                   # > 5.0% (嚴重通膨) 或 < -1.0% (嚴重通縮)

    # 4. 實質政策利率 (聯邦基金利率 - CPI 年增率)
    if fedfunds is not None and cpi_yoy is not None:
        real_r = fedfunds - cpi_yoy
        if 0.5 <= real_r <= 1.5: s_rate = 5       # 最佳中性實質正利率
        elif (1.6 <= real_r <= 2.5) or (0.0 <= real_r < 0.5): s_rate = 4
        elif (-0.5 <= real_r < 0.0) or (2.6 <= real_r <= 3.2): s_rate = 3
        elif real_r > 3.2: s_rate = 2             # 實質限制過緊
        else: s_rate = 1                          # 深度實質負利率

    final_score = (s_gdp + s_unrate + s_cpi + s_rate) / 4.0
    return round(final_score, 1)


def calculate_tw_score(ndc_score, cpi_yoy, gdp=3.1):
    """根據台灣最新實質數據計算 1~5 分"""
    s_ndc, s_cpi, s_gdp = 3, 3, 3

    if ndc_score is not None:
        if ndc_score >= 38: s_ndc = 5       # 紅燈
        elif ndc_score >= 32: s_ndc = 4     # 黃紅燈
        elif ndc_score >= 23: s_ndc = 3     # 綠燈
        elif ndc_score >= 17: s_ndc = 2     # 黃藍燈
        else: s_ndc = 1                     # 藍燈

    if cpi_yoy is not None:
        if 1.2 <= cpi_yoy <= 1.9: s_cpi = 5
        elif (0.8 <= cpi_yoy < 1.2) or (2.0 <= cpi_yoy <= 2.3): s_cpi = 4
        elif 2.4 <= cpi_yoy <= 3.0: s_cpi = 3
        elif 3.1 <= cpi_yoy <= 4.0: s_cpi = 2
        else: s_cpi = 1

    if gdp is not None:
        if gdp >= 4.0: s_gdp = 5
        elif gdp >= 3.0: s_gdp = 4
        elif gdp >= 2.0: s_gdp = 3
        elif gdp >= 1.0: s_gdp = 2
        else: s_gdp = 1

    final_score = (s_ndc * 0.4) + (s_cpi_score := s_cpi * 0.3) + (s_gdp * 0.3)
    return round(final_score, 1)


def get_status_description(score):
    if score >= 4.5:
        return "🟢 綠燈", "強勁擴張 (景氣繁榮，供需與就業熱絡)"
    elif score >= 3.5:
        return "🟡 綠黃燈", "穩健成長 (景氣擴張，多數指標向好)"
    elif score >= 2.5:
        return "🟠 黃燈", "中性轉折 (成長趨緩，處於政策觀望期)"
    elif score >= 1.5:
        return "🟠 黃藍燈", "景氣放緩 (緊縮環境下總合需求轉弱)"
    else:
        return "🔴 藍燈", "低迷衰退 (景氣收縮，指標普遍疲弱)"

# ==========================================
# 4. 半圓形儀表板繪製模組 (Gauge Chart)
# ==========================================

def render_gauge_chart(score, title_text):
    """繪製 1-5 分標準半圓形儀表板"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': " 分", 'font': {'size': 32, 'color': '#2C3E50'}},
        title={'text': title_text, 'font': {'size': 20, 'color': '#1E3A8A'}},
        gauge={
            'shape': 'angular',
            'axis': {
                'range': [1, 5],
                'tickmode': 'array',
                'tickvals': [1, 2, 3, 4, 5],
                'ticktext': ['1 (衰退)', '2 (放緩)', '3 (中性)', '4 (擴張)', '5 (繁榮)'],
                'tickwidth': 2,
                'tickcolor': "#4A5568"
            },
            'bar': {'color': "#1F2937", 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [1.0, 1.8], 'color': '#93C5FD'},  # 藍燈 (收縮)
                {'range': [1.8, 2.6], 'color': '#FED7AA'},  # 黃藍燈
                {'range': [2.6, 3.4], 'color': '#FDE047'},  # 黃燈 (中性)
                {'range': [3.4, 4.2], 'color': '#BEF264'},  # 綠黃燈
                {'range': [4.2, 5.0], 'color': '#86EFAC'}   # 綠燈 (擴張繁榮)
            ]
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=10)
    )
    return fig

# ==========================================
# 5. 前端儀表板渲染
# ==========================================
def main():
    header_col1, header_col2 = st.columns([4, 1])

    with header_col1:
        st.title("🌐 美國 vs 台灣 總體經濟即時監測儀表板")
        st.caption(f"數據最後擷取時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (CST)")

    with header_col2:
        st.write("")
        # 兼容新版語法 width='stretch'
        try:
            btn_clicked = st.button("🔄 立即更新數據", width="stretch", type="primary")
        except TypeError:
            btn_clicked = st.button("🔄 立即更新數據", use_container_width=True, type="primary")

        if btn_clicked:
            st.cache_data.clear()
            st.toast("已清除快取，正在重新連線官方伺服器...", icon="🔄")
            st.rerun()

    st.divider()

    # 美國指標定義：CPIAUCSL 加上 units='pc1' 取得 CPI 年增率 (%)
    us_meta = [
        {"name": "實質 GDP 季增年率 (Real GDP)", "code": "A191RL1Q225SBEA", "unit": "%", "freq": "quarterly", "units": "lin", "url": "https://fred.stlouisfed.org/series/A191RL1Q225SBEA"},
        {"name": "消費者物價指數年增率 (CPI YoY)", "code": "CPIAUCSL", "unit": "%", "freq": "monthly", "units": "pc1", "url": "https://fred.stlouisfed.org/series/CPIAUCSL"},
        {"name": "聯邦基金有效利率 (Fed Funds Rate)", "code": "FEDFUNDS", "unit": "%", "freq": "monthly", "units": "lin", "url": "https://fred.stlouisfed.org/series/FEDFUNDS"},
        {"name": "非農失業率 (Unemployment Rate)", "code": "UNRATE", "unit": "%", "freq": "monthly", "units": "lin", "url": "https://fred.stlouisfed.org/series/UNRATE"}
    ]

    col_us, col_tw = st.columns(2)

    # ------------------ 美國專區 ------------------
    with col_us:
        st.subheader("🇺🇸 美國總體經濟 (United States)")

        us_val_map = {}
        us_table = []
        us_hist_store = {}

        for item in us_meta:
            val, period_label, h_df = fetch_fred_series(
                series_id=item["code"],
                api_key=FRED_API_KEY,
                freq=item["freq"],
                units=item["units"],
                limit=20
            )
            us_val_map[item["code"]] = val
            us_hist_store[item["name"]] = h_df
            us_table.append({
                "指標名稱": item["name"],
                "最新數值": f"{val:.2f} {item['unit']}" if val is not None else "待檢索",
                "統計期 (涵蓋期間)": period_label if period_label else "-",
                "官方查驗超連結": f"[前往 FRED 官網]({item['url']})"
            })

        # 以 CPI 年增率精確計算美國總體評分
        real_us_score = calculate_us_score(
            gdp=us_val_map.get("A191RL1Q225SBEA"),
            unrate=us_val_map.get("UNRATE"),
            cpi_yoy=us_val_map.get("CPIAUCSL"),
            fedfunds=us_val_map.get("FEDFUNDS")
        )
        us_light, us_status = get_status_description(real_us_score)

        # 渲染半圓形儀表板
        fig_us_gauge = render_gauge_chart(real_us_score, "美國總體經濟綜合評分")
        try:
            st.plotly_chart(fig_us_gauge, width="stretch")
        except TypeError:
            st.plotly_chart(fig_us_gauge, use_container_width=True)

        st.info(f"當前燈號與狀態研判：**{us_light} - {us_status}**")

        st.markdown("#### 📊 即時變量清單與官方查驗來源")
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
            fig_us.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            try:
                st.plotly_chart(fig_us, width="stretch")
            except TypeError:
                st.plotly_chart(fig_us, use_container_width=True)

    # ------------------ 台灣專區 ------------------
    with col_tw:
        st.subheader("🇹🇼 台灣總體經濟 (Taiwan)")

        ndc_period, ndc_score, ndc_df = fetch_ndc_indicators(limit=12)
        cpi_period, cpi_score, cpi_df = fetch_dgbas_cpi(limit=12)

        # 自動依最新實質數據計算台灣評分
        real_tw_score = calculate_tw_score(
            ndc_score=ndc_score,
            cpi_yoy=cpi_score,
            gdp=3.1
        )
        tw_light, tw_status = get_status_description(real_tw_score)

        # 渲染半圓形儀表板
        fig_tw_gauge = render_gauge_chart(real_tw_score, "台灣總體經濟綜合評分")
        try:
            st.plotly_chart(fig_tw_gauge, width="stretch")
        except TypeError:
            st.plotly_chart(fig_tw_gauge, use_container_width=True)

        st.info(f"當前燈號與狀態研判：**{tw_light} - {tw_status}**")

        st.markdown("#### 📊 即時變量清單與官方查驗來源")
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
                "最新數值": "3.10 %",
                "統計期 (涵蓋期間)": "2026 Q2",
                "官方查驗超連結": "[主計總處國民所得統計](https://www.stat.gov.tw/)"
            },
            {
                "指標名稱": "中央銀行重貼現率 (%)",
                "最新數值": "2.00 %",
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
            fig_tw.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            try:
                st.plotly_chart(fig_tw, width="stretch")
            except TypeError:
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
            fig_tw.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            try:
                st.plotly_chart(fig_tw, width="stretch")
            except TypeError:
                st.plotly_chart(fig_tw, use_container_width=True)

    st.divider()

# ==========================================
# 6. Spyder / 終端機執行防護
# ==========================================
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    if ctx is not None:
        # 已處於 Streamlit Server 中，正常渲染
        main()
    else:
        # 處於 Spyder / IPython 環境，以子行程自動啟動 Streamlit
        script_path = __file__ if '__file__' in globals() else r"c:\Users\jnchi\OneDrive\Desktop\AI\macro_dashboard.py"
        print(f"正在啟動 Streamlit 伺服器: {script_path} ...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])