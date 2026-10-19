# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:23:14 2026

@author: jnchi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import re
from datetime import datetime

# -------------------------------------------------------------
# 頁面配置
# -------------------------------------------------------------
st.set_page_config(
    page_title="個股累計季財務五力分析儀表板 (同業自動識別校準版)",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------------
# 模組一：同業族群自動辨識與推薦引擎
# -------------------------------------------------------------
# 建立常見產業/供應鏈同業預設對照庫 (涵蓋台股電子權值與美股科技龍頭)
PRESET_PEER_GROUPS = {
    # 晶圓代工 / 半導體製造
    "2330.TW": ["2303.TW", "5347.TWO", "6770.TW", "TSM"],
    "2303.TW": ["2330.TW", "5347.TWO", "6770.TW"],
    "5347.TWO": ["2330.TW", "2303.TW", "6770.TW"],
    "6770.TW": ["2303.TW", "5347.TWO", "2330.TW"],
    
    # IC 設計 / SoC
    "2454.TW": ["2379.TW", "3034.TW", "3443.TW", "3661.TW", "QCOM"],
    "2379.TW": ["2454.TW", "3034.TW", "6415.TWO"],
    "3034.TW": ["2454.TW", "2379.TW", "4961.TW"],
    "3661.TW": ["3443.TW", "3035.TW", "2454.TW"],
    
    # AI 伺服器 / 系統組裝 (ODM)
    "6669.TW": ["2382.TW", "3231.TW", "2356.TW", "2376.TW"], # 緯穎
    "2382.TW": ["6669.TW", "3231.TW", "2317.TW", "2356.TW"], # 廣達
    "3231.TW": ["2382.TW", "6669.TW", "2356.TW", "2317.TW"], # 緯創
    "2317.TW": ["2382.TW", "3231.TW", "4938.TW", "2356.TW"], # 鴻海
    "2356.TW": ["2382.TW", "3231.TW", "6669.TW", "2376.TW"], # 英業達
    
    # 被動元件
    "2327.TW": ["2492.TW", "3026.TW", "2456.TW"], # 國巨
    "2492.TW": ["2327.TW", "3026.TW", "6173.TWO"], # 華新科
    
    # 記憶體製造 / 模組
    "2408.TW": ["8299.TW", "2344.TW", "2337.TW", "MU"], # 南亞科
    "8299.TW": ["2408.TW", "3260.TWO", "2451.TW"], # 群聯
    
    # 連接器 / 線束 (伺服器/車用)
    "3665.TW": ["3023.TW", "3533.TW", "3003.TW"], # 貿聯-KY
    
    # 美股科技龍頭對照
    "NVDA": ["AMD", "INTC", "TSM", "AVGO", "QCOM"],
    "AMD": ["NVDA", "INTC", "QCOM", "AVGO"],
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META"]
}

def auto_detect_peer_group(ticker: str) -> list:
    """
    依據輸入個股代碼自動識別同業族群代碼。
    若未收錄於精準對照表，則向 Yahoo Finance 查詢產業 (Industry) 進行通用歸類。
    """
    clean_ticker = ticker.strip().upper()
    if clean_ticker in PRESET_PEER_GROUPS:
        return PRESET_PEER_GROUPS[clean_ticker]
    
    # 次級檢索：透過 yfinance info 擷取產業資訊
    try:
        info = yf.Ticker(clean_ticker).info
        industry = info.get("industry", "")
        sector = info.get("sector", "")

        # 根據業界分類指派同業基準
        if "Semiconductor" in industry or "Semiconductors" in industry:
            return ["2330.TW", "2303.TW", "2454.TW", "TSM"]
        elif "Computer" in industry or "Hardware" in industry:
            return ["2382.TW", "3231.TW", "2317.TW", "6669.TW"]
        elif "Technology" in sector:
            return ["2330.TW", "2454.TW", "2382.TW"]
    except Exception:
        pass

    # 預設通用大型權值池
    return ["2330.TW", "2303.TW", "2454.TW"]

# -------------------------------------------------------------
# 模組二：財務資料抓取核心模組 (含成長力降級備援機制)
# -------------------------------------------------------------
def fetch_single_ticker_ratios(ticker: str, target_year: int, target_quarter: int):
    try:
        stock = yf.Ticker(ticker.strip())
        bs = stock.quarterly_balance_sheet
        inc = stock.quarterly_income_stmt

        if bs.empty or inc.empty:
            return None

        inc_cols = sorted(inc.columns)
        bs_cols = sorted(bs.columns)

        current_year_cols = [c for c in inc_cols if c.year == target_year]
        if len(current_year_cols) < target_quarter:
            return None

        selected_quarters = current_year_cols[:target_quarter]
        curr_q_date = selected_quarters[-1]

        # 1. 流量累計加總 (YTD)
        ytd_revenue = sum(inc.loc["Total Revenue", q] for q in selected_quarters if "Total Revenue" in inc.index)
        ytd_op_income = sum(inc.loc["Operating Income", q] for q in selected_quarters if "Operating Income" in inc.index)
        ytd_net_income = sum(inc.loc["Net Income", q] for q in selected_quarters if "Net Income" in inc.index)

        # 單季數據 (環比 QoQ 與 單季 YoY 備援用)
        curr_single_revenue = inc.loc["Total Revenue", curr_q_date] if "Total Revenue" in inc.index else 0
        curr_single_net = inc.loc["Net Income", curr_q_date] if "Net Income" in inc.index else 0

        # 期末資產負債表科目
        matching_bs = [c for c in bs_cols if c.year == curr_q_date.year and abs(c.month - curr_q_date.month) <= 1]
        bs_date = matching_bs[-1] if matching_bs else bs_cols[-1]

        curr_assets = bs.loc["Total Assets", bs_date] if "Total Assets" in bs.index else 1
        curr_equity = bs.loc["Stockholders Equity", bs_date] if "Stockholders Equity" in bs.index else 1
        curr_ca = bs.loc["Current Assets", bs_date] if "Current Assets" in bs.index else 0
        curr_cl = bs.loc["Current Liabilities", bs_date] if "Current Liabilities" in bs.index else 1
        curr_debt = bs.loc["Total Debt", bs_date] if "Total Debt" in bs.index else (curr_assets - curr_equity)

        # 2. 去年同期歷史季度比對
        prior_year_cols = sorted([c for c in inc_cols if c.year == (target_year - 1)])
        prior_ytd_revenue = None
        prior_ytd_op_income = None
        prior_single_rev_yoy = None

        if len(prior_year_cols) >= target_quarter:
            prior_quarters = prior_year_cols[:target_quarter]
            prior_ytd_revenue = sum(inc.loc["Total Revenue", q] for q in prior_quarters if "Total Revenue" in inc.index)
            prior_ytd_op_income = sum(inc.loc["Operating Income", q] for q in prior_quarters if "Operating Income" in inc.index)
            prior_single_rev_yoy = inc.loc["Total Revenue", prior_quarters[-1]] if "Total Revenue" in inc.index else None

        # 3. 前一季歷史季度比對 (QoQ 環比)
        curr_idx = inc_cols.index(curr_q_date)
        prior_single_rev_qoq = inc.loc["Total Revenue", inc_cols[curr_idx - 1]] if (curr_idx > 0 and "Total Revenue" in inc.index) else None
        prior_single_net_qoq = inc.loc["Net Income", inc_cols[curr_idx - 1]] if (curr_idx > 0 and "Net Income" in inc.index) else None

        # 4. 財務比率年化與指標運算
        annualize_factor = 4.0 / target_quarter
        roe = (ytd_net_income * annualize_factor / curr_equity) * 100
        op_margin = (ytd_op_income / ytd_revenue) * 100 if ytd_revenue != 0 else 0
        asset_turnover = (ytd_revenue * annualize_factor) / curr_assets
        current_ratio = (curr_ca / curr_cl) * 100
        debt_ratio = (curr_debt / curr_assets) * 100

        # 5. 成長力指標層級降級備援機制 (Fallback Waterfall)
        growth_val = np.nan
        growth_metric_name = "累計營收 YoY"

        if prior_ytd_revenue is not None and prior_ytd_revenue > 0:
            growth_val = ((ytd_revenue - prior_ytd_revenue) / prior_ytd_revenue) * 100
            growth_metric_name = "累計營收 YoY"
        elif prior_single_rev_yoy is not None and prior_single_rev_yoy > 0:
            growth_val = ((curr_single_revenue - prior_single_rev_yoy) / prior_single_rev_yoy) * 100
            growth_metric_name = "單季營收 YoY (備援)"
        elif prior_ytd_op_income is not None and prior_ytd_op_income != 0:
            growth_val = ((ytd_op_income - prior_ytd_op_income) / abs(prior_ytd_op_income)) * 100
            growth_metric_name = "累計營益 YoY (備援)"
        elif prior_single_rev_qoq is not None and prior_single_rev_qoq > 0:
            growth_val = ((curr_single_revenue - prior_single_rev_qoq) / prior_single_rev_qoq) * 100
            growth_metric_name = "單季營收 QoQ (備援)"
        else:
            growth_val = 0.0
            growth_metric_name = "數據缺失 (預設中性)"

        rev_qoq = ((curr_single_revenue - prior_single_rev_qoq) / prior_single_rev_qoq * 100) if (prior_single_rev_qoq and prior_single_rev_qoq > 0) else np.nan
        net_qoq = ((curr_single_net - prior_single_net_qoq) / abs(prior_single_net_qoq) * 100) if (prior_single_net_qoq and prior_single_net_qoq != 0) else np.nan

        return {
            "ticker": ticker.strip().upper(),
            "roe": roe,
            "operating_margin": op_margin,
            "asset_turnover": asset_turnover,
            "current_ratio": current_ratio,
            "debt_ratio": debt_ratio,
            "growth_val": growth_val,
            "growth_metric_name": growth_metric_name,
            "rev_qoq": rev_qoq,
            "net_qoq": net_qoq,
            "report_date": curr_q_date.strftime("%Y-%m-%d"),
            "bs_date": bs_date.strftime("%Y-%m-%d"),
            "inc_date_range": f"{selected_quarters[0].strftime('%Y-%m-%d')} ~ {curr_q_date.strftime('%Y-%m-%d')}"
        }
    except Exception:
        return None

# -------------------------------------------------------------
# 模組三：動態經驗百分位數引擎
# -------------------------------------------------------------
class DynamicPeerPercentileEngine:
    @staticmethod
    def compute_percentile_and_decile(series: pd.Series, target_value: float, ascending: bool = True):
        clean_series = series.dropna()
        if len(clean_series) == 0 or pd.isna(target_value):
            return 50.0, 5

        if ascending:
            rank = (clean_series < target_value).sum() + 0.5 * (clean_series == target_value).sum()
        else:
            rank = (clean_series > target_value).sum() + 0.5 * (clean_series == target_value).sum()

        pct = (rank / len(clean_series)) * 100.0
        decile = int(np.clip(np.floor(pct / 10) + 1, 1, 10))
        return pct, decile

# -------------------------------------------------------------
# 模組四：繪圖模組
# -------------------------------------------------------------
def plot_peer_radar_chart(target_deciles: dict, peer_median_deciles: dict, ticker: str, period: str):
    categories = ['收益力\n(Profitability)', '活動力\n(Activity)', '償債力\n(Liquidity)', 
                  '成長力\n(Growth)', '安定力\n(Solvency)']
    keys = ["profitability", "activity", "liquidity", "growth", "solvency"]
    
    scores_target = [target_deciles[k] for k in keys]
    scores_peer_med = [peer_median_deciles[k] for k in keys]

    categories_closed = categories + [categories[0]]
    scores_target_closed = scores_target + [scores_target[0]]
    scores_peer_closed = scores_peer_med + [scores_peer_med[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_target_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(26, 115, 232, 0.25)',
        line=dict(color='#1a73e8', width=3),
        name=f'{ticker} (動態分位)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=scores_peer_closed,
        theta=categories_closed,
        line=dict(color='rgba(220, 53, 69, 0.8)', dash='dash', width=2),
        name='同業中位數基準 (P50)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                ticktext=['D2', 'D4', 'D6 (中位)', 'D8', 'D10 (頂尖)']
            )
        ),
        showlegend=True,
        title=dict(text=f"<b>{ticker} vs 同業族群五力十分位校準圖</b> ({period})", x=0.5),
        height=500,
        margin=dict(l=60, r=60, t=60, b=40)
    )
    return fig

def fmt_pct(val, is_pts=False):
    if pd.isna(val):
        return "N/A"
    unit = " pts" if is_pts else "%"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}{unit}"

# -------------------------------------------------------------
# 模組五：Streamlit 前端介面
# -------------------------------------------------------------
st.title("📊 個股累計季財務五力分析儀表板 (同業自動識別校準版)")
st.caption("支援輸入代碼後自動產生推薦同業族群（Peer Group），並執行即時經驗百分位數橫向校準與層級成長力降級計算。")

col1, col2 = st.columns([1, 1])
with col1:
    ticker_input = st.text_input("目標個股代碼", value="2330.TW", help="支援台股 (加 .TW 或 .TWO) 與美股 Ticker")
    period_input = st.text_input("年度季別 (YYYYQq)", value="2026Q2", help="格式如：2026Q2、2025Q4")

# 自動推導同業族群代碼
detected_peers = auto_detect_peer_group(ticker_input)
default_peer_str = ", ".join(detected_peers)

with col2:
    peer_input = st.text_input(
        "自動生成之同業比對族群代碼 (可依研究需求自由增刪)", 
        value=default_peer_str,
        help="系統已依個股產業自動匹配同業池，亦可手動以逗號分隔新增代碼。"
    )

run_btn = st.button("執行五力動態橫向校準分析", type="primary", use_container_width=True)

if run_btn:
    match = re.match(r"^(\d{4})[Qq]([1-4])$", period_input.strip())
    if not match:
        st.error("年度季別格式不正確，請輸入如：2026Q2 或 2025Q3")
    else:
        target_year = int(match.group(1))
        target_quarter = int(match.group(2))
        sampling_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with st.spinner("正在連網抓取目標個股與同業群體即時財報數據..."):
            target_data = fetch_single_ticker_ratios(ticker_input, target_year, target_quarter)

            if target_data is None:
                st.error(f"無法取得 {ticker_input} 於 {period_input} 之有效財報數據，請確認代碼或該季度是否已發布。")
            else:
                peer_list = [p.strip().upper() for p in peer_input.split(",") if p.strip()]
                all_tickers = list(set([ticker_input.strip().upper()] + peer_list))
                
                peer_results = []
                for p_t in all_tickers:
                    res = fetch_single_ticker_ratios(p_t, target_year, target_quarter)
                    if res is not None:
                        peer_results.append(res)
                
                df_peers = pd.DataFrame(peer_results)

                # 橫向動態百分位數與十分位運算
                roe_pct, roe_dec = DynamicPeerPercentileEngine.compute_percentile_and_decile(df_peers["roe"], target_data["roe"], ascending=True)
                opm_pct, opm_dec = DynamicPeerPercentileEngine.compute_percentile_and_decile(df_peers["operating_margin"], target_data["operating_margin"], ascending=True)
                prof_pct = (roe_pct + opm_pct) / 2
                prof_dec = int(np.clip(round((roe_dec + opm_dec) / 2), 1, 10))

                act_pct, act_dec = DynamicPeerPercentileEngine.compute_percentile_and_decile(df_peers["asset_turnover"], target_data["asset_turnover"], ascending=True)
                liq_pct, liq_dec = DynamicPeerPercentileEngine.compute_percentile_and_decile(df_peers["current_ratio"], target_data["current_ratio"], ascending=True)
                gro_pct, gro_dec = DynamicPeerPercentileEngine.compute_percentile_and_decile(df_peers["growth_val"], target_data["growth_val"], ascending=True)
                sol_pct, sol_dec = DynamicPeerPercentileEngine.compute_percentile_and_decile(df_peers["debt_ratio"], target_data["debt_ratio"], ascending=False)

                target_deciles = {
                    "profitability": prof_dec,
                    "activity": act_dec,
                    "liquidity": liq_dec,
                    "growth": gro_dec,
                    "solvency": sol_dec
                }
                peer_median_deciles = {k: 5 for k in target_deciles}

                # 資料來源與稽核列
                source_url = f"https://finance.yahoo.com/quote/{ticker_input.strip()}/financials/"
                st.info(
                    f"**數據來源機構**：[Yahoo Finance Official API]({source_url}) ｜ "
                    f"**系統取樣時間**：`{sampling_time}` ｜ "
                    f"**自動識別納入樣本數**：共 `{len(df_peers)}` 檔標的 ｜ "
                    f"**成長力採用指標**：`{target_data['growth_metric_name']}`"
                )

                # 第一列：雷達圖與評分卡
                r1_c1, r1_c2 = st.columns([3, 2])
                with r1_c1:
                    st.plotly_chart(plot_peer_radar_chart(target_deciles, peer_median_deciles, ticker_input, period_input), use_container_width=True)

                with r1_c2:
                    st.subheader("同業動態橫向校準十分位總結")
                    overall_pct = np.mean([prof_pct, act_pct, liq_pct, gro_pct, sol_pct])
                    overall_dec = np.mean(list(target_deciles.values()))
                    
                    st.metric("同業群體綜合百分位數", f"PR {overall_pct:.1f}%", f"十分位評級: D{overall_dec:.1f}")
                    st.divider()
                    st.markdown(f"""
                    - **收益力**：`Decile {prof_dec}`（同業 PR: `{prof_pct:.1f}%` ｜ ROE: `{target_data['roe']:.1f}%`）
                    - **活動力**：`Decile {act_dec}`（同業 PR: `{act_pct:.1f}%` ｜ 資產週轉率: `{target_data['asset_turnover']:.2f}` 次/年）
                    - **償債力**：`Decile {liq_dec}`（同業 PR: `{liq_pct:.1f}%` ｜ 流動比率: `{target_data['current_ratio']:.1f}%`）
                    - **成長力**：`Decile {gro_dec}`（同業 PR: `{gro_pct:.1f}%` ｜ {target_data['growth_metric_name']}: `{fmt_pct(target_data['growth_val'])}`）
                    - **安定力**：`Decile {sol_dec}`（同業 PR: `{sol_pct:.1f}%` ｜ 負債比率: `{target_data['debt_ratio']:.1f}%`）
                    """)

                # 第二列：指標橫向對照表
                st.subheader("五力指標橫向對齊比較表 (含同業中位數與分位排名)")
                metrics_summary = [
                    {
                        "分析面向": "收益力 (Profitability)",
                        "代表指標": "累計年化 ROE / 累計營益率",
                        "目標個股數值": f"{target_data['roe']:.2f}% / {target_data['operating_margin']:.2f}%",
                        "同業中位數 (P50)": f"{df_peers['roe'].median():.2f}% / {df_peers['operating_margin'].median():.2f}%",
                        "同業動態百分位 (PR)": f"{prof_pct:.1f}%",
                        "十分位評分": f"D{prof_dec}"
                    },
                    {
                        "分析面向": "活動力 (Activity)",
                        "代表指標": "累計年化總資產週轉率",
                        "目標個股數值": f"{target_data['asset_turnover']:.2f} 次/年",
                        "同業中位數 (P50)": f"{df_peers['asset_turnover'].median():.2f} 次/年",
                        "同業動態百分位 (PR)": f"{act_pct:.1f}%",
                        "十分位評分": f"D{act_dec}"
                    },
                    {
                        "分析面向": "償債力 (Liquidity)",
                        "代表指標": "期末流動比率",
                        "目標個股數值": f"{target_data['current_ratio']:.2f}%",
                        "同業中位數 (P50)": f"{df_peers['current_ratio'].median():.2f}%",
                        "同業動態百分位 (PR)": f"{liq_pct:.1f}%",
                        "十分位評分": f"D{liq_dec}"
                    },
                    {
                        "分析面向": "成長力 (Growth)",
                        "代表指標": target_data["growth_metric_name"],
                        "目標個股數值": fmt_pct(target_data["growth_val"]),
                        "同業中位數 (P50)": fmt_pct(df_peers["growth_val"].median()),
                        "同業動態百分位 (PR)": f"{gro_pct:.1f}%",
                        "十分位評分": f"D{gro_dec}"
                    },
                    {
                        "分析面向": "安定力 (Solvency)",
                        "代表指標": "期末負債比率 (逆向)",
                        "目標個股數值": f"{target_data['debt_ratio']:.2f}%",
                        "同業中位數 (P50)": f"{df_peers['debt_ratio'].median():.2f}%",
                        "同業動態百分位 (PR)": f"{sol_pct:.1f}%",
                        "十分位評分": f"D{sol_dec}"
                    },
                ]
                st.dataframe(pd.DataFrame(metrics_summary), use_container_width=True)

                # 第三列：同業群體明細
                with st.expander("🔍 檢視本次自動納入同業群體（Peer Pool）完整財務數據清單"):
                    display_peer_df = df_peers[[
                        "ticker", "roe", "operating_margin", "asset_turnover", 
                        "current_ratio", "debt_ratio", "growth_val", "growth_metric_name", "report_date"
                    ]].copy()
                    display_peer_df.columns = [
                        "代碼", "累計年化 ROE(%)", "累計營益率(%)", "資產週轉率(次)", 
                        "流動比率(%)", "負債比率(%)", "成長率(%)", "成長指標採用項目", "財報結算日"
                    ]
                    st.dataframe(display_peer_df, use_container_width=True)