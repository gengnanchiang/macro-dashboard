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
    page_title="美台個股累計季財務五力分析儀表板 (含同比與環比)",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------------
# 核心計算模組：十分位數（Decile）轉換引擎
# -------------------------------------------------------------
class FinancialDecileEngine:
    THRESHOLDS = {
        "roe": [0, 4, 8, 12, 16, 20, 24, 28, 35],                 # % (累計年化)
        "operating_margin": [-2, 3, 7, 12, 18, 25, 32, 40, 50],   # % (累計利益率)
        "asset_turnover": [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5], # 次/年 (累計年化)
        "current_ratio": [80, 100, 120, 150, 180, 220, 280, 350, 450],   # %
        "revenue_growth": [-15, -5, 0, 5, 10, 18, 25, 35, 50],    # % (累計 YoY)
        "debt_ratio": [80, 70, 60, 55, 50, 45, 40, 35, 25]        # % (逆向指標)
    }

    @classmethod
    def calculate_score(cls, metric_name: str, value: float) -> int:
        if pd.isna(value) or np.isinf(value):
            return 1
        thresholds = cls.THRESHOLDS.get(metric_name, [])
        if metric_name == "debt_ratio":
            for i, th in enumerate(thresholds):
                if value >= th:
                    return i + 1
            return 10
        else:
            for i, th in enumerate(thresholds):
                if value <= th:
                    return i + 1
            return 10

# -------------------------------------------------------------
# 累計季資料擷取、指標運算、同比與環比模組
# -------------------------------------------------------------
def fetch_cumulative_financial_ratios(ticker: str, period_str: str):
    sampling_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sampling_date = datetime.now().strftime("%Y-%m-%d")
    
    match = re.match(r"^(\d{4})[Qq]([1-4])$", period_str.strip())
    if not match:
        return None, "年度季別格式錯誤，請輸入如：2026Q2 或 2025Q4"
    
    target_year = int(match.group(1))
    target_quarter = int(match.group(2))

    stock = yf.Ticker(ticker)
    bs = stock.quarterly_balance_sheet
    inc = stock.quarterly_income_stmt

    if bs.empty or inc.empty:
        return None, f"查無 {ticker} 之季度財報數據，請確認代碼與網路連線。"

    try:
        inc_cols = sorted(inc.columns)
        bs_cols = sorted(bs.columns)

        # 篩選當年度季度
        current_year_cols = [c for c in inc_cols if c.year == target_year]

        if len(current_year_cols) < target_quarter:
            latest_available = f"{inc_cols[-1].year}Q{(inc_cols[-1].month - 1) // 3 + 1}"
            return None, f"{target_year}Q{target_quarter} 財報尚未完全公告或數據未更新（系統最新可取得季度為 {latest_available}）。"

        # 本期累計 (YTD)
        selected_quarters = current_year_cols[:target_quarter]
        curr_q_date = selected_quarters[-1]
        inc_period_start = selected_quarters[0].strftime("%Y-%m-%d")
        inc_period_end = curr_q_date.strftime("%Y-%m-%d")

        ytd_revenue = sum(inc.loc["Total Revenue", q] for q in selected_quarters if "Total Revenue" in inc.index)
        ytd_op_income = sum(inc.loc["Operating Income", q] for q in selected_quarters if "Operating Income" in inc.index)
        ytd_net_income = sum(inc.loc["Net Income", q] for q in selected_quarters if "Net Income" in inc.index)

        # 單季本期 (用於流量環比 QoQ)
        curr_single_revenue = inc.loc["Total Revenue", curr_q_date] if "Total Revenue" in inc.index else 0
        curr_single_net_income = inc.loc["Net Income", curr_q_date] if "Net Income" in inc.index else 0

        # 期末資產負債表取最新累計季度
        matching_bs = [c for c in bs_cols if c.year == curr_q_date.year and abs(c.month - curr_q_date.month) <= 1]
        bs_date = matching_bs[-1] if matching_bs else bs_cols[-1]
        bs_period_date = bs_date.strftime("%Y-%m-%d")

        curr_assets = bs.loc["Total Assets", bs_date] if "Total Assets" in bs.index else 1
        curr_equity = bs.loc["Stockholders Equity", bs_date] if "Stockholders Equity" in bs.index else 1
        curr_ca = bs.loc["Current Assets", bs_date] if "Current Assets" in bs.index else 0
        curr_cl = bs.loc["Current Liabilities", bs_date] if "Current Liabilities" in bs.index else 1
        curr_debt = bs.loc["Total Debt", bs_date] if "Total Debt" in bs.index else (curr_assets - curr_equity)

        # -----------------------------
        # 1. 環比基準期 (Prior Quarter)
        # -----------------------------
        curr_idx_inc = inc_cols.index(curr_q_date)
        has_qoq_inc = curr_idx_inc > 0
        prior_q_revenue = inc.loc["Total Revenue", inc_cols[curr_idx_inc - 1]] if has_qoq_inc and "Total Revenue" in inc.index else None
        prior_q_net_income = inc.loc["Net Income", inc_cols[curr_idx_inc - 1]] if has_qoq_inc and "Net Income" in inc.index else None

        curr_idx_bs = bs_cols.index(bs_date)
        has_qoq_bs = curr_idx_bs > 0
        prev_bs_date = bs_cols[curr_idx_bs - 1] if has_qoq_bs else None
        
        prev_cr, prev_dr = None, None
        if prev_bs_date is not None:
            prev_ca = bs.loc["Current Assets", prev_bs_date] if "Current Assets" in bs.index else 0
            prev_cl = bs.loc["Current Liabilities", prev_bs_date] if "Current Liabilities" in bs.index else 1
            prev_ta = bs.loc["Total Assets", prev_bs_date] if "Total Assets" in bs.index else 1
            prev_eq = bs.loc["Stockholders Equity", prev_bs_date] if "Stockholders Equity" in bs.index else 1
            prev_d = bs.loc["Total Debt", prev_bs_date] if "Total Debt" in bs.index else (prev_ta - prev_eq)
            prev_cr = (prev_ca / prev_cl) * 100
            prev_dr = (prev_d / prev_ta) * 100

        # -----------------------------
        # 2. 同比基準期 (Prior Year Same Period)
        # -----------------------------
        prior_year_cols = sorted([c for c in inc_cols if c.year == (target_year - 1)])
        has_yoy_inc = len(prior_year_cols) >= target_quarter
        prior_ytd_revenue, prior_ytd_net_income = None, None
        prior_period_range = "無足夠歷史季度"

        if has_yoy_inc:
            prior_quarters = prior_year_cols[:target_quarter]
            prior_ytd_revenue = sum(inc.loc["Total Revenue", q] for q in prior_quarters if "Total Revenue" in inc.index)
            prior_ytd_net_income = sum(inc.loc["Net Income", q] for q in prior_quarters if "Net Income" in inc.index)
            prior_period_range = f"{prior_quarters[0].strftime('%Y-%m-%d')} ~ {prior_quarters[-1].strftime('%Y-%m-%d')}"

        matching_prior_bs = [c for c in bs_cols if c.year == (target_year - 1) and abs(c.month - curr_q_date.month) <= 1]
        prior_bs_cr, prior_bs_dr = None, None
        if matching_prior_bs:
            p_bs_date = matching_prior_bs[-1]
            p_ca = bs.loc["Current Assets", p_bs_date] if "Current Assets" in bs.index else 0
            p_cl = bs.loc["Current Liabilities", p_bs_date] if "Current Liabilities" in bs.index else 1
            p_ta = bs.loc["Total Assets", p_bs_date] if "Total Assets" in bs.index else 1
            p_eq = bs.loc["Stockholders Equity", p_bs_date] if "Stockholders Equity" in bs.index else 1
            p_d = bs.loc["Total Debt", p_bs_date] if "Total Debt" in bs.index else (p_ta - p_eq)
            prior_bs_cr = (p_ca / p_cl) * 100
            prior_bs_dr = (p_d / p_ta) * 100

        # -----------------------------
        # 3. 財務比率運算與同比/環比計算
        # -----------------------------
        annualize_factor = 4.0 / target_quarter

        roe_annualized = (ytd_net_income * annualize_factor / curr_equity) * 100
        op_margin_cumulative = (ytd_op_income / ytd_revenue) * 100 if ytd_revenue != 0 else 0
        asset_turnover_annualized = (ytd_revenue * annualize_factor) / curr_assets
        current_ratio = (curr_ca / curr_cl) * 100
        debt_ratio = (curr_debt / curr_assets) * 100

        # 營收 同比 (累計 YoY) & 環比 (單季 QoQ)
        rev_yoy = ((ytd_revenue - prior_ytd_revenue) / prior_ytd_revenue * 100) if (prior_ytd_revenue and prior_ytd_revenue > 0) else np.nan
        rev_qoq = ((curr_single_revenue - prior_q_revenue) / prior_q_revenue * 100) if (prior_q_revenue and prior_q_revenue > 0) else np.nan

        # 淨利 同比 (累計 YoY) & 環比 (單季 QoQ)
        net_yoy = ((ytd_net_income - prior_ytd_net_income) / abs(prior_ytd_net_income) * 100) if prior_ytd_net_income else np.nan
        net_qoq = ((curr_single_net_income - prior_q_net_income) / abs(prior_q_net_income) * 100) if prior_q_net_income else np.nan

        # 存量比率增減 (以百分點 percentage points, pts 呈現)
        cr_qoq_diff = (current_ratio - prev_cr) if prev_cr is not None else np.nan
        cr_yoy_diff = (current_ratio - prior_bs_cr) if prior_bs_cr is not None else np.nan
        dr_qoq_diff = (debt_ratio - prev_dr) if prev_dr is not None else np.nan
        dr_yoy_diff = (debt_ratio - prior_bs_dr) if prior_bs_dr is not None else np.nan

        clean_ticker = ticker.strip()
        data_source_name = "Yahoo Finance Official API"
        source_url = f"https://finance.yahoo.com/quote/{clean_ticker}/financials/"

        metrics = {
            "roe": roe_annualized,
            "operating_margin": op_margin_cumulative,
            "asset_turnover": asset_turnover_annualized,
            "current_ratio": current_ratio,
            "debt_ratio": debt_ratio,
            "revenue_growth": rev_yoy,
            "target_period": f"{target_year}Q{target_quarter} (累計前 {target_quarter} 季)",
            "annualize_factor": f"{annualize_factor:.2f}x",
            # 來源與時間
            "data_source": data_source_name,
            "source_url": source_url,
            "sampling_date": sampling_date,
            "sampling_timestamp": sampling_timestamp,
            "inc_date_range": f"{inc_period_start} ~ {inc_period_end}",
            "prior_inc_date_range": prior_period_range,
            "bs_date": bs_period_date,
            # 同比與環比數據
            "rev_yoy": rev_yoy,
            "rev_qoq": rev_qoq,
            "net_yoy": net_yoy,
            "net_qoq": net_qoq,
            "cr_qoq_diff": cr_qoq_diff,
            "cr_yoy_diff": cr_yoy_diff,
            "dr_qoq_diff": dr_qoq_diff,
            "dr_yoy_diff": dr_yoy_diff,
        }
        return metrics, None
    except Exception as e:
        return None, f"解析報表時發生錯誤：{str(e)}"

# -------------------------------------------------------------
# 繪圖模組：五力分析雷達圖
# -------------------------------------------------------------
def plot_radar_chart(decile_scores: dict, ticker: str, period_label: str):
    categories = ['收益力\n(Profitability)', '活動力\n(Activity)', '償債力\n(Liquidity)', 
                  '成長力\n(Growth)', '安定力\n(Solvency)']
    scores = [
        decile_scores["profitability"],
        decile_scores["activity"],
        decile_scores["liquidity"],
        decile_scores["growth"],
        decile_scores["solvency"]
    ]
    categories_closed = categories + [categories[0]]
    scores_closed = scores + [scores[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(26, 115, 232, 0.25)',
        line=dict(color='#1a73e8', width=2.5),
        name=f'{ticker} ({period_label})'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[5, 5, 5, 5, 5, 5],
        theta=categories_closed,
        line=dict(color='rgba(150, 150, 150, 0.6)', dash='dash'),
        name='市場中位數基準 (Decile 5)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                ticktext=['D2', 'D4', 'D6', 'D8', 'D10']
            )
        ),
        showlegend=True,
        title=dict(text=f"<b>{ticker} 累計五力十分位雷達圖</b>", x=0.5),
        height=480,
        margin=dict(l=60, r=60, t=60, b=40)
    )
    return fig

# 輔助格式化字串
def fmt_pct(val, is_pts=False):
    if pd.isna(val):
        return "N/A"
    unit = " pts" if is_pts else "%"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}{unit}"

# -------------------------------------------------------------
# Streamlit 前端介面
# -------------------------------------------------------------
st.title("📊 個股累計季財務五力分析儀表板 (含同比與環比)")
st.caption("支援指定年度季別之累計財務五力十分位數評級，並完整揭露資料來源、取樣日期、同比 (YoY) 與環比 (QoQ) 變動率。")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    ticker_input = st.text_input("個股代碼", value="2330.TW", help="台股代碼後加 .TW 或 .TWO，美股直接輸入 Ticker")
with col2:
    period_input = st.text_input("年度季別 (YYYYQq)", value="2026Q2", help="格式：2026Q2、2025Q3")
with col3:
    st.write("")
    st.write("")
    run_btn = st.button("計算累計五力", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("正在連線擷取即時財報並運算同比、環比與五力分位..."):
        data, err = fetch_cumulative_financial_ratios(ticker_input, period_input)
        
        if err:
            st.warning(err)
        else:
            # 轉換十分位評分
            score_prof = round((FinancialDecileEngine.calculate_score("roe", data["roe"]) + 
                                FinancialDecileEngine.calculate_score("operating_margin", data["operating_margin"])) / 2)
            score_act = FinancialDecileEngine.calculate_score("asset_turnover", data["asset_turnover"])
            score_liq = FinancialDecileEngine.calculate_score("current_ratio", data["current_ratio"])
            score_sol = FinancialDecileEngine.calculate_score("debt_ratio", data["debt_ratio"])
            score_gro = FinancialDecileEngine.calculate_score("revenue_growth", data["revenue_growth"]) if not pd.isna(data["revenue_growth"]) else 5

            decile_dict = {
                "profitability": score_prof,
                "activity": score_act,
                "liquidity": score_liq,
                "growth": score_gro,
                "solvency": score_sol
            }

            # -------------------------------------------------------------
            # 資料來源與取樣資訊面板
            # -------------------------------------------------------------
            st.info(
                f"**數據來源機構**：[{data['data_source']}]({data['source_url']}) ｜ "
                f"**系統取樣日期**：`{data['sampling_date']}`（取樣時間：`{data['sampling_timestamp']}`） ｜ "
                f"**分析對象與期別**：`{ticker_input}` / `{data['target_period']}`"
            )

            # 第一列：雷達圖與評分卡
            c_left, c_right = st.columns([3, 2])
            with c_left:
                st.plotly_chart(plot_radar_chart(decile_dict, ticker_input, period_input), use_container_width=True)
            
            with c_right:
                st.subheader("五力十分位表現 (Decile 1–10)")
                avg_decile = np.mean(list(decile_dict.values()))
                st.metric("五力綜合平均十分位", f"D{avg_decile:.1f}", 
                          delta="前段班領先" if avg_decile >= 7 else ("居中" if avg_decile >= 4.5 else "偏弱"))
                
                st.divider()
                st.markdown(f"""
                - **收益力**：`Decile {score_prof}`（累計年化 ROE: `{data['roe']:.1f}%`，累計營益率: `{data['operating_margin']:.1f}%`）
                - **活動力**：`Decile {score_act}`（累計年化資產週轉率: `{data['asset_turnover']:.2f}` 次/年）
                - **償債力**：`Decile {score_liq}`（期末流動比率: `{data['current_ratio']:.1f}%`）
                - **成長力**：`Decile {score_gro}`（累計營收 YoY: `{fmt_pct(data['revenue_growth'])}`）
                - **安定力**：`Decile {score_sol}`（期末負債比率: `{data['debt_ratio']:.1f}%`）
                """)

            # -------------------------------------------------------------
            # 第二列：核心指標同比 (YoY) 與環比 (QoQ) 變動卡片
            # -------------------------------------------------------------
            st.subheader("核心財務指標 同比 (YoY) 與 環比 (QoQ) 變動概覽")
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

            kpi_c1.metric(
                label="累計營業收入",
                value=f"{data['target_period'].split(' ')[0]} 累計",
                delta=f"同比(YoY) {fmt_pct(data['rev_yoy'])}",
                help=f"環比(單季QoQ): {fmt_pct(data['rev_qoq'])}"
            )
            kpi_c2.metric(
                label="累計稅後淨利",
                value=f"{data['target_period'].split(' ')[0]} 累計",
                delta=f"同比(YoY) {fmt_pct(data['net_yoy'])}",
                help=f"環比(單季QoQ): {fmt_pct(data['net_qoq'])}"
            )
            kpi_c3.metric(
                label="期末流動比率",
                value=f"{data['current_ratio']:.2f}%",
                delta=f"環比 {fmt_pct(data['cr_qoq_diff'], is_pts=True)}",
                help=f"同比變動: {fmt_pct(data['cr_yoy_diff'], is_pts=True)}"
            )
            kpi_c4.metric(
                label="期末負債比率",
                value=f"{data['debt_ratio']:.2f}%",
                delta=f"環比 {fmt_pct(data['dr_qoq_diff'], is_pts=True)}",
                delta_color="inverse", # 負債比上升為警訊
                help=f"同比變動: {fmt_pct(data['dr_yoy_diff'], is_pts=True)}"
            )

            # -------------------------------------------------------------
            # 第三列：詳細報表、同比/環比與取樣歷程表
            # -------------------------------------------------------------
            st.subheader("財務比率明細暨同比、環比與資料來源歷程")
            table_data = [
                {
                    "分析面向": "收益力 (Profitability)",
                    "代表指標": "累計年化 ROE / 累計營益率",
                    "本期數值": f"{data['roe']:.2f}% / {data['operating_margin']:.2f}%",
                    "十分位": f"D{score_prof}",
                    "環比變動 (QoQ)": f"淨利 QoQ: {fmt_pct(data['net_qoq'])}",
                    "同比變動 (YoY)": f"淨利 YoY: {fmt_pct(data['net_yoy'])}",
                    "取樣日期": data["sampling_date"],
                    "資料來源 / 涵蓋期": f"{data['data_source']} ({data['inc_date_range']})"
                },
                {
                    "分析面向": "活動力 (Activity)",
                    "代表指標": "累計年化總資產週轉率",
                    "本期數值": f"{data['asset_turnover']:.2f} 次/年",
                    "十分位": f"D{score_act}",
                    "環比變動 (QoQ)": f"營收 QoQ: {fmt_pct(data['rev_qoq'])}",
                    "同比變動 (YoY)": f"營收 YoY: {fmt_pct(data['rev_yoy'])}",
                    "取樣日期": data["sampling_date"],
                    "資料來源 / 涵蓋期": f"{data['data_source']} ({data['inc_date_range']})"
                },
                {
                    "分析面向": "償債力 (Liquidity)",
                    "代表指標": "期末流動比率",
                    "本期數值": f"{data['current_ratio']:.2f}%",
                    "十分位": f"D{score_liq}",
                    "環比變動 (QoQ)": fmt_pct(data['cr_qoq_diff'], is_pts=True),
                    "同比變動 (YoY)": fmt_pct(data['cr_yoy_diff'], is_pts=True),
                    "取樣日期": data["sampling_date"],
                    "資料來源 / 涵蓋期": f"{data['data_source']} (結算日 {data['bs_date']})"
                },
                {
                    "分析面向": "成長力 (Growth)",
                    "代表指標": "累計營收年增率 (YoY)",
                    "本期數值": fmt_pct(data['revenue_growth']),
                    "十分位": f"D{score_gro}",
                    "環比變動 (QoQ)": fmt_pct(data['rev_qoq']),
                    "同比變動 (YoY)": fmt_pct(data['rev_yoy']),
                    "取樣日期": data["sampling_date"],
                    "資料來源 / 涵蓋期": f"本期 {data['inc_date_range']} vs 基期 {data['prior_inc_date_range']}"
                },
                {
                    "分析面向": "安定力 (Solvency)",
                    "代表指標": "期末負債比率",
                    "本期數值": f"{data['debt_ratio']:.2f}%",
                    "十分位": f"D{score_sol}",
                    "環比變動 (QoQ)": fmt_pct(data['dr_qoq_diff'], is_pts=True),
                    "同比變動 (YoY)": fmt_pct(data['dr_yoy_diff'], is_pts=True),
                    "取樣日期": data["sampling_date"],
                    "資料來源 / 涵蓋期": f"{data['data_source']} (結算日 {data['bs_date']})"
                },
            ]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            st.caption(f"官方財報源頭連結：[{data['source_url']}]({data['source_url']})（點擊可直接檢視最新財務報表官方頁面）。")