# -*- coding: utf-8 -*-
"""
Created on Tue Nov 03 23:30:00 2026

@author: James Chiang
"""

import datetime
import gc
import io
import os
import time
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from sklearn.linear_model import LogisticRegression
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# 頁面基本配置（採用寬螢幕與暗色系量化終端風格）
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TAIEX 次日漲跌機率監控儀表板 (17因子-含近5日借券賣出餘額變動升級版)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card {
        background-color: #1a1e24;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #f43f5e;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .text-bullish { color: #f43f5e; } /* 台股紅漲 */
    .text-bearish { color: #10b981; } /* 台股綠跌 */

    /* 強制隱藏 Streamlit metric delta 預設自動生成的箭頭 svg */
    [data-testid="stMetricDelta"] svg {
        display: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 台灣市場休市日、國定假日與國安基金護盤期間配置表
# -----------------------------------------------------------------------------
TW_HOLIDAYS_MAP = {
    datetime.date(2024, 1, 1): "元旦開國紀念日",
    datetime.date(2024, 2, 6): "農曆春節前封關休市",
    datetime.date(2024, 2, 7): "農曆春節前休市",
    datetime.date(2024, 2, 8): "農曆小年夜調整放假",
    datetime.date(2024, 2, 9): "農曆除夕",
    datetime.date(2024, 2, 12): "春節初三",
    datetime.date(2024, 2, 13): "春節初四補假",
    datetime.date(2024, 2, 14): "春節初五調整放假",
    datetime.date(2024, 2, 28): "二二八和平紀念日",
    datetime.date(2024, 4, 4): "兒童節與清明節",
    datetime.date(2024, 4, 5): "清明節調整放假",
    datetime.date(2024, 5, 1): "勞動節",
    datetime.date(2024, 6, 10): "端午節",
    datetime.date(2024, 7, 24): "凱米颱風停班停課休市",
    datetime.date(2024, 7, 25): "凱米颱風停班停課休市",
    datetime.date(2024, 9, 17): "中秋節",
    datetime.date(2024, 10, 2): "山陀兒颱風停班停課休市",
    datetime.date(2024, 10, 3): "山陀兒颱風停班停課休市",
    datetime.date(2024, 10, 10): "國慶日",
    datetime.date(2024, 10, 31): "康芮颱風停班停課休市",
    datetime.date(2025, 1, 1): "元旦開國紀念日",
    datetime.date(2025, 1, 23): "農曆春節前封關休市",
    datetime.date(2025, 1, 24): "農曆春節前休市",
    datetime.date(2025, 1, 27): "農曆小年夜調整放假",
    datetime.date(2025, 1, 28): "農曆除夕",
    datetime.date(2025, 1, 29): "春節初一",
    datetime.date(2025, 1, 30): "春節初二",
    datetime.date(2025, 1, 31): "春節初三",
    datetime.date(2025, 2, 28): "二二八和平紀念日",
    datetime.date(2025, 4, 3): "兒童節前一日調整放假",
    datetime.date(2025, 4, 4): "兒童節與清明節",
    datetime.date(2025, 5, 1): "勞動節",
    datetime.date(2025, 5, 30): "端午節補假",
    datetime.date(2025, 10, 6): "中秋節",
    datetime.date(2025, 10, 10): "國慶日",
    datetime.date(2026, 1, 1): "元旦開國紀念日",
    datetime.date(2026, 2, 16): "農曆除夕前封關/春節調整",
    datetime.date(2026, 2, 17): "農曆除夕",
    datetime.date(2026, 2, 18): "春節初一",
    datetime.date(2026, 2, 19): "春節初二",
    datetime.date(2026, 2, 20): "春節初三",
    datetime.date(2026, 2, 27): "和平紀念日補假",
    datetime.date(2026, 2, 28): "二二八和平紀念日",
    datetime.date(2026, 4, 3): "兒童節補假",
    datetime.date(2026, 4, 6): "清明節補假",
    datetime.date(2026, 5, 1): "勞動節",
    datetime.date(2026, 6, 19): "端午節",
    datetime.date(2026, 9, 25): "中秋節",
    datetime.date(2026, 10, 9): "國慶日補假",
    datetime.date(2026, 10, 10): "國慶日",
    datetime.date(2026, 10, 23): "天然災害停止上班上課（颱風假休市）",
    datetime.date(2026, 10, 26): "光復節/國慶調移補假",
}

US_MACRO_EVENTS_MAP = {
    # 2025 年 FOMC 利率決策會議發布日
    datetime.date(2025, 1, 29): "⚡ FOMC 利率決策公布",
    datetime.date(2025, 3, 19): "⚡ FOMC 利率決策與經濟展望",
    datetime.date(2025, 5, 7): "⚡ FOMC 利率決策公布",
    datetime.date(2025, 6, 18): "⚡ FOMC 利率決策與點陣圖",
    datetime.date(2025, 7, 30): "⚡ FOMC 利率決策公布",
    datetime.date(2025, 9, 17): "⚡ FOMC 利率決策與點陣圖",
    datetime.date(2025, 10, 29): "⚡ FOMC 利率決策公布",
    datetime.date(2025, 12, 10): "⚡ FOMC 利率決策與點陣圖",
    # 2025 年 美國 CPI 通膨報告發布日 (BLS 官方日程與實際發布日)
    datetime.date(2025, 1, 15): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 2, 12): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 3, 12): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 4, 10): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 5, 13): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 6, 11): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 7, 15): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 8, 12): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 9, 11): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 10, 15): "🚨 美國 CPI 通膨發布 (原定日程)",
    datetime.date(2025, 10, 24): "🚨 美國 CPI 通膨發布",
    datetime.date(2025, 11, 13): "🚨 美國 CPI 通膨發布 (原定日程)",
    datetime.date(2025, 12, 18): "🚨 美國 CPI 通膨發布",
    # 2025 年 美國非農就業報告 (NFP) 發布日 (BLS 官方日程與實際發布日)
    datetime.date(2025, 1, 10): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 2, 7): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 3, 7): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 4, 4): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 5, 2): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 6, 6): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 7, 3): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 8, 1): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 9, 5): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 10, 3): "💼 美國非農就業報告 (NFP) (原定日程)",
    datetime.date(2025, 11, 7): "💼 美國非農就業報告 (NFP) (原定日程)",
    datetime.date(2025, 11, 20): "💼 美國非農就業報告 (NFP)",
    datetime.date(2025, 12, 5): "💼 美國非農就業報告 (NFP) (原定日程)",
    datetime.date(2025, 12, 16): "💼 美國非農就業報告 (NFP)",
    # 2026 年 FOMC 利率決策會議發布日
    datetime.date(2026, 1, 28): "⚡ FOMC 利率決策公布",
    datetime.date(2026, 3, 18): "⚡ FOMC 利率決策與經濟展望",
    datetime.date(2026, 5, 6): "⚡ FOMC 利率決策公布",
    datetime.date(2026, 6, 17): "⚡ FOMC 利率決策與點陣圖",
    datetime.date(2026, 7, 29): "⚡ FOMC 利率決策公布",
    datetime.date(2026, 9, 16): "⚡ FOMC 利率決策與點陣圖",
    datetime.date(2026, 10, 28): "⚡ FOMC 利率決策公布",
    datetime.date(2026, 12, 9): "⚡ FOMC 利率決策與點陣圖",
    # 2026 年 美國 CPI 通膨報告發布日
    datetime.date(2026, 1, 14): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 2, 13): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 3, 11): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 4, 10): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 5, 12): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 6, 10): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 7, 14): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 8, 12): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 9, 11): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 10, 14): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 11, 12): "🚨 美國 CPI 通膨發布",
    datetime.date(2026, 12, 10): "🚨 美國 CPI 通膨發布",
    # 2026 年 美國非農就業報告 (NFP) 發布日
    datetime.date(2026, 1, 9): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 2, 6): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 3, 6): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 4, 3): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 5, 8): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 6, 5): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 7, 2): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 8, 7): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 9, 4): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 10, 2): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 11, 6): "💼 美國非農就業報告 (NFP)",
    datetime.date(2026, 12, 4): "💼 美國非農就業報告 (NFP)",
}



def check_is_taiwan_holiday(d: datetime.date) -> Tuple[bool, str]:
    if d.weekday() == 5:
        return True, "週六非交易日"
    if d.weekday() == 6:
        return True, "週日非交易日"
    if d in TW_HOLIDAYS_MAP:
        return True, f"台灣休假日（{TW_HOLIDAYS_MAP[d]}）"
    return False, ""


# 國安基金第9次與第10次完整進場護盤期間設定
NSF_ACTIVE_PERIODS = [
    # 第9次進場護盤期間：2025/04/09 ~ 2026/01/12
    (datetime.date(2025, 4, 9), datetime.date(2026, 1, 12)),
    # 第10次進場護盤期間：2026/11/09 ~ 持續護盤中 (預設至 2099/12/31)
    (datetime.date(2026, 11, 9), datetime.date(2099, 12, 31)),
]


def check_is_nsf_active(d: datetime.date) -> int:
    """
    判斷國安基金進場護盤期間虛擬變數 (Dummy Variable)：
    - 凡處於第9次及第10次國安基金進場護盤期間內，皆設為 1，未護盤期間設為 0：
      * 第9次護盤期間：2025/04/09 - 2026/01/12
      * 第10次護盤期間：2026/11/09 - 持續進行中
    """
    if isinstance(d, pd.Timestamp):
        d = d.date()
    elif isinstance(d, datetime.datetime):
        d = d.date()

    for start_d, end_d in NSF_ACTIVE_PERIODS:
        if start_d <= d <= end_d:
            return 1
    return 0


# -----------------------------------------------------------------------------
# 0. 多使用者權限與身分驗證層
# -----------------------------------------------------------------------------
def safe_get_secret(key: str, default=None):
    """安全讀取 st.secrets，避免無 .streamlit/secrets.toml 時拋出 StreamlitSecretNotFoundError"""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default


def check_password() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    default_users = {
        "admin": "taiex888",
        "analyst01": "quant2026",
        "manager": "fund999",
        "guest": "read123",
    }

    creds = safe_get_secret("credentials", None)
    if creds is not None and "users" in creds:
        valid_users = dict(creds["users"])
    else:
        valid_users = default_users

    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        st.markdown("### 🔒 TAIEX 量化監控終端使用者登入")
        st.caption("請輸入授權之帳號與密碼以解鎖即時模型與數據權限。")
        with st.form("login_form"):
            input_user = st.text_input("使用者帳號 (Username)").strip()
            input_pwd = st.text_input("存取密碼 (Password)", type="password").strip()
            submit = st.form_submit_button("登入系統", use_container_width=True)

            if submit:
                if input_user in valid_users and valid_users[input_user] == input_pwd:
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = input_user
                    st.success(f"身分驗證成功，歡迎 {input_user}！正在載入...")
                    st.rerun()
                else:
                    st.error("❌ 帳號不存在或密碼錯誤，請重新確認！")

    return False


if not check_password():
    st.stop()


# -----------------------------------------------------------------------------
# 1. 資料爬蟲管線（多軌備援、穿透補齊、零空值保證）
# -----------------------------------------------------------------------------
class MarketDataPipeline:

    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        self.taifex_parquet_path = "taifex_chip_520d.parquet"
        self.twse_parquet_path = "twse_chip_520d.parquet"
        self.txf_price_parquet_path = "txf_prices_520d.parquet"
        self.sbl_parquet_path = "twse_sbl_520d.parquet"

    def get_cache_file_info(self, file_path: str) -> Dict[str, str]:
        if not os.path.exists(file_path):
            return {
                "exists": False,
                "size_str": "未建立",
                "mtime_str": "無記錄",
                "records_str": "0 筆",
                "span_str": "-",
            }

        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024.0
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")

        try:
            df = pd.read_parquet(file_path)
            records_count = len(df)
            cols_count = len(df.columns)
            if not df.empty:
                span_str = f"{df.index.min().strftime('%Y/%m/%d')} ~ {df.index.max().strftime('%Y/%m/%d')}"
            else:
                span_str = "空資料表"
        except Exception:
            records_count = 0
            cols_count = 0
            span_str = "讀取異常"

        return {
            "exists": True,
            "size_str": f"{size_kb:.1f} KB",
            "mtime_str": mtime_str,
            "records_str": f"{records_count:,} 筆 ({cols_count} 欄)",
            "span_str": span_str,
        }

    def get_previous_weekday(self, target_date: datetime.date) -> datetime.date:
        d = target_date - datetime.timedelta(days=1)
        while d.weekday() >= 5:
            d -= datetime.timedelta(days=1)
        return d

    def fetch_twse_single_day(
        self, target_date: datetime.date
    ) -> Tuple[pd.DataFrame, bool, str]:
        date_str = target_date.strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/zh/fund/BFI82U?response=json&dayDate={date_str}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            data = resp.json()
            if (
                data.get("stat") != "OK"
                or "data" not in data
                or len(data["data"]) == 0
            ):
                return pd.DataFrame(), False, f"查無資料或休市 ({date_str})"

            df = pd.DataFrame(
                data["data"], columns=["item", "buy", "sell", "net"]
            )
            for c in ["buy", "sell", "net"]:
                df[c] = (
                    df[c]
                    .astype(str)
                    .str.replace(",", "")
                    .astype(float)
                    / 100_000_000.0
                )

            mapping = {
                "外資及陸資(不含外資自營商)": "foreign_net",
                "外資及陸資": "foreign_net",
                "外資": "foreign_net",
                "投信": "trust_net",
                "自營商(合計)": "dealer_net",
            }
            df["key"] = df["item"].map(mapping)
            feat_df = df.dropna(subset=["key"])[["key", "net"]]
            feat_dict = dict(zip(feat_df["key"], feat_df["net"]))
            feat_dict["date"] = pd.to_datetime(target_date)
            return pd.DataFrame([feat_dict]).set_index("date"), True, "正常取得"
        except Exception as e:
            return pd.DataFrame(), False, f"擷取異常: {str(e)}"

    def fetch_twse_institutional_with_fallback(
        self, start_date: datetime.date, max_lookback_days: int = 10
    ) -> Tuple[pd.DataFrame, datetime.date, str]:
        curr = start_date
        while (start_date - curr).days <= max_lookback_days:
            if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                df, ok, msg = self.fetch_twse_single_day(curr)
                if ok and not df.empty:
                    note = (
                        "正常取得"
                        if curr == start_date
                        else f"原日前推至有效交易日 ({curr.strftime('%Y-%m-%d')})"
                    )
                    return df, curr, note
            curr -= datetime.timedelta(days=1)
        return pd.DataFrame(), start_date, "超過回溯天數，查無有效數據"

    def fetch_taifex_single_day(
        self, target_date: datetime.date
    ) -> Tuple[pd.DataFrame, bool, str]:
        date_str = target_date.strftime("%Y/%m/%d")
        payload = {
            "queryStartDate": date_str,
            "queryEndDate": date_str,
            "commodityId": "TXF",
        }

        csv_url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
        try:
            resp = requests.post(
                csv_url, data=payload, headers=self.headers, timeout=10
            )
            resp.encoding = "big5"
            content = resp.text

            if (
                "查無資料" not in content
                and resp.status_code == 200
                and len(content.strip()) > 0
            ):
                df = pd.read_csv(io.StringIO(content))
                df.columns = [
                    str(c).strip().replace(" ", "").replace("\ufeff", "")
                    for c in df.columns
                ]

                name_cols = [c for c in df.columns if "商品名稱" in c]
                id_cols = [c for c in df.columns if "身份" in c or "身分" in c]

                if name_cols and id_cols:
                    sub_df = df[
                        df[name_cols[0]]
                        .astype(str)
                        .str.contains("TX|臺股期貨|臺指期", regex=True)
                    ].copy()
                    if not sub_df.empty:
                        res = {"date": pd.to_datetime(target_date)}
                        for _, r in sub_df.iterrows():
                            ident = str(r[id_cols[0]]).strip()
                            target_key = None
                            if "外資" in ident:
                                target_key = "tx_foreign_oi"
                            elif "投信" in ident:
                                target_key = "tx_trust_oi"
                            elif "自營商" in ident:
                                target_key = "tx_dealer_oi"

                            if target_key:
                                oi_val = None
                                for col_name in r.index:
                                    if "多空淨額" in col_name and (
                                        "口數" in col_name or "未平倉" in col_name
                                    ):
                                        try:
                                            raw_str = (
                                                str(r[col_name])
                                                .replace(",", "")
                                                .replace(" ", "")
                                                .replace("--", "0")
                                            )
                                            oi_val = float(raw_str)
                                            break
                                        except Exception:
                                            pass

                                if oi_val is None:
                                    try:
                                        raw_val = (
                                            str(r.iloc[-2])
                                            .replace(",", "")
                                            .replace(" ", "")
                                            .replace("--", "0")
                                        )
                                        oi_val = float(raw_val)
                                    except Exception:
                                        pass

                                if oi_val is not None:
                                    res[target_key] = oi_val

                        if "tx_foreign_oi" in res:
                            return (
                                pd.DataFrame([res]).set_index("date"),
                                True,
                                "正常取得",
                            )
        except Exception:
            pass

        return pd.DataFrame(), False, f"查無資料或休市 ({date_str})"

    def fetch_taifex_oi_with_fallback(
        self, start_date: datetime.date, max_lookback_days: int = 10
    ) -> Tuple[pd.DataFrame, datetime.date, str]:
        curr = start_date
        while (start_date - curr).days <= max_lookback_days:
            if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                df, ok, msg = self.fetch_taifex_single_day(curr)
                if ok and not df.empty:
                    note = (
                        "正常取得"
                        if curr == start_date
                        else f"原日前推至有效交易日 ({curr.strftime('%Y-%m-%d')})"
                    )
                    return df, curr, note
            curr -= datetime.timedelta(days=1)
        return pd.DataFrame(), start_date, "超過回溯天數，查無有效數據"

    # -------------------------------------------------------------------------
    # 批次取得真實三大法人籌碼（FinMind API 批次版 + 本地 Parquet 差集增量與橫向擴增快取）
    # -------------------------------------------------------------------------
    def fetch_finmind_chips_history(
        self, start_d: datetime.date, end_d: datetime.date, token: str = ""
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        api_token = token
        if not api_token:
            api_token = safe_get_secret("FINMIND_TOKEN", "")

        url = "https://api.finmindtrade.com/api/v4/data"
        s_date = start_d.strftime("%Y-%m-%d")
        e_date = end_d.strftime("%Y-%m-%d")

        # 1. 抓取現貨三大法人買賣超 (橫向擴增：外資、投信、自營商)
        twse_df = pd.DataFrame()
        try:
            params = {
                "dataset": "TaiwanStockTotalInstitutionalInvestors",
                "start_date": s_date,
                "end_date": e_date,
            }
            if api_token:
                params["token"] = api_token
            resp = requests.get(url, params=params, headers=self.headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    raw_df = pd.DataFrame(data)
                    name_col = next(
                        (c for c in ["name", "institutional_investors", "identity"] if c in raw_df.columns),
                        None,
                    )
                    buy_c = next((c for c in raw_df.columns if "buy" in str(c).lower()), None)
                    sell_c = next((c for c in raw_df.columns if "sell" in str(c).lower()), None)

                    if name_col and buy_c and sell_c:
                        raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.normalize()
                        b_val = pd.to_numeric(raw_df[buy_c], errors="coerce").fillna(0.0)
                        s_val = pd.to_numeric(raw_df[sell_c], errors="coerce").fillna(0.0)
                        raw_df["net_yi"] = (b_val - s_val) / 100_000_000.0

                        def map_twse_role(val: str) -> Optional[str]:
                            v = str(val).strip()
                            v_low = v.lower()
                            if "外資" in v or "foreign" in v_low:
                                if "自營" in v or "dealer" in v_low:
                                    return None
                                return "foreign_net"
                            if "投信" in v or "trust" in v_low:
                                return "trust_net"
                            if "自營" in v or "dealer" in v_low:
                                return "dealer_net"
                            return None

                        raw_df["role"] = raw_df[name_col].apply(map_twse_role)
                        sub_df = raw_df.dropna(subset=["role"])
                        if not sub_df.empty:
                            piv = sub_df.pivot_table(
                                index="date", columns="role", values="net_yi", aggfunc="sum"
                            )
                            for req_c in ["foreign_net", "trust_net", "dealer_net"]:
                                if req_c not in piv.columns:
                                    piv[req_c] = np.nan
                            twse_df = piv[["foreign_net", "trust_net", "dealer_net"]].sort_index()
                            twse_df = twse_df[~twse_df.index.duplicated(keep="last")]
        except Exception as e:
            st.warning(f"FinMind 現貨籌碼取得異常: {e}")

        # 2. 抓取臺指期三大法人留倉淨額 (橫向擴增：外資、投信、自營商未平倉口數)
        taifex_df = pd.DataFrame()
        try:
            params = {
                "dataset": "TaiwanFuturesInstitutionalInvestors",
                "data_id": "TX",
                "start_date": s_date,
                "end_date": e_date,
            }
            if api_token:
                params["token"] = api_token
            resp = requests.get(url, params=params, headers=self.headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    raw_df = pd.DataFrame(data)
                    name_col = next(
                        (c for c in ["institutional_investors", "name", "identity", "身份", "身分"] if c in raw_df.columns),
                        None,
                    )
                    if name_col:
                        raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.normalize()
                        long_col = None
                        short_col = None
                        net_oi_col = None

                        for c in raw_df.columns:
                            c_str = str(c).lower()
                            if "amount" in c_str or "金額" in c_str:
                                continue
                            if any(k in c_str for k in ["net_open_interest", "oi_net", "多空淨額口數", "未平倉淨額"]):
                                net_oi_col = c
                            elif any(k in c_str for k in ["long_open_interest", "open_interest_buy", "buy_open_interest", "多方未平倉"]):
                                long_col = c
                            elif any(k in c_str for k in ["short_open_interest", "open_interest_sell", "sell_open_interest", "空方未平倉"]):
                                short_col = c

                        if net_oi_col:
                            raw_df["oi_val"] = pd.to_numeric(raw_df[net_oi_col], errors="coerce").fillna(0.0)
                        elif long_col and short_col:
                            l_oi = pd.to_numeric(raw_df[long_col], errors="coerce").fillna(0.0)
                            s_oi = pd.to_numeric(raw_df[short_col], errors="coerce").fillna(0.0)
                            raw_df["oi_val"] = l_oi - s_oi
                        else:
                            raw_df["oi_val"] = 0.0

                        def map_taifex_role(val: str) -> Optional[str]:
                            v = str(val).strip()
                            v_low = v.lower()
                            if "外資" in v or "foreign" in v_low:
                                return "tx_foreign_oi"
                            if "投信" in v or "trust" in v_low:
                                return "tx_trust_oi"
                            if "自營" in v or "dealer" in v_low:
                                return "tx_dealer_oi"
                            return None

                        raw_df["role"] = raw_df[name_col].apply(map_taifex_role)
                        sub_fut = raw_df.dropna(subset=["role"])
                        if not sub_fut.empty:
                            piv_fut = sub_fut.pivot_table(
                                index="date", columns="role", values="oi_val", aggfunc="sum"
                            )
                            for req_c in ["tx_foreign_oi", "tx_trust_oi", "tx_dealer_oi"]:
                                if req_c not in piv_fut.columns:
                                    piv_fut[req_c] = np.nan
                            taifex_df = piv_fut[["tx_foreign_oi", "tx_trust_oi", "tx_dealer_oi"]].sort_index()
                            taifex_df = taifex_df[~taifex_df.index.duplicated(keep="last")]
        except Exception as e:
            st.warning(f"FinMind 期貨籌碼取得異常: {e}")

        return twse_df, taifex_df

    def fetch_twse_historical_range_parquet(
        self, start_d: datetime.date, end_d: datetime.date, target_trading_dates: Optional[List[pd.Timestamp]] = None
    ) -> pd.DataFrame:
        disk_df = pd.DataFrame()
        if os.path.exists(self.twse_parquet_path):
            try:
                disk_df = pd.read_parquet(self.twse_parquet_path)
                if not disk_df.empty:
                    disk_df.index = pd.to_datetime(disk_df.index).normalize()
            except Exception:
                disk_df = pd.DataFrame()

        required_cols = {"foreign_net", "trust_net", "dealer_net"}
        has_all_cols = required_cols.issubset(set(disk_df.columns)) if not disk_df.empty else False
        existing_dates = set(disk_df.index.date) if (not disk_df.empty and has_all_cols) else set()

        if target_trading_dates is not None and len(target_trading_dates) > 0:
            target_dates_clean = [pd.to_datetime(d).date() for d in target_trading_dates]
            dates_to_fetch = [d for d in target_dates_clean if d not in existing_dates and start_d <= d <= end_d]
        else:
            dates_to_fetch = []
            curr = start_d
            while curr <= end_d:
                if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                    if curr not in existing_dates:
                        dates_to_fetch.append(curr)
                curr += datetime.timedelta(days=1)

        if len(dates_to_fetch) == 0 and not disk_df.empty:
            return disk_df[(disk_df.index.date >= start_d) & (disk_df.index.date <= end_d)].copy()

        # 若缺失天數較多 (> 5 日) 或欄位擴增升級，優先以 FinMind 批次補齊以加速快取擴增
        if len(dates_to_fetch) > 5:
            fetch_s = min(dates_to_fetch)
            fetch_e = max(dates_to_fetch)
            api_twse, _ = self.fetch_finmind_chips_history(fetch_s, fetch_e)
            if not api_twse.empty:
                api_twse.index = pd.to_datetime(api_twse.index).normalize()
                disk_df = api_twse.combine_first(disk_df) if not disk_df.empty else api_twse
                disk_df = disk_df[~disk_df.index.duplicated(keep="last")].sort_index()
                try:
                    disk_df.to_parquet(self.twse_parquet_path)
                except Exception:
                    pass
                has_all_cols = required_cols.issubset(set(disk_df.columns))
                existing_dates = set(disk_df.index.date) if has_all_cols else set()
                dates_to_fetch = [d for d in dates_to_fetch if d not in existing_dates]

        if len(dates_to_fetch) == 0 and not disk_df.empty:
            return disk_df[(disk_df.index.date >= start_d) & (disk_df.index.date <= end_d)].copy()

        # 避免當 FinMind 未回傳早期資料時逐日爬抓數百天造成 Streamlit 連線逾時，單日爬蟲僅補齊最近 15 個營業日
        if len(dates_to_fetch) > 15:
            dates_to_fetch = sorted(dates_to_fetch)[-15:]

        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.twse.com.tw/zh/trading/foreign/bfi82u.html",
            "X-Requested-With": "XMLHttpRequest",
        })

        new_records = []
        progress_text = st.empty()
        total_fetch = len(dates_to_fetch)

        for idx, target_date in enumerate(dates_to_fetch):
            date_str = target_date.strftime("%Y%m%d")
            url = f"https://www.twse.com.tw/rwd/zh/fund/BFI82U?response=json&dayDate={date_str}"

            if total_fetch > 1:
                progress_text.caption(f"⏳ 正在精準增量補齊證交所現貨真實籌碼 ({idx + 1}/{total_fetch} 營業日: {date_str})...")

            for attempt in range(4):
                try:
                    resp = session.get(url, timeout=12)
                    if resp.status_code == 200:
                        data = resp.json()
                        stat_msg = str(data.get("stat", ""))

                        if stat_msg == "OK" and "data" in data and len(data["data"]) > 0:
                            df = pd.DataFrame(data["data"], columns=["item", "buy", "sell", "net"])
                            for c in ["buy", "sell", "net"]:
                                df[c] = (
                                    df[c].astype(str).str.replace(",", "").astype(float) / 100_000_000.0
                                )
                            mapping = {
                                "外資及陸資(不含外資自營商)": "foreign_net",
                                "外資及陸資": "foreign_net",
                                "外資": "foreign_net",
                                "投信": "trust_net",
                                "自營商(合計)": "dealer_net",
                            }
                            df["key"] = df["item"].map(mapping)
                            feat_df = df.dropna(subset=["key"])[["key", "net"]]
                            feat_dict = dict(zip(feat_df["key"], feat_df["net"]))
                            feat_dict["date"] = pd.to_datetime(target_date).normalize()

                            day_df = pd.DataFrame([feat_dict]).set_index("date")
                            new_records.append(day_df)
                            break

                        elif any(k in stat_msg for k in ["頻繁", "稍候再試", "過於頻繁"]):
                            time.sleep(3.0 * (attempt + 1))
                            continue

                        elif any(k in stat_msg for k in ["很抱歉", "查無資料", "沒有符合"]):
                            break

                    elif resp.status_code == 429:
                        time.sleep(4.0 * (attempt + 1))
                        continue

                    time.sleep(1.0 * (attempt + 1))
                except Exception:
                    time.sleep(1.5 * (attempt + 1))

            time.sleep(0.8)

        progress_text.empty()

        if new_records:
            appended = pd.concat(new_records)
            full_df = appended.combine_first(disk_df) if not disk_df.empty else appended
            full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()
            try:
                full_df.to_parquet(self.twse_parquet_path)
            except Exception:
                pass
            return full_df[(full_df.index.date >= start_d) & (full_df.index.date <= end_d)].copy()

        return (
            disk_df[(disk_df.index.date >= start_d) & (disk_df.index.date <= end_d)].copy()
            if not disk_df.empty
            else pd.DataFrame()
        )

    def fetch_sbl_single_day(
        self, target_date: datetime.date, session: Optional[requests.Session] = None
    ) -> Tuple[pd.DataFrame, bool, str]:
        """擷取證交所 TWT93U 單日整體市場借券賣出當日餘額（轉換為張數 = 股數 / 1000）"""
        date_str = target_date.strftime("%Y%m%d")
        urls = [
            f"https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U?response=json&date={date_str}",
            f"https://www.twse.com.tw/exchangeReport/TWT93U?response=json&date={date_str}",
        ]
        http_client = session if session is not None else requests
        for url in urls:
            for attempt in range(3):
                try:
                    resp = http_client.get(url, headers=self.headers, timeout=12)
                    if resp.status_code == 429:
                        time.sleep(2.5 * (attempt + 1))
                        continue
                    if resp.status_code != 200:
                        break
                    data = resp.json()
                    stat_msg = str(data.get("stat", ""))
                    if any(k in stat_msg for k in ["頻繁", "稍候再試", "過於頻繁"]):
                        time.sleep(2.5 * (attempt + 1))
                        continue
                    if (
                        stat_msg != "OK"
                        or "data" not in data
                        or len(data["data"]) == 0
                    ):
                        break

                    rows = data["data"]
                    fields = data.get("fields", [])

                    # 定位「借券賣出－當日餘額」欄位索引（TWT93U 標準第 12 欄；亦依欄位名稱動態比對）
                    bal_col_idx = 12
                    if fields:
                        bal_candidates = [
                            i for i, fld in enumerate(fields)
                            if ("餘額" in str(fld)) and ("前日" not in str(fld))
                        ]
                        if len(bal_candidates) >= 2:
                            bal_col_idx = bal_candidates[-1]
                        elif len(bal_candidates) == 1 and bal_candidates[0] >= 8:
                            bal_col_idx = bal_candidates[0]

                    total_shares = None
                    for row in reversed(rows):
                        if len(row) > bal_col_idx and "合計" in str(row[0]):
                            raw_str = str(row[bal_col_idx]).replace(",", "").strip()
                            if raw_str not in ("", "-", "--"):
                                total_shares = float(raw_str)
                                break

                    if total_shares is None:
                        acc_shares = 0.0
                        valid_cnt = 0
                        for row in rows:
                            if len(row) > bal_col_idx and "合計" not in str(row[0]):
                                raw_str = str(row[bal_col_idx]).replace(",", "").strip()
                                try:
                                    acc_shares += float(raw_str)
                                    valid_cnt += 1
                                except Exception:
                                    pass
                        if valid_cnt > 0:
                            total_shares = acc_shares

                    if total_shares is not None:
                        total_lots = total_shares / 1000.0
                        df = pd.DataFrame(
                            [{"date": pd.to_datetime(target_date).normalize(), "sbl_balance": total_lots}]
                        ).set_index("date")
                        return df, True, "正常取得"
                    break
                except Exception:
                    time.sleep(1.0 * (attempt + 1))
                    continue

        return pd.DataFrame(), False, f"查無借券賣出餘額資料或休市 ({date_str})"

    def fetch_sbl_with_fallback(
        self, start_date: datetime.date, max_lookback_days: int = 10
    ) -> Tuple[pd.DataFrame, datetime.date, str]:
        curr = start_date
        while (start_date - curr).days <= max_lookback_days:
            if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                df, ok, msg = self.fetch_sbl_single_day(curr)
                if ok and not df.empty:
                    note = (
                        "正常取得"
                        if curr == start_date
                        else f"原日前推至有效交易日 ({curr.strftime('%Y-%m-%d')})"
                    )
                    return df, curr, note
            curr -= datetime.timedelta(days=1)
        return pd.DataFrame(), start_date, "超過回溯天數，查無有效借券數據"

    def fetch_sbl_historical_range_parquet(
        self, start_d: datetime.date, end_d: datetime.date, target_trading_dates: Optional[List[pd.Timestamp]] = None
    ) -> pd.DataFrame:
        """
        完整擷取整體市場借券賣出餘額 (張) 並永久增量快取至 twse_sbl_520d.parquet：
        1. 第一軌道（快速探測）：若具備 FinMind 付費 Sponsor/Backer 權限，以單日全市場查詢補齊；
           若為免費帳號（回傳 HTTP 400/402 權限不足），立刻熔斷停止無效請求，不浪費時間。
        2. 第二軌道（證交所官方 TWT93U 全量補齊 + 斷點續傳）：
           移除原本僅抓最新 3 日之限制，透過 HTTP Keep-Alive Session 逐日補齊所有缺失交易日
           （並額外往前多補 6 個交易日供期初計算近 5 日變動量），每 20 筆即時寫入 Parquet 檔，
           確保首次建檔穩定可靠、日後每日更新僅需 < 1 秒。
        """
        disk_df = pd.DataFrame()
        if os.path.exists(self.sbl_parquet_path):
            try:
                disk_df = pd.read_parquet(self.sbl_parquet_path)
                if not disk_df.empty:
                    disk_df.index = pd.to_datetime(disk_df.index).normalize()
            except Exception:
                disk_df = pd.DataFrame()

        has_col = (not disk_df.empty) and ("sbl_balance" in disk_df.columns)
        existing_dates = set(disk_df.index.date) if has_col else set()

        if target_trading_dates is not None and len(target_trading_dates) > 0:
            target_dates_clean = sorted([pd.to_datetime(d).date() for d in target_trading_dates])
            # 額外往前補 6 個營業日，確保 520 筆樣本最前端也能計算出真實的「近 5 日累計變動量」
            extra_lead_dates = []
            lead_curr = target_dates_clean[0] - datetime.timedelta(days=1)
            while len(extra_lead_dates) < 6 and (target_dates_clean[0] - lead_curr).days <= 25:
                if lead_curr.weekday() < 5 and lead_curr not in TW_HOLIDAYS_MAP:
                    extra_lead_dates.append(lead_curr)
                lead_curr -= datetime.timedelta(days=1)
            all_needed = sorted(set(extra_lead_dates + target_dates_clean))
            dates_to_fetch = [d for d in all_needed if d not in existing_dates and d <= end_d]
        else:
            all_expected = []
            curr = start_d
            while curr <= end_d:
                if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                    all_expected.append(curr)
                curr += datetime.timedelta(days=1)
            dates_to_fetch = [d for d in all_expected if d not in existing_dates]

        # 排除當日尚未盤後公告之日期（證交所 TWT93U 每日約 21:00 後正式公告當日借券賣出餘額）
        now_dt = datetime.datetime.now()
        dates_to_fetch = [
            d for d in dates_to_fetch
            if not (d == now_dt.date() and now_dt.time() < datetime.time(21, 0))
        ]

        effective_start = min(start_d, min(dates_to_fetch)) if dates_to_fetch else start_d

        if len(dates_to_fetch) == 0 and has_col:
            return disk_df[(disk_df.index.date >= effective_start) & (disk_df.index.date <= end_d)].copy()

        # =====================================================================
        # 第一軌道：FinMind TaiwanDailyShortSaleBalances 單日全市場權限探測（免費帳號自動熔斷）
        # =====================================================================
        api_token = safe_get_secret("FINMIND_TOKEN", "")
        progress_text = st.empty()
        progress_bar = st.empty()
        if api_token and len(dates_to_fetch) > 3:
            try:
                fm_url = "https://api.finmindtrade.com/api/v4/data"
                fm_session = requests.Session()
                probe_date = sorted(dates_to_fetch)[-1]
                probe_params = {
                    "dataset": "TaiwanDailyShortSaleBalances",
                    "start_date": probe_date.strftime("%Y-%m-%d"),
                    "token": api_token,
                }
                r_probe = fm_session.get(fm_url, params=probe_params, headers=self.headers, timeout=12)
                fm_allowed = False
                if r_probe.status_code == 200:
                    probe_json = r_probe.json()
                    probe_rows = probe_json.get("data", [])
                    if probe_rows:
                        fm_allowed = True
                        tmp_p = pd.DataFrame(probe_rows)
                        if "date" in tmp_p.columns and "SBLShortSalesCurrentDayBalance" in tmp_p.columns:
                            tmp_p["date"] = pd.to_datetime(tmp_p["date"]).dt.normalize()
                            tmp_p["sbl_bal"] = pd.to_numeric(
                                tmp_p["SBLShortSalesCurrentDayBalance"], errors="coerce"
                            ).fillna(0.0)
                            p_df = (tmp_p.groupby("date")["sbl_bal"].sum() / 1000.0).to_frame("sbl_balance")
                            disk_df = p_df.combine_first(disk_df) if not disk_df.empty else p_df
                            disk_df = disk_df[~disk_df.index.duplicated(keep="last")].sort_index()
                            existing_dates = set(disk_df.index.date)
                            dates_to_fetch = [d for d in dates_to_fetch if d not in existing_dates]

                if fm_allowed and len(dates_to_fetch) > 0:
                    fm_records = []
                    total_fm = len(dates_to_fetch)
                    for idx, target_date in enumerate(reversed(sorted(dates_to_fetch))):
                        progress_text.caption(
                            f"⏳ 正在自 FinMind 補齊整體借券賣出餘額 ({idx + 1}/{total_fm}: {target_date})..."
                        )
                        progress_bar.progress((idx + 1) / total_fm)
                        params = {
                            "dataset": "TaiwanDailyShortSaleBalances",
                            "start_date": target_date.strftime("%Y-%m-%d"),
                            "token": api_token,
                        }
                        r_fm = fm_session.get(fm_url, params=params, headers=self.headers, timeout=12)
                        if r_fm.status_code in (400, 402):
                            break
                        if r_fm.status_code == 200:
                            rows_fm = r_fm.json().get("data", [])
                            if rows_fm:
                                tmp_fm = pd.DataFrame(rows_fm)
                                if "date" in tmp_fm.columns and "SBLShortSalesCurrentDayBalance" in tmp_fm.columns:
                                    tmp_fm["date"] = pd.to_datetime(tmp_fm["date"]).dt.normalize()
                                    tmp_fm["sbl_bal"] = pd.to_numeric(
                                        tmp_fm["SBLShortSalesCurrentDayBalance"], errors="coerce"
                                    ).fillna(0.0)
                                    daily_sum = tmp_fm.groupby("date")["sbl_bal"].sum() / 1000.0
                                    fm_records.append(daily_sum.to_frame("sbl_balance"))
                        if len(fm_records) >= 20:
                            appended_fm = pd.concat(fm_records)
                            disk_df = appended_fm.combine_first(disk_df) if not disk_df.empty else appended_fm
                            disk_df = disk_df[~disk_df.index.duplicated(keep="last")].sort_index()
                            try:
                                disk_df.to_parquet(self.sbl_parquet_path)
                            except Exception:
                                pass
                            fm_records = []

                    if fm_records:
                        appended_fm = pd.concat(fm_records)
                        disk_df = appended_fm.combine_first(disk_df) if not disk_df.empty else appended_fm
                        disk_df = disk_df[~disk_df.index.duplicated(keep="last")].sort_index()
                        try:
                            disk_df.to_parquet(self.sbl_parquet_path)
                        except Exception:
                            pass

                    existing_dates = set(disk_df.index.date) if not disk_df.empty else set()
                    dates_to_fetch = [d for d in dates_to_fetch if d not in existing_dates]
            except Exception:
                pass
            finally:
                progress_text.empty()
                progress_bar.empty()

        # =====================================================================
        # 第二軌道（主力官方來源）：證交所 TWT93U 全量補齊 + 每 20 筆斷點存檔
        # =====================================================================
        if len(dates_to_fetch) > 0:
            twse_dates = sorted(dates_to_fetch, reverse=True)
            total_twse = len(twse_dates)
            twse_session = requests.Session()
            twse_session.headers.update({
                "User-Agent": self.headers["User-Agent"],
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.twse.com.tw/zh/trading/margin/twt93u.html",
            })

            batch_records = []
            try:
                for idx, target_date in enumerate(twse_dates):
                    date_str = target_date.strftime("%Y%m%d")
                    if total_twse > 1:
                        progress_text.caption(
                            f"⏳ 正在自證交所 TWT93U 補齊整體借券賣出餘額 ({idx + 1}/{total_twse} 營業日: {date_str}，每 20 筆自動存檔)..."
                        )
                        progress_bar.progress((idx + 1) / total_twse)

                    day_df, ok, _ = self.fetch_sbl_single_day(target_date, session=twse_session)
                    if ok and not day_df.empty:
                        batch_records.append(day_df)

                    # 每滿 20 筆立即寫入磁碟 Parquet，確保斷點續傳不遺失進度
                    if len(batch_records) >= 20:
                        appended = pd.concat(batch_records)
                        disk_df = appended.combine_first(disk_df) if not disk_df.empty else appended
                        disk_df = disk_df[~disk_df.index.duplicated(keep="last")].sort_index()
                        try:
                            disk_df.to_parquet(self.sbl_parquet_path)
                        except Exception:
                            pass
                        batch_records = []

                    # 禮貌性節流避免觸發證交所限速
                    time.sleep(0.28 if total_twse > 5 else 0.15)

                if batch_records:
                    appended = pd.concat(batch_records)
                    disk_df = appended.combine_first(disk_df) if not disk_df.empty else appended
                    disk_df = disk_df[~disk_df.index.duplicated(keep="last")].sort_index()
                    try:
                        disk_df.to_parquet(self.sbl_parquet_path)
                    except Exception:
                        pass
            finally:
                progress_text.empty()
                progress_bar.empty()

        return (
            disk_df[(disk_df.index.date >= effective_start) & (disk_df.index.date <= end_d)].copy()
            if not disk_df.empty
            else pd.DataFrame()
        )

    def fetch_taifex_historical_range_parquet(
        self, start_d: datetime.date, end_d: datetime.date, target_trading_dates: Optional[List[pd.Timestamp]] = None
    ) -> pd.DataFrame:
        disk_df = pd.DataFrame()
        if os.path.exists(self.taifex_parquet_path):
            try:
                disk_df = pd.read_parquet(self.taifex_parquet_path)
                if not disk_df.empty:
                    disk_df.index = pd.to_datetime(disk_df.index).normalize()
            except Exception:
                disk_df = pd.DataFrame()

        # 檢查橫向擴增欄位與縱向交易日差集
        required_cols = {"tx_foreign_oi", "tx_trust_oi", "tx_dealer_oi"}
        has_all_cols = required_cols.issubset(set(disk_df.columns)) if not disk_df.empty else False
        existing_dates = set(disk_df.index.date) if (not disk_df.empty and has_all_cols) else set()

        if target_trading_dates is not None and len(target_trading_dates) > 0:
            target_dates_clean = [pd.to_datetime(d).date() for d in target_trading_dates]
            missing_dates = [d for d in target_dates_clean if d not in existing_dates and start_d <= d <= end_d]
        else:
            missing_dates = []
            curr = start_d
            while curr <= end_d:
                if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                    if curr not in existing_dates:
                        missing_dates.append(curr)
                curr += datetime.timedelta(days=1)

        if len(missing_dates) > 0:
            fetch_s = min(missing_dates) if has_all_cols else start_d
            fetch_e = max(missing_dates) if has_all_cols else end_d
            try:
                res = self.fetch_finmind_chips_history(fetch_s, fetch_e)
                if isinstance(res, tuple) and len(res) == 2:
                    _, api_taifex = res
                elif isinstance(res, pd.DataFrame):
                    api_taifex = res
                else:
                    api_taifex = pd.DataFrame()

                if not api_taifex.empty:
                    api_taifex.index = pd.to_datetime(api_taifex.index).normalize()
                    disk_df = api_taifex.combine_first(disk_df) if not disk_df.empty else api_taifex
                    disk_df = disk_df[~disk_df.index.duplicated(keep="last")].sort_index()
            except Exception as e:
                st.warning(f"擷取期指未平倉數據時發生異常: {e}")

            # 若仍有少數近期營業日未在 FinMind 更新，以期交所官方單日爬蟲差集補齊
            still_existing = set(disk_df.index.date) if not disk_df.empty else set()
            still_missing = [d for d in missing_dates if d not in still_existing]
            if 0 < len(still_missing) <= 15:
                single_records = []
                for d_miss in still_missing:
                    day_fut_df, ok, _ = self.fetch_taifex_single_day(d_miss)
                    if ok and not day_fut_df.empty:
                        day_fut_df.index = pd.to_datetime(day_fut_df.index).normalize()
                        single_records.append(day_fut_df)
                if single_records:
                    appended_fut = pd.concat(single_records)
                    disk_df = appended_fut.combine_first(disk_df) if not disk_df.empty else appended_fut
                    disk_df = disk_df[~disk_df.index.duplicated(keep="last")].sort_index()

            if not disk_df.empty:
                try:
                    disk_df.to_parquet(self.taifex_parquet_path)
                except Exception:
                    pass

        if not disk_df.empty:
            sub_df = disk_df[(disk_df.index.date >= start_d) & (disk_df.index.date <= end_d)].copy()
            if target_trading_dates is not None and len(target_trading_dates) > 0:
                sub_df = sub_df.reindex(target_trading_dates)
            return sub_df

        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # 批次取得近月臺指期真實開、高、低、收、成交量與結算價
    # 【全面極速優化版】：
    # 1. 優先檢查本地 Parquet 快取完整性；若快取已涵蓋至前一有效交易日，直接秒級載入，完全不連網。
    # 2. 僅針對「快取真正缺失的交易日」進行精準增量補齊（Incremental Fetch），杜絕整批 520 天重複下載。
    # 3. 具備快取健康度檢驗（TXF_Close != TXF_Settle 比例防污染檢驗）。
    # -------------------------------------------------------------------------
    def fetch_txf_prices_historical_parquet(
        self, start_d: datetime.date, end_d: datetime.date, tw_raw: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        disk_df = pd.DataFrame()
        if os.path.exists(self.txf_price_parquet_path):
            try:
                disk_df = pd.read_parquet(self.txf_price_parquet_path)
                if not disk_df.empty:
                    disk_df.index = pd.to_datetime(disk_df.index).normalize()
            except Exception:
                disk_df = pd.DataFrame()

        # 【全面快取健康度檢查】：
        # 近月臺指期每日 13:45 收盤價 (TXF_Close) 與結算價 (TXF_Settle) 絕大多數交易日不相等。
        # 若收盤價等於結算價比例過高，判定遭污染並作廢
        if not disk_df.empty:
            if "TXF_Settle" not in disk_df.columns or "TXF_Close" not in disk_df.columns:
                disk_df = pd.DataFrame()
            else:
                valid_sub = disk_df.dropna(subset=["TXF_Close", "TXF_Settle"])
                if len(valid_sub) > 10:
                    diff_ratio = (valid_sub["TXF_Close"] != valid_sub["TXF_Settle"]).mean()
                    if diff_ratio < 0.20:
                        disk_df = pd.DataFrame()
                        try:
                            os.remove(self.txf_price_parquet_path)
                        except Exception:
                            pass

        required_cols = {
            "TXF_Open", "TXF_High", "TXF_Low", "TXF_Close", "TXF_Volume", "TXF_Settle",
            "TXF_AH_Open", "TXF_AH_High", "TXF_AH_Low", "TXF_AH_Close", "TXF_AH_Volume", "txf_ah_ret",
        }
        has_all_cols = required_cols.issubset(set(disk_df.columns)) if not disk_df.empty else False
        cached_dates = set(disk_df.index.date) if (not disk_df.empty and has_all_cols) else set()

        # 決定預期需要包含的交易日
        if tw_raw is not None and not tw_raw.empty:
            required_dates = [pd.to_datetime(d).date() for d in tw_raw.index if start_d <= pd.to_datetime(d).date() <= end_d]
        else:
            required_dates = []
            curr = start_d
            while curr <= end_d:
                if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                    required_dates.append(curr)
                curr += datetime.timedelta(days=1)

        # 排除「目標當日 (end_d) 且目前時間尚在盤中/開盤前」尚未有官方日盤結算的日期，避免永遠觸發缺少當日資料而重複爬取
        now_dt = datetime.datetime.now()
        missing_dates = []
        for d in required_dates:
            if d not in cached_dates:
                # 若缺失日期為今天，且現在尚未過 14:15（期交所一般日盤尚未結算產出），不視為缺失
                if d == now_dt.date() and now_dt.time() < datetime.time(14, 15):
                    continue
                missing_dates.append(d)

        # 【極速放行檢查】：若快取已有資料，且所有歷史交易日皆已收錄（或僅缺尚未收盤的今日），直接秒級回傳快取！
        if len(missing_dates) == 0 and not disk_df.empty:
            return disk_df[(disk_df.index.date >= start_d) & (disk_df.index.date <= end_d)].copy()

        # 若快取已有充足歷史（最新快取日距離 end_d 在 3 天內），且缺失僅為最近 1~2 天，僅增量抓取最近幾天
        if not disk_df.empty and has_all_cols:
            latest_cached = disk_df.index.max().date()
            if latest_cached >= start_d and (end_d - latest_cached).days <= 4:
                # 僅需補足最新快取日之後的營業日
                missing_dates = [d for d in missing_dates if d > latest_cached]
                if len(missing_dates) == 0:
                    return disk_df[(disk_df.index.date >= start_d) & (disk_df.index.date <= end_d)].copy()

        fetch_s = min(missing_dates) if (has_all_cols and missing_dates) else start_d
        fetch_e = max(missing_dates) if (has_all_cols and missing_dates) else end_d

        txf_df = pd.DataFrame()

        # =========================================================================
        # 第一優先軌道：期交所官方 CSV 真實下載端點 (dlFutDataDown) — 同步擷取「一般日盤」與「盤後夜盤」
        # =========================================================================
        taifex_collected = []
        chunk_s = fetch_s
        taifex_dl_url = "https://www.taifex.com.tw/cht/3/dlFutDataDown"

        session = requests.Session()
        session.headers.update({
            "User-Agent": self.headers["User-Agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.taifex.com.tw/cht/3/futDataDown",
            "Origin": "https://www.taifex.com.tw",
        })

        total_days_span = max((fetch_e - fetch_s).days, 1)
        total_chunks = (total_days_span // 28) + 1
        curr_chunk_idx = 0
        progress_bar = st.empty()

        while chunk_s <= fetch_e:
            curr_chunk_idx += 1
            chunk_e = min(chunk_s + datetime.timedelta(days=27), fetch_e)
            if total_chunks > 1:
                progress_bar.caption(
                    f"⏳ 正在自台灣期交所 (dlFutDataDown) 下載真實日盤與盤後夜盤行情 "
                    f"({curr_chunk_idx}/{total_chunks} 批次: {chunk_s.strftime('%Y/%m/%d')} ~ {chunk_e.strftime('%Y/%m/%d')})..."
                )

            payload = {
                "down_type": "1",
                "commodity_id": "TX",
                "commodity_id2": "",
                "queryStartDate": chunk_s.strftime("%Y/%m/%d"),
                "queryEndDate": chunk_e.strftime("%Y/%m/%d"),
            }
            try:
                t_resp = session.post(taifex_dl_url, data=payload, timeout=15)
                t_resp.encoding = "big5"
                text_content = t_resp.text.strip()

                if (
                    t_resp.status_code == 200
                    and "查無資料" not in text_content
                    and "<html" not in text_content.lower()
                    and "<!doctype" not in text_content.lower()
                    and "交易日期" in text_content
                ):
                    t_df = pd.read_csv(io.StringIO(text_content), index_col=False, on_bad_lines="skip")
                    t_df.columns = [str(c).strip().replace(" ", "").replace("﻿", "") for c in t_df.columns]

                    date_c = next((c for c in t_df.columns if any(k in c for k in ["交易日期", "日期"])), None)
                    contract_c = next((c for c in t_df.columns if any(k in c for k in ["到期月份", "月份"])), None)
                    open_c = next((c for c in t_df.columns if "開盤價" in c), None)
                    high_c = next((c for c in t_df.columns if "最高價" in c), None)
                    low_c = next((c for c in t_df.columns if "最低價" in c), None)
                    close_c = next((c for c in t_df.columns if "收盤價" in c or "最後成交價" in c), None)
                    settle_c = next((c for c in t_df.columns if "結算價" in c), None)
                    vol_c = next((c for c in t_df.columns if "成交量" in c), None)
                    halt_c = next((c for c in t_df.columns if "暫停交易" in c), None)

                    if date_c and contract_c and open_c and close_c:
                        t_df[contract_c] = t_df[contract_c].astype(str).str.strip()
                        t_df = t_df[t_df[contract_c].str.match(r"^\d{6}$", na=False)].copy()
                        t_df["date"] = pd.to_datetime(t_df[date_c].astype(str).str.strip().str.replace("/", "-"), errors="coerce").dt.normalize()

                        sess_col = None
                        if "交易時段" in t_df.columns:
                            sess_col = "交易時段"
                        elif halt_c and t_df[halt_c].astype(str).str.contains("一般|盤後", na=False).any():
                            sess_col = halt_c

                        # (A) 解析「一般」交易時段（08:45 ~ 13:45 日盤）
                        reg_df = t_df[t_df[sess_col].astype(str).str.contains("一般", na=False)].copy() if sess_col else t_df.copy()
                        reg_df["TXF_Open"] = pd.to_numeric(reg_df[open_c].astype(str).str.replace(",", "").str.strip().replace("-", np.nan), errors="coerce")
                        reg_df["TXF_Close"] = pd.to_numeric(reg_df[close_c].astype(str).str.replace(",", "").str.strip().replace("-", np.nan), errors="coerce")
                        reg_df["TXF_Settle"] = (
                            pd.to_numeric(reg_df[settle_c].astype(str).str.replace(",", "").str.strip().replace("-", np.nan), errors="coerce")
                            if settle_c
                            else np.nan
                        )
                        reg_df["TXF_High"] = (
                            pd.to_numeric(reg_df[high_c].astype(str).str.replace(",", "").str.strip().replace("-", np.nan), errors="coerce")
                            if high_c
                            else reg_df[["TXF_Open", "TXF_Close"]].max(axis=1)
                        )
                        reg_df["TXF_Low"] = (
                            pd.to_numeric(reg_df[low_c].astype(str).str.replace(",", "").str.strip().replace("-", np.nan), errors="coerce")
                            if low_c
                            else reg_df[["TXF_Open", "TXF_Close"]].min(axis=1)
                        )
                        reg_df["TXF_Volume"] = (
                            pd.to_numeric(reg_df[vol_c].astype(str).str.replace(",", "").str.strip().replace("-", "0"), errors="coerce").fillna(0.0)
                            if vol_c
                            else 0.0
                        )
                        reg_df = reg_df.dropna(subset=["date", "TXF_Open", "TXF_Close"])
                        reg_df = reg_df.sort_values(["date", contract_c]).groupby("date").first().reset_index()
                        clean_reg = reg_df[["date", "TXF_Open", "TXF_High", "TXF_Low", "TXF_Close", "TXF_Volume", "TXF_Settle"]].set_index("date")

                        # (B) 解析「盤後」交易時段（15:00 ~ 次日 05:00 夜盤）
                        if sess_col:
                            ah_df = t_df[t_df[sess_col].astype(str).str.contains("盤後", na=False)].copy()
                        else:
                            ah_df = pd.DataFrame()

                        if not ah_df.empty:
                            ah_df["TXF_AH_Open"] = pd.to_numeric(ah_df[open_c].astype(str).str.replace(",", "").str.strip().replace("-", np.nan), errors="coerce")
                            ah_df["TXF_AH_Close"] = pd.to_numeric(ah_df[close_c].astype(str).str.replace(",", "").str.strip().replace("-", np.nan), errors="coerce")
                            ah_df["TXF_AH_High"] = (
                                pd.to_numeric(ah_df[high_c].astype(str).str.replace(",", "").str.strip().replace("-", np.nan), errors="coerce")
                                if high_c
                                else ah_df[["TXF_AH_Open", "TXF_AH_Close"]].max(axis=1)
                            )
                            ah_df["TXF_AH_Low"] = (
                                pd.to_numeric(ah_df[low_c].astype(str).str.replace(",", "").str.strip().replace("-", np.nan), errors="coerce")
                                if low_c
                                else ah_df[["TXF_AH_Open", "TXF_AH_Close"]].min(axis=1)
                            )
                            ah_df["TXF_AH_Volume"] = (
                                pd.to_numeric(ah_df[vol_c].astype(str).str.replace(",", "").str.strip().replace("-", "0"), errors="coerce").fillna(0.0)
                                if vol_c
                                else 0.0
                            )
                            ah_df = ah_df.dropna(subset=["date", "TXF_AH_Open", "TXF_AH_Close"])
                            ah_df = ah_df.sort_values(["date", contract_c]).groupby("date").first().reset_index()
                            ah_df["txf_ah_ret"] = ((ah_df["TXF_AH_Close"] - ah_df["TXF_AH_Open"]) / ah_df["TXF_AH_Open"]) * 100.0
                            clean_ah = ah_df[["date", "TXF_AH_Open", "TXF_AH_High", "TXF_AH_Low", "TXF_AH_Close", "TXF_AH_Volume", "txf_ah_ret"]].set_index("date")
                            clean_batch = clean_reg.join(clean_ah, how="left")
                        else:
                            clean_batch = clean_reg.copy()
                            clean_batch["TXF_AH_Open"] = clean_batch["TXF_Open"]
                            clean_batch["TXF_AH_High"] = clean_batch["TXF_High"]
                            clean_batch["TXF_AH_Low"] = clean_batch["TXF_Low"]
                            clean_batch["TXF_AH_Close"] = clean_batch["TXF_Close"]
                            clean_batch["TXF_AH_Volume"] = 0.0
                            clean_batch["txf_ah_ret"] = 0.0

                        clean_batch["TXF_AH_Open"] = clean_batch["TXF_AH_Open"].fillna(clean_batch["TXF_Open"])
                        clean_batch["TXF_AH_High"] = clean_batch["TXF_AH_High"].fillna(clean_batch["TXF_High"])
                        clean_batch["TXF_AH_Low"] = clean_batch["TXF_AH_Low"].fillna(clean_batch["TXF_Low"])
                        clean_batch["TXF_AH_Close"] = clean_batch["TXF_AH_Close"].fillna(clean_batch["TXF_Close"])
                        clean_batch["TXF_AH_Volume"] = clean_batch["TXF_AH_Volume"].fillna(0.0)
                        clean_batch["txf_ah_ret"] = clean_batch["txf_ah_ret"].fillna(0.0)

                        if not clean_batch.empty:
                            taifex_collected.append(clean_batch)
            except Exception as e:
                st.warning(f"期交所批次 ({chunk_s} ~ {chunk_e}) 下載異常: {e}")

            chunk_s = chunk_e + datetime.timedelta(days=1)
            time.sleep(0.25)

        progress_bar.empty()

        if taifex_collected:
            all_official = pd.concat(taifex_collected)
            txf_df = all_official[~all_official.index.duplicated(keep="last")].sort_index()

        # =========================================================================
        # 第二備援軌道：僅針對期交所尚未公告或漏失之極少數日期，才以 FinMind 補齊（含夜盤解析）
        # =========================================================================
        fetched_dates = set(txf_df.index.date) if not txf_df.empty else set()
        remaining_missing = [d for d in missing_dates if d not in fetched_dates]

        if len(remaining_missing) > 0:
            api_token = safe_get_secret("FINMIND_TOKEN", "")
            url = "https://api.finmindtrade.com/api/v4/data"
            fm_s = min(remaining_missing)
            fm_e = max(remaining_missing)
            fm_batches = []
            cur_s = fm_s

            while cur_s <= fm_e:
                cur_e = min(cur_s + datetime.timedelta(days=180), fm_e)
                params = {
                    "dataset": "TaiwanFuturesDaily",
                    "data_id": "TX",
                    "start_date": cur_s.strftime("%Y-%m-%d"),
                    "end_date": cur_e.strftime("%Y-%m-%d"),
                }
                if api_token:
                    params["token"] = api_token

                try:
                    resp = requests.get(url, params=params, headers=self.headers, timeout=20)
                    if resp.status_code == 200:
                        data = resp.json().get("data", [])
                        if data:
                            raw_df = pd.DataFrame(data)
                            raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.normalize()
                            raw_df = raw_df[raw_df["date"].dt.date.isin(remaining_missing)]
                            if "contract_date" in raw_df.columns:
                                raw_df = raw_df[raw_df["contract_date"].astype(str).str.strip().str.len() == 6]

                            if not raw_df.empty:
                                high_col = next((c for c in raw_df.columns if str(c).lower() in ["max", "high", "最高價"]), None)
                                low_col = next((c for c in raw_df.columns if str(c).lower() in ["min", "low", "最低價"]), None)
                                vol_col = next((c for c in raw_df.columns if str(c).lower() in ["volume", "deal_volume", "成交量"]), None)
                                settle_col = next((c for c in raw_df.columns if any(k in str(c).lower() for k in ["settle", "結算價", "結算"])), None)
                                close_col = next((c for c in raw_df.columns if str(c).lower() in ["close", "close_price", "trade_close", "last_price", "收盤價", "最後成交價"]), None)

                                if "trading_session" in raw_df.columns:
                                    reg_fm = raw_df[raw_df["trading_session"].astype(str).str.contains("position|regular|一般", case=False, na=True)].copy()
                                    ah_fm = raw_df[raw_df["trading_session"].astype(str).str.contains("after|night|盤後", case=False, na=False)].copy()
                                else:
                                    reg_fm = raw_df.copy()
                                    ah_fm = pd.DataFrame()

                                if "contract_date" in reg_fm.columns:
                                    reg_fm = reg_fm.sort_values(["date", "contract_date"]).groupby("date").first().reset_index()

                                reg_fm["TXF_Open"] = pd.to_numeric(reg_fm["open"], errors="coerce")
                                reg_fm["TXF_Close"] = pd.to_numeric(reg_fm[close_col], errors="coerce") if close_col else np.nan
                                reg_fm["TXF_Settle"] = pd.to_numeric(reg_fm[settle_col], errors="coerce") if settle_col else np.nan
                                reg_fm["TXF_High"] = pd.to_numeric(reg_fm[high_col], errors="coerce") if high_col else reg_fm[["TXF_Open", "TXF_Close"]].max(axis=1)
                                reg_fm["TXF_Low"] = pd.to_numeric(reg_fm[low_col], errors="coerce") if low_col else reg_fm[["TXF_Open", "TXF_Close"]].min(axis=1)
                                reg_fm["TXF_Volume"] = pd.to_numeric(reg_fm[vol_col], errors="coerce").fillna(0.0) if vol_col else 0.0
                                fm_df = reg_fm[["date", "TXF_Open", "TXF_High", "TXF_Low", "TXF_Close", "TXF_Volume", "TXF_Settle"]].dropna(subset=["TXF_Open", "TXF_Close"]).set_index("date").sort_index()

                                if not ah_fm.empty:
                                    if "contract_date" in ah_fm.columns:
                                        ah_fm = ah_fm.sort_values(["date", "contract_date"]).groupby("date").first().reset_index()
                                    ah_fm["TXF_AH_Open"] = pd.to_numeric(ah_fm["open"], errors="coerce")
                                    ah_fm["TXF_AH_Close"] = pd.to_numeric(ah_fm[close_col], errors="coerce") if close_col else np.nan
                                    ah_fm["TXF_AH_High"] = pd.to_numeric(ah_fm[high_col], errors="coerce") if high_col else ah_fm[["TXF_AH_Open", "TXF_AH_Close"]].max(axis=1)
                                    ah_fm["TXF_AH_Low"] = pd.to_numeric(ah_fm[low_col], errors="coerce") if low_col else ah_fm[["TXF_AH_Open", "TXF_AH_Close"]].min(axis=1)
                                    ah_fm["TXF_AH_Volume"] = pd.to_numeric(ah_fm[vol_col], errors="coerce").fillna(0.0) if vol_col else 0.0
                                    ah_fm = ah_fm.dropna(subset=["TXF_AH_Open", "TXF_AH_Close"])
                                    ah_fm["txf_ah_ret"] = ((ah_fm["TXF_AH_Close"] - ah_fm["TXF_AH_Open"]) / ah_fm["TXF_AH_Open"]) * 100.0
                                    ah_clean = ah_fm[["date", "TXF_AH_Open", "TXF_AH_High", "TXF_AH_Low", "TXF_AH_Close", "TXF_AH_Volume", "txf_ah_ret"]].set_index("date")
                                    fm_df = fm_df.join(ah_clean, how="left")
                                else:
                                    fm_df["TXF_AH_Open"] = fm_df["TXF_Open"]
                                    fm_df["TXF_AH_High"] = fm_df["TXF_High"]
                                    fm_df["TXF_AH_Low"] = fm_df["TXF_Low"]
                                    fm_df["TXF_AH_Close"] = fm_df["TXF_Close"]
                                    fm_df["TXF_AH_Volume"] = 0.0
                                    fm_df["txf_ah_ret"] = 0.0

                                fm_df["TXF_AH_Open"] = fm_df["TXF_AH_Open"].fillna(fm_df["TXF_Open"])
                                fm_df["TXF_AH_High"] = fm_df["TXF_AH_High"].fillna(fm_df["TXF_High"])
                                fm_df["TXF_AH_Low"] = fm_df["TXF_AH_Low"].fillna(fm_df["TXF_Low"])
                                fm_df["TXF_AH_Close"] = fm_df["TXF_AH_Close"].fillna(fm_df["TXF_Close"])
                                fm_df["TXF_AH_Volume"] = fm_df["TXF_AH_Volume"].fillna(0.0)
                                fm_df["txf_ah_ret"] = fm_df["txf_ah_ret"].fillna(0.0)

                                if not fm_df.empty:
                                    fm_batches.append(fm_df)
                except Exception as e:
                    st.warning(f"FinMind 備援行情下載異常: {e}")
                cur_s = cur_e + datetime.timedelta(days=1)

            if fm_batches:
                all_fm = pd.concat(fm_batches)
                all_fm = all_fm[~all_fm.index.duplicated(keep="last")].sort_index()
                txf_df = txf_df.combine_first(all_fm) if not txf_df.empty else all_fm

        # 寫入磁碟 Parquet 快取（新抓到的期交所官方資料優先覆蓋舊快取）
        if not txf_df.empty:
            full_txf = txf_df.combine_first(disk_df) if not disk_df.empty else txf_df
            full_txf = full_txf[~full_txf.index.duplicated(keep="last")].sort_index()
            try:
                full_txf.to_parquet(self.txf_price_parquet_path)
            except Exception:
                pass
            disk_df = full_txf

        if not disk_df.empty:
            return disk_df[(disk_df.index.date >= start_d) & (disk_df.index.date <= end_d)].copy()

        return pd.DataFrame()
    
    def fetch_web_fallback_stock_return(self, ticker: str) -> Optional[float]:
        """當 Yahoo Finance 下載為 0.00% 偽平盤時，利用網路即時搜尋/檢索端點直接攫取真實最新報酬率"""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        sym = ticker.upper().strip()

        # 第一軌：Yahoo Finance 即時 Chart API (不透過 yfinance 快取)
        try:
            url_yf = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=5d"
            r = requests.get(url_yf, headers=headers, timeout=6)
            if r.status_code == 200:
                data = r.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                reg_price = meta.get("regularMarketPrice")
                prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
                if reg_price and prev_close and float(prev_close) > 0:
                    ret_val = ((float(reg_price) - float(prev_close)) / float(prev_close)) * 100.0
                    if abs(ret_val) > 1e-5:
                        return round(ret_val, 4)
        except Exception:
            pass

        # 第二軌：Google Finance 網頁即時解析
        try:
            exchange = "NYSE" if sym == "TSM" else "NASDAQ"
            url_gf = f"https://www.google.com/finance/quote/{sym}:{exchange}"
            r_gf = requests.get(url_gf, headers=headers, timeout=6)
            if r_gf.status_code == 200:
                import re
                m = re.search(r'data-percentage-change="([+-]?\d+\.?\d*)"', r_gf.text)
                if m:
                    ret_val = float(m.group(1))
                    if abs(ret_val) > 1e-5:
                        return round(ret_val, 4)
                m2 = re.search(r'>([+-]?\d+\.\d+)%</span>', r_gf.text)
                if m2:
                    ret_val = float(m2.group(1))
                    if abs(ret_val) > 1e-5:
                        return round(ret_val, 4)
        except Exception:
            pass

        # 第三軌：CNBC 即時行情 WebService
        try:
            url_cnbc = f"https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol?symbols={sym}&requestMethod=itv&output=json"
            r_cnbc = requests.get(url_cnbc, headers=headers, timeout=6)
            if r_cnbc.status_code == 200:
                c_data = r_cnbc.json()
                q_list = c_data.get("FormattedQuoteResult", {}).get("FormattedQuote", [])
                if q_list:
                    pct_str = q_list[0].get("change_pct", "").replace("%", "").replace(",", "")
                    if pct_str:
                        ret_val = float(pct_str)
                        if abs(ret_val) > 1e-5:
                            return round(ret_val, 4)
        except Exception:
            pass

        return None
    
    def fetch_us_market(
        self, target_date: datetime.date, days_back: int = 40
    ) -> Tuple[pd.DataFrame, datetime.date, str]:
        tickers = {
            "TWD=X": "usdtwd_ret",
            "^SOX": "sox_ret",
            "TSM": "tsm_ret",
            "NVDA": "nvda_ret",
            "^GSPC": "sp500_ret",
            "^VIX": "vix_close",
        }
        try:
            start_d = (
                target_date - datetime.timedelta(days=days_back)
            ).strftime("%Y-%m-%d")
            end_d = (target_date + datetime.timedelta(days=3)).strftime(
                "%Y-%m-%d"
            )

            raw = yf.download(
                list(tickers.keys()),
                start=start_d,
                end=end_d,
                progress=False,
                auto_adjust=False,
            )

            if raw.empty:
                return pd.DataFrame(), target_date, "Yahoo Finance 未回傳行情"

            price = raw["Close"] if "Close" in raw else raw["Adj Close"]

            if isinstance(price, pd.Series):
                price = price.to_frame()

            if hasattr(price.index, "tz") and price.index.tz is not None:
                price.index = price.index.tz_convert(
                    "America/New_York"
                ).tz_localize(None)
            price.index = pd.to_datetime([d.date() for d in price.index])
            price = price[~price.index.duplicated(keep="last")].sort_index()

            # 僅針對核心美股標的 (^SOX, TSM, NVDA, ^VIX) 做前向補值，避免單一匯率/指數缺漏剔除整天交易日
            core_us_cols = [c for c in ["^SOX", "TSM", "NVDA", "^VIX"] if c in price.columns]
            if core_us_cols:
                price[core_us_cols] = price[core_us_cols].ffill()
            if "TWD=X" in price.columns:
                price["TWD=X"] = price["TWD=X"].ffill()

            features = pd.DataFrame(index=price.index)
            for tk, name in tickers.items():
                if tk in price.columns:
                    if name == "vix_close":
                        features[name] = price[tk]
                        features["vix_chg"] = price[tk].diff().fillna(0.0)
                    else:
                        features[name] = price[tk].pct_change() * 100.0

            # 只要 ^SOX 或 TSM 其中之一有有效報酬率即保留該交易日
            req_ret_cols = [c for c in ["sox_ret", "tsm_ret"] if c in features.columns]
            if req_ret_cols:
                features = features.dropna(subset=req_ret_cols, how="all")
            features = features.ffill().dropna(how="all")

            valid_us = features[features.index <= pd.to_datetime(target_date)]

            if valid_us.empty:
                return pd.DataFrame(), target_date, "無有效美股/匯率數據"

    

            # 僅當主要美股與波動率同時完全無任何波動 (全部接近 0) 時，才視為尚未更新之暫態列
            actual_idx = len(valid_us) - 1
            while actual_idx > 0:
                row = valid_us.iloc[actual_idx]
                sox_v = abs(float(row.get("sox_ret", 0.0)))
                tsm_v = abs(float(row.get("tsm_ret", 0.0)))
                nvda_v = abs(float(row.get("nvda_ret", 0.0)))
                vix_v = abs(float(row.get("vix_chg", 0.0)))
                is_stale = (sox_v < 1e-6 and tsm_v < 1e-6 and nvda_v < 1e-6 and vix_v < 1e-6)
                if not is_stale:
                    break
                actual_idx -= 1

            matched_us = valid_us.iloc[: actual_idx + 1].copy()

            # 檢驗最後一筆美股資料：若台積電 ADR 或 NVDA 呈現 0 波動偽平盤，自動啟動網路檢索替代方案
            if not matched_us.empty:
                last_dt = matched_us.index[-1]
                if abs(float(matched_us.loc[last_dt, "tsm_ret"])) < 1e-5:
                    web_tsm = self.fetch_web_fallback_stock_return("TSM")
                    if web_tsm is not None:
                        matched_us.loc[last_dt, "tsm_ret"] = web_tsm
                if abs(float(matched_us.loc[last_dt, "nvda_ret"])) < 1e-5:
                    web_nvda = self.fetch_web_fallback_stock_return("NVDA")
                    if web_nvda is not None:
                        matched_us.loc[last_dt, "nvda_ret"] = web_nvda

            latest_us_date = matched_us.index[-1].date()     #2026/11/18

            msg = (
                f"正常取得 ({latest_us_date})"
                if latest_us_date == target_date
                else f"原日前推至有效交易日 ({latest_us_date})"
            )
            return matched_us, latest_us_date, msg

        except Exception as e:
            return pd.DataFrame(), target_date, f"擷取異常: {str(e)}"

    
    
    def fetch_historical_training_and_verification_data(
        self, target_date: datetime.date, exact_train_days: int = 520
    ) -> pd.DataFrame:
        """抓取 520 個台股真實交易日樣本（含近月台指期日盤/盤後夜盤、期現貨價差率、技術分析均線虛擬變數、VIX變動與國安基金虛擬變數）"""
        try:
            calendar_days = int(exact_train_days * 1.65) + 60
            start_date_obj = target_date - datetime.timedelta(days=calendar_days)
            start_d = start_date_obj.strftime("%Y-%m-%d")
            # 確保僅抓取目標預測日 target_date 前已確定收盤的美股歷史行情
            end_d = (target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d") #2026/11/18

            tickers = {
                "^TWII": "twii",
                "TWD=X": "usdtwd",
                "^SOX": "sox_close",
                "TSM": "tsm_close",
                "NVDA": "nvda_close",
                "^VIX": "vix_close",
            }

            raw = yf.download(
                list(tickers.keys()),
                start=start_d,
                end=end_d,
                progress=False,
                auto_adjust=False,
            )

            def get_series(metric: str, ticker: str):
                if isinstance(raw.columns, pd.MultiIndex):
                    if metric in raw and ticker in raw[metric]:
                        return raw[metric][ticker]
                elif metric in raw:
                    return raw[metric]
                return pd.Series(dtype=float)

            twii_open = get_series("Open", "^TWII")
            twii_high = get_series("High", "^TWII")
            twii_low = get_series("Low", "^TWII")
            twii_close = get_series("Close", "^TWII")

            usdtwd_close = get_series("Close", "TWD=X")
            sox_close = get_series("Close", "^SOX")
            tsm_close = get_series("Close", "TSM")
            nvda_close = get_series("Close", "NVDA")
            vix_close = get_series("Close", "^VIX")

            tw_raw = pd.DataFrame({
                "TWII_Open": twii_open,
                "TWII_High": twii_high,
                "TWII_Low": twii_low,
                "TWII_Close": twii_close,
            }).dropna(subset=["TWII_Open", "TWII_Close"])

            if hasattr(tw_raw.index, "tz") and tw_raw.index.tz is not None:
                tw_raw.index = tw_raw.index.tz_convert("Asia/Taipei").tz_localize(None)
            tw_raw.index = pd.to_datetime([d.date() for d in tw_raw.index])
            tw_raw = tw_raw[~tw_raw.index.duplicated(keep="last")].sort_index()

            # 建立完整預期台股交易日曆，避免 Yahoo Finance 漏失特定台股交易日 (如 2026-11-16)
            expected_tw_dates = []
            curr_cal = start_date_obj
            while curr_cal <= target_date:
                if curr_cal.weekday() < 5 and curr_cal not in TW_HOLIDAYS_MAP:
                    expected_tw_dates.append(pd.to_datetime(curr_cal))
                curr_cal += datetime.timedelta(days=1)

            tw_calendar_dummy = pd.DataFrame(index=expected_tw_dates)

            # 2. 獲取台指期近月獨立日盤與盤後夜盤開盤/收盤價/結算價（傳入完整台股預期交易日曆，強制檢核並補齊所有營業日）
            txf_prices_df = self.fetch_txf_prices_historical_parquet(
                start_date_obj, target_date, tw_raw=tw_calendar_dummy
            )
            if not txf_prices_df.empty:
                txf_prices_df.index = pd.to_datetime(txf_prices_df.index).normalize()
                txf_prices_df = txf_prices_df[~txf_prices_df.index.duplicated(keep="last")].sort_index()

                union_idx = tw_raw.index.union(txf_prices_df.index).sort_values()
                union_idx = [
                    d for d in union_idx
                    if d.weekday() < 5 and d.date() not in TW_HOLIDAYS_MAP and start_date_obj <= d.date() <= target_date
                ]
                tw_raw = tw_raw.reindex(union_idx)
                tw_raw = tw_raw.join(txf_prices_df, how="left")

                tw_raw["TWII_Open"] = tw_raw["TWII_Open"].fillna(tw_raw["TXF_Open"])
                tw_raw["TWII_High"] = tw_raw["TWII_High"].fillna(tw_raw["TXF_High"])
                tw_raw["TWII_Low"] = tw_raw["TWII_Low"].fillna(tw_raw["TXF_Low"])
                tw_raw["TWII_Close"] = tw_raw["TWII_Close"].fillna(tw_raw["TXF_Close"])

                tw_raw["TXF_Open"] = tw_raw["TXF_Open"].fillna(tw_raw["TWII_Open"])
                tw_raw["TXF_High"] = tw_raw["TXF_High"].fillna(tw_raw["TWII_High"])
                tw_raw["TXF_Low"] = tw_raw["TXF_Low"].fillna(tw_raw["TWII_Low"])
                tw_raw["TXF_Close"] = tw_raw["TXF_Close"].fillna(tw_raw["TWII_Close"])
                tw_raw["TXF_Settle"] = tw_raw["TXF_Settle"].fillna(tw_raw["TXF_Close"])
                tw_raw["TXF_Volume"] = tw_raw["TXF_Volume"].fillna(0.0)

                tw_raw["TXF_AH_Open"] = tw_raw.get("TXF_AH_Open", tw_raw["TXF_Open"]).fillna(tw_raw["TXF_Open"])
                tw_raw["TXF_AH_High"] = tw_raw.get("TXF_AH_High", tw_raw["TXF_High"]).fillna(tw_raw["TXF_High"])
                tw_raw["TXF_AH_Low"] = tw_raw.get("TXF_AH_Low", tw_raw["TXF_Low"]).fillna(tw_raw["TXF_Low"])
                tw_raw["TXF_AH_Close"] = tw_raw.get("TXF_AH_Close", tw_raw["TXF_Close"]).fillna(tw_raw["TXF_Close"])
                tw_raw["TXF_AH_Volume"] = tw_raw.get("TXF_AH_Volume", pd.Series(0.0, index=tw_raw.index)).fillna(0.0)
                tw_raw["txf_ah_ret"] = tw_raw.get("txf_ah_ret", pd.Series(0.0, index=tw_raw.index)).fillna(0.0)
                tw_raw = tw_raw.dropna(subset=["TWII_Open", "TWII_Close", "TXF_Open", "TXF_Close"])
            else:
                valid_tw_dates = [
                    d
                    for d in tw_raw.index
                    if d.weekday() < 5 and d.date() not in TW_HOLIDAYS_MAP
                ]
                tw_raw = tw_raw.loc[valid_tw_dates]
                st.warning("⚠️ 暫無法自期交所取得完整歷史期貨行情，已自動以加權指數行情備援。")
                tw_raw["TXF_Open"] = tw_raw["TWII_Open"]
                tw_raw["TXF_High"] = tw_raw["TWII_High"]
                tw_raw["TXF_Low"] = tw_raw["TWII_Low"]
                tw_raw["TXF_Close"] = tw_raw["TWII_Close"]
                tw_raw["TXF_Settle"] = tw_raw["TWII_Close"]
                tw_raw["TXF_Volume"] = 0.0
                tw_raw["TXF_AH_Open"] = tw_raw["TWII_Open"]
                tw_raw["TXF_AH_High"] = tw_raw["TWII_High"]
                tw_raw["TXF_AH_Low"] = tw_raw["TWII_Low"]
                tw_raw["TXF_AH_Close"] = tw_raw["TWII_Close"]
                tw_raw["TXF_AH_Volume"] = 0.0
                tw_raw["txf_ah_ret"] = 0.0

            # 美股與匯率獨立清理並對價格先做 ffill() 後再計算報酬率與 VIX 單日變動量
            fx_df = pd.DataFrame({"usdtwd_close": usdtwd_close}).dropna()
            if hasattr(fx_df.index, "tz") and fx_df.index.tz is not None:
                fx_df.index = fx_df.index.tz_convert("America/New_York").tz_localize(None)
            fx_df.index = pd.to_datetime([d.date() for d in fx_df.index])
            fx_df = fx_df[~fx_df.index.duplicated(keep="last")].sort_index()
            fx_df["usdtwd_ret"] = fx_df["usdtwd_close"].pct_change() * 100.0
            fx_df["usdtwd_ret"] = fx_df["usdtwd_ret"].ffill().fillna(0.0)

            us_mkt_close = pd.DataFrame({
                "sox_close": sox_close,
                "tsm_close": tsm_close,
                "nvda_close": nvda_close,
                "vix_close": vix_close,
            }).dropna(how="all")
            if hasattr(us_mkt_close.index, "tz") and us_mkt_close.index.tz is not None:
                us_mkt_close.index = us_mkt_close.index.tz_convert("America/New_York").tz_localize(None)
            us_mkt_close.index = pd.to_datetime([d.date() for d in us_mkt_close.index])
            us_mkt_close = us_mkt_close[~us_mkt_close.index.duplicated(keep="last")].sort_index()

            # 修正前（易產生偽平盤）：
            # us_mkt_close = us_mkt_close.dropna(subset=["sox_close", "tsm_close", "nvda_close"], how="all")
            # us_mkt_close = us_mkt_close.ffill()
            # us_raw["tsm_ret"] = us_mkt_close["tsm_close"].pct_change() * 100.0

            # 修正後（先算真實波動，缺漏才以報酬率前向填補）：
            us_mkt_close = us_mkt_close.dropna(how="all")
            
            us_raw = pd.DataFrame(index=us_mkt_close.index)
            # 先對各標的個別計算 pct_change()
            us_raw["sox_ret"] = us_mkt_close["sox_close"].pct_change() * 100.0
            us_raw["tsm_ret"] = us_mkt_close["tsm_close"].pct_change() * 100.0
            us_raw["nvda_ret"] = us_mkt_close["nvda_close"].pct_change() * 100.0
            us_raw["vix_close"] = us_mkt_close["vix_close"]
            us_raw["vix_chg"] = us_mkt_close["vix_close"].diff()

            # 若特定個股當日尚未更新 (NaN)，以「前一交易日真實報酬率」填補，杜絕 0% 偽平盤
            us_raw = us_raw.ffill().fillna(0.0)

            # 若最末一筆對齊交易日的 TSM 或 NVDA 依然為 0.00%，啟動網路即時檢索替換
            if not us_raw.empty:
                last_us_idx = us_raw.index[-1]
                if abs(float(us_raw.loc[last_us_idx, "tsm_ret"])) < 1e-5:
                    web_tsm = self.fetch_web_fallback_stock_return("TSM")
                    if web_tsm is not None:
                        us_raw.loc[last_us_idx, "tsm_ret"] = web_tsm
                if abs(float(us_raw.loc[last_us_idx, "nvda_ret"])) < 1e-5:
                    web_nvda = self.fetch_web_fallback_stock_return("NVDA")
                    if web_nvda is not None:
                        us_raw.loc[last_us_idx, "nvda_ret"] = web_nvda
            
            aligned_rows = []
            for dt in tw_raw.index:
                # 跨時區精準對齊：台灣交易日 dt (T) 早上開盤前已收盤之最新美股為 < dt (即 T-1 美股交易日，台灣時間 T 日凌晨 04:00 收盤)
                matched_us = us_raw[us_raw.index < dt]
                matched_fx = fx_df[fx_df.index < dt]
                if matched_us.empty:
                    matched_us = us_raw[us_raw.index <= dt]
                if matched_fx.empty:
                    matched_fx = fx_df[fx_df.index <= dt]
                if not matched_us.empty:
                    last_us = matched_us.iloc[-1]
                    tw_row = tw_raw.loc[dt]
                    last_us_dt = matched_us.index[-1].date()
                    fx_ret_val = (
                        float(matched_fx.iloc[-1]["usdtwd_ret"])
                        if not matched_fx.empty
                        else 0.0
                    )

                    aligned_rows.append({
                        "date": dt,
                        "TWII_Open": float(tw_row["TWII_Open"]),
                        "TWII_High": float(tw_row["TWII_High"]),
                        "TWII_Low": float(tw_row["TWII_Low"]),
                        "TWII_Close": float(tw_row["TWII_Close"]),
                        "TXF_Open": float(tw_row["TXF_Open"]),
                        "TXF_High": float(tw_row.get("TXF_High", tw_row["TXF_Open"])),
                        "TXF_Low": float(tw_row.get("TXF_Low", tw_row["TXF_Close"])),
                        "TXF_Close": float(tw_row["TXF_Close"]),
                        "TXF_Settle": float(tw_row.get("TXF_Settle", tw_row["TXF_Close"])),
                        "TXF_Volume": float(tw_row.get("TXF_Volume", 0.0)),
                        "TXF_AH_Open": float(tw_row.get("TXF_AH_Open", tw_row["TXF_Open"])),
                        "TXF_AH_High": float(tw_row.get("TXF_AH_High", tw_row["TXF_High"])),
                        "TXF_AH_Low": float(tw_row.get("TXF_AH_Low", tw_row["TXF_Low"])),
                        "TXF_AH_Close": float(tw_row.get("TXF_AH_Close", tw_row["TXF_Close"])),
                        "TXF_AH_Volume": float(tw_row.get("TXF_AH_Volume", 0.0)),
                        "txf_ah_ret": float(tw_row.get("txf_ah_ret", 0.0)),
                        "usdtwd_ret": fx_ret_val,
                        "sox_ret": float(last_us["sox_ret"]),
                        "tsm_ret": float(last_us["tsm_ret"]),
                        "nvda_ret": float(last_us.get("nvda_ret", 0.0)),
                        "vix_close": float(last_us["vix_close"]),
                        "vix_chg": float(last_us.get("vix_chg", 0.0)),
                        "nsf_active": float(check_is_nsf_active(dt.date())),
                        "us_date": last_us_dt,
                        "us_event": US_MACRO_EVENTS_MAP.get(last_us_dt, ""),
                    })

            if not aligned_rows:
                return pd.DataFrame()

            full_df = pd.DataFrame(aligned_rows).set_index("date").sort_index()
            full_df = full_df[
                full_df.index <= pd.to_datetime(target_date)
            ].copy()

            full_df["prev_twii_close"] = full_df["TWII_Close"].shift(1)
            full_df["gap_ret"] = (
                (full_df["TWII_Open"] - full_df["prev_twii_close"])
                / full_df["prev_twii_close"]
            ) * 100.0

            # 修正第 10 個解釋變數【期現貨價差率】：([預測日近月台指期夜盤收盤價] - [預測日前一交易日台灣加權指數收盤價]) / [預測日前一交易日台灣加權指數收盤價]
            # 預測日夜盤於日盤開盤前已結束；若無法正常取得則以前一交易日數據取代 (ffill)
            ah_close_clean = full_df["TXF_AH_Close"].replace(0, np.nan).ffill().bfill()
            full_df["basis_ret"] = (
                (ah_close_clean - full_df["prev_twii_close"])
                / full_df["prev_twii_close"]
            ) * 100.0
            full_df["basis_ret"] = full_df["basis_ret"].ffill().bfill().fillna(0.0)

            # 修正第 11 個解釋變數【預測日近月台指期夜盤的報酬率】：(TXF_AH_Close - TXF_AH_Open) / TXF_AH_Open
            # 若無法正常取得則以前一交易日數據取代 (ffill)
            ah_open_clean = full_df["TXF_AH_Open"].replace(0, np.nan).ffill().bfill()
            full_df["txf_ah_ret"] = (
                (ah_close_clean - ah_open_clean)
                / ah_open_clean
            ) * 100.0
            full_df["txf_ah_ret"] = full_df["txf_ah_ret"].ffill().bfill().fillna(0.0)
            
            # --- 技術分析均線指標計算區塊 (供 T-1 虛擬變數使用) ---
            ma5_series = full_df["TWII_Close"].rolling(5, min_periods=1).mean()
            ma10_series = full_df["TWII_Close"].rolling(10, min_periods=1).mean()
            full_df["ma5_bias_pct"] = (
                (full_df["TWII_Close"] - ma5_series)
                / ma5_series
            ) * 100.0
            full_df["ma10_bias_pct"] = (
                (full_df["TWII_Close"] - ma10_series)
                / ma10_series
            ) * 100.0
            # 第 14 解釋變數 ma5_bias (虛擬變數)：if ma5_bias > 0 then 1 else 0
            full_df["ma5_bias"] = (full_df["ma5_bias_pct"] > 0).astype(float)
            # 第 15 解釋變數 ma10_bias (虛擬變數)：if ma10_bias > 0 then 1 else 0
            full_df["ma10_bias"] = (full_df["ma10_bias_pct"] > 0).astype(float)
            # 第 16 解釋變數 ma5_ma10 (虛擬變數)：if ma5 > ma10 then 1 else 0
            full_df["ma5_ma10"] = (ma5_series > ma10_series).astype(float)

            gap_up_filled = (full_df["gap_ret"] > 0) & (
                full_df["TWII_Low"] <= full_df["prev_twii_close"]
            )
            gap_down_filled = (full_df["gap_ret"] < 0) & (
                full_df["TWII_High"] >= full_df["prev_twii_close"]
            )
            full_df["gap_filled"] = gap_up_filled | gap_down_filled

            full_df["twii_ret"] = (
                (full_df["TWII_Close"] - full_df["prev_twii_close"])
                / full_df["prev_twii_close"]
            ) * 100.0
            full_df["ret_open_to_close"] = (
                (full_df["TWII_Close"] - full_df["TWII_Open"])
                / full_df["TWII_Open"]
            ) * 100.0
            full_df["txf_day_trade_ret"] = (
                (full_df["TXF_Close"] - full_df["TXF_Open"])
                / full_df["TXF_Open"]
            ) * 100.0

            full_df = full_df.dropna(subset=["TWII_Open", "TWII_Close", "TXF_Open", "TXF_Close", "prev_twii_close"])

            res_df = full_df.tail(exact_train_days).copy()
            res_df.index = pd.to_datetime(res_df.index).tz_localize(None)
            return res_df

        except Exception:
            return pd.DataFrame()


# -----------------------------------------------------------------------------
# 2. Logistic Regression 機率推論引擎 (16 大升級特徵變量版 - 含技術分析虛擬變數區塊)
# -----------------------------------------------------------------------------
class TaiexLogisticEngine:

    def __init__(
        self,
        transform_mode: str = "原始線性水準項",
        auto_select_features: bool = True,
    ):
        self.ALL_FEATURES = [
            # --- 外資籌碼面 ---
            "foreign_net",      # 1. 外資現貨買賣超 (水準項)
            "tx_oi_diff",       # 2. 單日未平倉增減變動量 (ΔOI, 水準項)
            "fx_inter",         # 3. 外資期現幾何共振項 (sign(Z_f+Z_dOI)*|Z_f*Z_dOI|)
            # --- 國際美股與匯率波動面 ---
            "usdtwd_ret",       # 4. 美元兌台幣變動 (USD/TWD %)
            "sox_ret",          # 5. 美股費半漲跌幅 (SOX %)
            "tsm_ret",          # 6. 台積電 ADR 漲跌幅 (TSM %)
            "sox_tsm_inter",    # 7. 費半×台積電ADR幾何共振項 (sign(Z_sox+Z_tsm)*|Z_sox*Z_tsm|)
            "vix_chg",          # 8. VIX 恐慌指數變動 (ΔVIX)
            "nvda_ret",         # 9. 輝達股票漲跌幅 (NVDA %)
            # --- 期現貨與夜盤面 ---
            "basis_ret",        # 10. 期現貨價差率 (Basis %)
            "txf_ah_ret",       # 11. 台指期盤後夜盤漲跌幅 (Night Session %)
            # --- 政策面 ---
            "nsf_active",       # 12. 國安基金進場護盤虛擬變數 (Dummy 0/1)
            # --- 技術分析相關解釋變數區塊 (皆為預測日前一交易日 T-1 確立數據) ---
            "gap_ret",          # 13. 開盤跳空幅度 (Gap %)
            "ma5_bias",         # 14. 5日均線正乖離虛擬變數 (T-1: ma5_bias > 0 則 1, 否則 0)
            "ma10_bias",        # 15. 10日均線正乖離虛擬變數 (T-1: ma10_bias > 0 則 1, 否則 0)
            "ma5_ma10",         # 16. 5日/10日均線多頭排列虛擬變數 (T-1: ma5 > ma10 則 1, 否則 0)
            # --- 借券放空籌碼面 ---
            "sbl_5d_chg",       # 17. 近 5 日借券賣出餘額累計變動 (張, T-1)
        ]
        self.transform_mode = transform_mode

        self.default_scaler = {
            "foreign_net": {"mu": 0.0, "sigma": 150.0},
            "tx_oi_diff": {"mu": 0.0, "sigma": 3000.0},
            "fx_inter": {"mu": 0.00, "sigma": 1.10},
            "usdtwd_ret": {"mu": 0.00, "sigma": 0.25},
            "sox_ret": {"mu": 0.05, "sigma": 1.8},
            "tsm_ret": {"mu": 0.10, "sigma": 2.2},
            "sox_tsm_inter": {"mu": 0.00, "sigma": 1.25},
            "vix_chg": {"mu": 0.00, "sigma": 1.8},
            "basis_ret": {"mu": -0.05, "sigma": 0.35},
            "txf_ah_ret": {"mu": 0.02, "sigma": 0.75},
            "nsf_active": {"mu": 0.0, "sigma": 1.0},  # 虛擬變數 (0/1) 不標準化偏移
            "nvda_ret": {"mu": 0.15, "sigma": 3.10},
            # --- 技術分析相關變數區塊 ---
            "gap_ret": {"mu": 0.05, "sigma": 0.85},
            "ma5_bias": {"mu": 0.0, "sigma": 1.0},    # 虛擬變數 (0/1) 不標準化偏移
            "ma10_bias": {"mu": 0.0, "sigma": 1.0},   # 虛擬變數 (0/1) 不標準化偏移
            "ma5_ma10": {"mu": 0.0, "sigma": 1.0},    # 虛擬變數 (0/1) 不標準化偏移
            # --- 借券放空籌碼面 ---
            "sbl_5d_chg": {"mu": 0.0, "sigma": 150000.0},
        }

        self.default_beta = {
            "foreign_net": 0.45,
            "tx_oi_diff": 0.40,
            "fx_inter": 0.30,
            "usdtwd_ret": -0.40,
            "sox_ret": 0.60,
            "tsm_ret": 0.55,
            "sox_tsm_inter": 0.25,
            "vix_chg": -0.35,
            "basis_ret": 0.25,
            "txf_ah_ret": 0.45,
            "nsf_active": 0.50,
            "nvda_ret": 0.40,
            # --- 技術分析相關變數區塊 ---
            "gap_ret": -0.25,
            "ma5_bias": 0.20,
            "ma10_bias": 0.20,
            "ma5_ma10": 0.25,
            # --- 借券放空籌碼面 ---
            "sbl_5d_chg": -0.25,
        }
        self.default_intercept = 0.05

        self.active_scaler = self.default_scaler.copy()
        self.active_beta = self.default_beta.copy()
        self.active_intercept = self.default_intercept
        self.best_c_param = 1.0
        self.auto_select_features = auto_select_features
        self.selected_features = self.ALL_FEATURES.copy()
        self.eliminated_features = []
        self.full_13_accuracy = None
        self.fit_accuracy = None
        self.fit_sample_size = 0
        self.in_sample_probs = []
        self.in_sample_df = pd.DataFrame()

    @staticmethod
    def compute_directional_interaction(z1, z2):
        """
        【幾何共振項 (Directional Geometric Resonance)】：
        公式：sign(Z_1 + Z_2) * |Z_1 * Z_2|
        - 當兩因子同向看多 (Z_1 > 0, Z_2 > 0) 時為正值 (+)
        - 當兩因子同向看空 (Z_1 < 0, Z_2 < 0) 時為負值 (-)，徹底解決負負得正問題
        """
        return np.sign(z1 + z2) * np.abs(z1 * z2)

    def transform_oi_series(self, val_or_series):
        if "Signed-Log" in self.transform_mode:
            if isinstance(val_or_series, pd.Series):
                return np.sign(val_or_series) * np.log1p(np.abs(val_or_series))
            else:
                return float(
                    np.sign(val_or_series) * np.log1p(abs(val_or_series))
                )
        else:
            return val_or_series

    def transform_foreign_net_series(self, val_or_series):
        if "Signed-Log" in self.transform_mode:
            if isinstance(val_or_series, pd.Series):
                return np.sign(val_or_series) * np.log1p(np.abs(val_or_series))
            else:
                return float(
                    np.sign(val_or_series) * np.log1p(abs(val_or_series))
                )
        else:
            return val_or_series

    def transform_sbl_series(self, val_or_series):
        if "Signed-Log" in self.transform_mode:
            if isinstance(val_or_series, pd.Series):
                return np.sign(val_or_series) * np.log1p(np.abs(val_or_series))
            else:
                return float(
                    np.sign(val_or_series) * np.log1p(abs(val_or_series))
                )
        else:
            return val_or_series

    def fit_model_on_sample(
        self,
        train_sample: pd.DataFrame,
        hist_twse: Optional[pd.DataFrame] = None,
        hist_taifex: Optional[pd.DataFrame] = None,
        hist_sbl: Optional[pd.DataFrame] = None,
    ) -> bool:
        """
        專用於傳入之訓練樣本進行 17 變量 Logistic 訓練（對齊波段開盤進出場目標 TXF_Open_{t+1} > TXF_Open_t）：
        - 被解釋變數 (應變數) y_t：當日 (t) 開盤進場持有至次一交易日 (t+1) 開盤之臺指期漲跌方向
          (TXF_Open_{t+1} > TXF_Open_t 為漲=1，否則為跌=0；樣本最末日以當日 TXF_Close_t 代入避免偷看樣本外數據)
        - 解釋變數 (自變數) X_{t-1}：
          * 台股日內、盤後夜盤與籌碼因子嚴格採用 T-1 數據 (.shift(1))；
          * 美股隔夜因子已於資料管線嚴格對齊台灣 T 日開盤前剛收盤之 T-1 美股交易日（us_raw.index < dt）。
        """
        if len(train_sample) < 40:
            return False

        self.fit_sample_size = len(train_sample)
        train_df = train_sample.copy()
        next_txf_open = train_df["TXF_Open"].shift(-1).fillna(train_df["TXF_Close"])
        train_df["y"] = (next_txf_open > train_df["TXF_Open"]).astype(int)  
        #train_df["y"] = (train_df["TXF_Close"] > train_df["TXF_Open"]).astype(int)  
        
        X_raw = pd.DataFrame(index=train_df.index)

        # (A) 隔夜美股與匯率特徵 (已對齊 us_raw.index < dt，即台灣 T 日凌晨 04:00 剛收盤之 T-1 美股)
        for col in ["usdtwd_ret", "sox_ret", "tsm_ret", "nvda_ret", "vix_chg"]:
            if col in train_df.columns:
                X_raw[col] = train_df[col].ffill().bfill().fillna(0.0)
            else:
                X_raw[col] = 0.0

        # (B) 技術分析相關因子區塊：跳空幅度與均線虛擬變數 (嚴格採用 T-1 前一交易日確立數據 .shift(1))
        for tw_lag_col in ["gap_ret", "ma5_bias", "ma10_bias", "ma5_ma10"]:
            if tw_lag_col in train_df.columns:
                X_raw[tw_lag_col] = train_df[tw_lag_col].ffill().shift(1).bfill().fillna(0.0)
            else:
                X_raw[tw_lag_col] = 0.0

        # (B-2) 預測日 (T) 開盤前即已結束之夜盤特徵 (第 10 因子期現貨價差率、第 11 因子夜盤報酬率，不需 shift(1)，遇缺以前一交易日取代)
        for ah_col in ["basis_ret", "txf_ah_ret"]:
            if ah_col in train_df.columns:
                X_raw[ah_col] = train_df[ah_col].ffill().bfill().fillna(0.0)
            else:
                X_raw[ah_col] = 0.0

        # 國安基金進場護盤期間虛擬變數（採用前一交易日 t-1 已確立之護盤狀態）
        nsf_series = pd.Series(
            [float(check_is_nsf_active(d)) for d in train_df.index],
            index=train_df.index,
        )
        X_raw["nsf_active"] = nsf_series.shift(1).bfill()

        # (C) 外資現貨買賣超 (T-1)
        if (
            hist_twse is not None
            and not hist_twse.empty
            and "foreign_net" in hist_twse.columns
        ):
            twse_clean = hist_twse[~hist_twse.index.duplicated(keep="last")].sort_index()
            f_series = twse_clean["foreign_net"].reindex(train_df.index).ffill().shift(1).bfill().fillna(0.0)
        else:
            f_series = pd.Series(0.0, index=train_df.index)

        # (D) 外資期指單日未平倉增減變動量 ΔOI (T-1)
        if (
            hist_taifex is not None
            and not hist_taifex.empty
            and "tx_foreign_oi" in hist_taifex.columns
        ):
            taifex_clean = hist_taifex[~hist_taifex.index.duplicated(keep="last")].sort_index()
            raw_tx_aligned = taifex_clean["tx_foreign_oi"].reindex(train_df.index).ffill()
            tx_diff_series = raw_tx_aligned.diff().shift(1).bfill().fillna(0.0)
        else:
            tx_diff_series = pd.Series(0.0, index=train_df.index)

        # (E) 近 5 日借券賣出餘額累計變動 (張, T-1)
        if (
            hist_sbl is not None
            and not hist_sbl.empty
            and "sbl_balance" in hist_sbl.columns
        ):
            sbl_clean = hist_sbl[~hist_sbl.index.duplicated(keep="last")].sort_index()
            raw_sbl_aligned = sbl_clean["sbl_balance"].reindex(train_df.index).ffill()
            sbl_5d_series = raw_sbl_aligned.diff(5).shift(1).bfill().fillna(0.0)
        else:
            sbl_5d_series = pd.Series(0.0, index=train_df.index)

        X_raw["foreign_net"] = self.transform_foreign_net_series(f_series)
        X_raw["tx_oi_diff"] = self.transform_oi_series(tx_diff_series)
        X_raw["sbl_5d_chg"] = self.transform_sbl_series(sbl_5d_series)

        X_raw = X_raw.dropna()
        y = train_df.loc[X_raw.index, "y"]

        new_scaler = self.default_scaler.copy()
        base_cols_to_scale = [
            "foreign_net",
            "tx_oi_diff",
            "usdtwd_ret",
            "sox_ret",
            "tsm_ret",
            "nvda_ret",
            "vix_chg",
            "gap_ret",
            "basis_ret",
            "txf_ah_ret",
            "sbl_5d_chg",
        ]
        for col in base_cols_to_scale:
            mu = float(X_raw[col].mean())
            sigma = float(X_raw[col].std()) if X_raw[col].std() > 1e-4 else 1.0
            new_scaler[col] = {"mu": mu, "sigma": sigma}

        # 1. 外資期現幾何共振項: sign(Z_f + Z_dOI) * |Z_f * Z_dOI|
        z_f = (
            X_raw["foreign_net"] - new_scaler["foreign_net"]["mu"]
        ) / new_scaler["foreign_net"]["sigma"]
        z_tx = (
            X_raw["tx_oi_diff"] - new_scaler["tx_oi_diff"]["mu"]
        ) / new_scaler["tx_oi_diff"]["sigma"]
        X_raw["fx_inter"] = self.compute_directional_interaction(z_f, z_tx)

        inter_mu = float(X_raw["fx_inter"].mean())
        inter_sigma = (
            float(X_raw["fx_inter"].std())
            if X_raw["fx_inter"].std() > 1e-4
            else 1.0
        )
        new_scaler["fx_inter"] = {"mu": inter_mu, "sigma": inter_sigma}

        # 2. [美股費半]*[台積電ADR] 幾何共振項: sign(Z_sox + Z_tsm) * |Z_sox * Z_tsm|
        z_sox = (
            X_raw["sox_ret"] - new_scaler["sox_ret"]["mu"]
        ) / new_scaler["sox_ret"]["sigma"]
        z_tsm = (
            X_raw["tsm_ret"] - new_scaler["tsm_ret"]["mu"]
        ) / new_scaler["tsm_ret"]["sigma"]
        X_raw["sox_tsm_inter"] = self.compute_directional_interaction(z_sox, z_tsm)

        st_inter_mu = float(X_raw["sox_tsm_inter"].mean())
        st_inter_sigma = (
            float(X_raw["sox_tsm_inter"].std())
            if X_raw["sox_tsm_inter"].std() > 1e-4
            else 1.0
        )
        new_scaler["sox_tsm_inter"] = {
            "mu": st_inter_mu,
            "sigma": st_inter_sigma,
        }

        # 3. 所有虛擬變數 (國安基金與技術分析均線虛擬變數) 固定維持 (0, 1) 原生尺度
        for dummy_col in ["nsf_active", "ma5_bias", "ma10_bias", "ma5_ma10"]:
            new_scaler[dummy_col] = {"mu": 0.0, "sigma": 1.0}

        feature_cols = self.ALL_FEATURES
        X_scaled = pd.DataFrame(index=X_raw.index)
        for col in feature_cols:
            mu = new_scaler[col]["mu"]
            sigma = new_scaler[col]["sigma"]
            X_scaled[col] = (X_raw[col] - mu) / sigma

        X_mat = X_scaled.values
        y_vec = y.values
        n_feats = len(feature_cols)

        # 先計算全 13 因子基準準確率 (Full 13-Factor Baseline)
        full_best_acc = -1.0
        full_best_clf = None
        full_best_c = 1.0
        for c_val in [0.25, 0.5, 1.0, 2.0, 5.0]:
            for cw in [None, "balanced"]:
                clf_try = LogisticRegression(solver="lbfgs", C=c_val, class_weight=cw, max_iter=300)
                clf_try.fit(X_mat, y_vec)
                acc_try = float(clf_try.score(X_mat, y_vec)) * 100.0
                if acc_try > full_best_acc:
                    full_best_acc = acc_try
                    full_best_clf = clf_try
                    full_best_c = c_val
        self.full_13_accuracy = full_best_acc

        if self.auto_select_features:
            # =====================================================================
            # 【自動特徵篩選引擎 (Auto Feature Selection)】：
            # 結合「後向逐步淘汰法 (Backward Elimination)」、「前向逐步選取法 (Forward Selection)」、
            # 「遞迴特徵消除法 (RFE)」與「L1 Lasso 稀疏正則化」，
            # 自動剔除冗餘雜訊變數，僅保留對預測漲跌成功率較有影響之核心解釋變數 (3 ~ 10 個)！
            # 在預測準確率更高（或持平）的前提下，優先選取最精簡之高影響力特徵子集。
            # =====================================================================
            best_acc = -1.0
            best_subset_idx = None
            best_clf = None
            best_c = 1.0
            best_cw = None

            for cw in [None, "balanced"]:
                # (1) 後向逐步淘汰法 (Backward Elimination: 保留 3 ~ 10 個關鍵變數)
                curr_idx = list(range(n_feats))
                while len(curr_idx) > 3:
                    best_step_acc = -1.0
                    best_drop = None
                    for idx in curr_idx:
                        cand_idx = [i for i in curr_idx if i != idx]
                        clf_try = LogisticRegression(solver="lbfgs", C=1.0, class_weight=cw, max_iter=200)
                        clf_try.fit(X_mat[:, cand_idx], y_vec)
                        acc_try = float(clf_try.score(X_mat[:, cand_idx], y_vec)) * 100.0
                        if acc_try > best_step_acc:
                            best_step_acc = acc_try
                            best_drop = idx
                    curr_idx.remove(best_drop)
                    if len(curr_idx) <= 10:
                        if (best_step_acc > best_acc) or (
                            abs(best_step_acc - best_acc) < 1e-9
                            and (best_subset_idx is None or len(curr_idx) < len(best_subset_idx))
                        ):
                            best_acc = best_step_acc
                            best_subset_idx = curr_idx.copy()
                            best_cw = cw

                # (2) 前向逐步選取法 (Forward Stepwise Selection: 由空集合挑選最具預測力之 3 ~ 10 個變數)
                fwd_idx = []
                rem_idx = list(range(n_feats))
                while len(rem_idx) > 0 and len(fwd_idx) < 10:
                    best_step_acc = -1.0
                    best_add = None
                    for idx in rem_idx:
                        cand_idx = fwd_idx + [idx]
                        clf_try = LogisticRegression(solver="lbfgs", C=1.0, class_weight=cw, max_iter=200)
                        clf_try.fit(X_mat[:, cand_idx], y_vec)
                        acc_try = float(clf_try.score(X_mat[:, cand_idx], y_vec)) * 100.0
                        if acc_try > best_step_acc:
                            best_step_acc = acc_try
                            best_add = idx
                    fwd_idx.append(best_add)
                    rem_idx.remove(best_add)
                    if 3 <= len(fwd_idx) <= 10:
                        if (best_step_acc > best_acc) or (
                            abs(best_step_acc - best_acc) < 1e-9
                            and (best_subset_idx is None or len(fwd_idx) < len(best_subset_idx))
                        ):
                            best_acc = best_step_acc
                            best_subset_idx = fwd_idx.copy()
                            best_cw = cw

                # (3) 遞迴特徵消除法 (RFE: 依據標準化 |Beta| 貢獻度逐一剔除最弱變數)
                rfe_idx = list(range(n_feats))
                while len(rfe_idx) >= 3:
                    clf_try = LogisticRegression(solver="lbfgs", C=1.0, class_weight=cw, max_iter=200)
                    clf_try.fit(X_mat[:, rfe_idx], y_vec)
                    acc_try = float(clf_try.score(X_mat[:, rfe_idx], y_vec)) * 100.0
                    if 3 <= len(rfe_idx) <= 10:
                        if (acc_try > best_acc) or (
                            abs(acc_try - best_acc) < 1e-9
                            and (best_subset_idx is None or len(rfe_idx) < len(best_subset_idx))
                        ):
                            best_acc = acc_try
                            best_subset_idx = rfe_idx.copy()
                            best_cw = cw
                    min_pos = int(np.argmin(np.abs(clf_try.coef_[0])))
                    rfe_idx.pop(min_pos)

            # (4) 結合 L1 Lasso 稀疏正則化與超參數 C 精調
            if best_subset_idx is None:
                best_subset_idx = list(range(n_feats))

            candidate_subsets = [best_subset_idx]
            for c_l1 in [0.15, 0.25, 0.5, 1.0]:
                for cw in [None, "balanced"]:
                    clf_l1 = LogisticRegression(
                        penalty="l1", solver="liblinear", C=c_l1, class_weight=cw, max_iter=200
                    )
                    clf_l1.fit(X_mat, y_vec)
                    l1_idx = [i for i, b in enumerate(clf_l1.coef_[0]) if abs(b) > 1e-4]
                    if 3 <= len(l1_idx) <= 10 and l1_idx not in candidate_subsets:
                        candidate_subsets.append(l1_idx)

            best_final_acc = -1.0
            final_subset_idx = best_subset_idx
            for sub_idx in candidate_subsets:
                for c_val in [0.25, 0.5, 1.0, 2.0, 5.0]:
                    for cw in [None, "balanced"]:
                        clf_sub = LogisticRegression(
                            solver="lbfgs", C=c_val, class_weight=cw, max_iter=300
                        )
                        clf_sub.fit(X_mat[:, sub_idx], y_vec)
                        acc_sub = float(clf_sub.score(X_mat[:, sub_idx], y_vec)) * 100.0
                        if (acc_sub > best_final_acc) or (
                            abs(acc_sub - best_final_acc) < 1e-9
                            and len(sub_idx) < len(final_subset_idx)
                        ):
                            best_final_acc = acc_sub
                            final_subset_idx = sub_idx
                            best_clf = clf_sub
                            best_c = c_val

            final_subset_idx = sorted(final_subset_idx)
            # 重新以排序後的欄位順序擬合 best_clf，確保係數順序與欄位嚴格對應
            best_clf.fit(X_mat[:, final_subset_idx], y_vec)
            best_acc = float(best_clf.score(X_mat[:, final_subset_idx], y_vec)) * 100.0

            self.selected_features = [feature_cols[i] for i in final_subset_idx]
            self.eliminated_features = [
                c for c in feature_cols if c not in self.selected_features
            ]

            self.active_scaler = new_scaler
            self.active_intercept = float(best_clf.intercept_[0])
            self.best_c_param = best_c

            new_beta = {col: 0.0 for col in feature_cols}
            for pos_i, col_idx in enumerate(final_subset_idx):
                col_name = feature_cols[col_idx]
                new_beta[col_name] = float(best_clf.coef_[0][pos_i])
            self.active_beta = new_beta
            self.fit_accuracy = best_acc

            probs_in_sample = best_clf.predict_proba(X_mat[:, final_subset_idx])[:, 1]
            self.in_sample_probs = [float(p) for p in probs_in_sample]
            self.in_sample_df = train_df.loc[X_scaled.index].copy()
            return True
        else:
            # 保留全 16 因子模式
            self.selected_features = feature_cols.copy()
            self.eliminated_features = []
            self.active_scaler = new_scaler
            self.active_intercept = float(full_best_clf.intercept_[0])
            self.best_c_param = full_best_c

            new_beta = {}
            for idx, col in enumerate(feature_cols):
                new_beta[col] = float(full_best_clf.coef_[0][idx])
            self.active_beta = new_beta

            self.fit_accuracy = full_best_acc

            probs_in_sample = full_best_clf.predict_proba(X_mat)[:, 1]
            self.in_sample_probs = [float(p) for p in probs_in_sample]
            self.in_sample_df = train_df.loc[X_scaled.index].copy()
            return True

    def reset_to_static(
        self,
        train_sample: Optional[pd.DataFrame] = None,
        hist_twse: Optional[pd.DataFrame] = None,
        hist_taifex: Optional[pd.DataFrame] = None,
        hist_sbl: Optional[pd.DataFrame] = None,
    ):
        self.active_scaler = self.default_scaler.copy()
        self.active_beta = self.default_beta.copy()
        self.active_intercept = self.default_intercept
        self.selected_features = self.ALL_FEATURES.copy()
        self.eliminated_features = []
        self.full_13_accuracy = None
        self.fit_accuracy = None
        self.fit_sample_size = 0
        self.in_sample_probs = []
        self.in_sample_df = pd.DataFrame()

        if train_sample is not None and len(train_sample) >= 40:
            train_df = train_sample.copy()
            next_txf_open = train_df["TXF_Open"].shift(-1).fillna(train_df["TXF_Close"])
            train_df["y"] = (next_txf_open > train_df["TXF_Open"]).astype(int)

            X_raw = pd.DataFrame(index=train_df.index)
            for col in ["usdtwd_ret", "sox_ret", "tsm_ret", "nvda_ret", "vix_chg"]:
                X_raw[col] = train_df[col].ffill().bfill().fillna(0.0) if col in train_df.columns else 0.0
            for tw_lag_col in ["gap_ret", "ma5_bias", "ma10_bias", "ma5_ma10"]:
                X_raw[tw_lag_col] = train_df[tw_lag_col].ffill().shift(1).bfill().fillna(0.0) if tw_lag_col in train_df.columns else 0.0
            for ah_col in ["basis_ret", "txf_ah_ret"]:
                X_raw[ah_col] = train_df[ah_col].ffill().bfill().fillna(0.0) if ah_col in train_df.columns else 0.0

            nsf_series = pd.Series([float(check_is_nsf_active(d)) for d in train_df.index], index=train_df.index)
            X_raw["nsf_active"] = nsf_series.shift(1).bfill()

            if hist_twse is not None and not hist_twse.empty and "foreign_net" in hist_twse.columns:
                twse_clean = hist_twse[~hist_twse.index.duplicated(keep="last")].sort_index()
                f_series = twse_clean["foreign_net"].reindex(train_df.index).ffill().shift(1).bfill().fillna(0.0)
            else:
                f_series = pd.Series(0.0, index=train_df.index)

            if hist_taifex is not None and not hist_taifex.empty and "tx_foreign_oi" in hist_taifex.columns:
                taifex_clean = hist_taifex[~hist_taifex.index.duplicated(keep="last")].sort_index()
                raw_tx_aligned = taifex_clean["tx_foreign_oi"].reindex(train_df.index).ffill()
                tx_diff_series = raw_tx_aligned.diff().shift(1).bfill().fillna(0.0)
            else:
                tx_diff_series = pd.Series(0.0, index=train_df.index)

            if hist_sbl is not None and not hist_sbl.empty and "sbl_balance" in hist_sbl.columns:
                sbl_clean = hist_sbl[~hist_sbl.index.duplicated(keep="last")].sort_index()
                raw_sbl_aligned = sbl_clean["sbl_balance"].reindex(train_df.index).ffill()
                sbl_5d_series = raw_sbl_aligned.diff(5).shift(1).bfill().fillna(0.0)
            else:
                sbl_5d_series = pd.Series(0.0, index=train_df.index)

            X_raw["foreign_net"] = self.transform_foreign_net_series(f_series)
            X_raw["tx_oi_diff"] = self.transform_oi_series(tx_diff_series)
            X_raw["sbl_5d_chg"] = self.transform_sbl_series(sbl_5d_series)
            X_raw = X_raw.dropna()
            y = train_df.loc[X_raw.index, "y"]

            z_f = (X_raw["foreign_net"] - self.active_scaler["foreign_net"]["mu"]) / self.active_scaler["foreign_net"]["sigma"]
            z_tx = (X_raw["tx_oi_diff"] - self.active_scaler["tx_oi_diff"]["mu"]) / self.active_scaler["tx_oi_diff"]["sigma"]
            X_raw["fx_inter"] = self.compute_directional_interaction(z_f, z_tx)

            z_sox = (X_raw["sox_ret"] - self.active_scaler["sox_ret"]["mu"]) / self.active_scaler["sox_ret"]["sigma"]
            z_tsm = (X_raw["tsm_ret"] - self.active_scaler["tsm_ret"]["mu"]) / self.active_scaler["tsm_ret"]["sigma"]
            X_raw["sox_tsm_inter"] = self.compute_directional_interaction(z_sox, z_tsm)

            z_total = pd.Series(self.active_intercept, index=X_raw.index)
            for col in self.ALL_FEATURES:
                mu = self.active_scaler[col]["mu"]
                sigma = self.active_scaler[col]["sigma"] if self.active_scaler[col]["sigma"] > 0 else 1.0
                z_total += self.active_beta[col] * ((X_raw[col] - mu) / sigma)

            probs = 1.0 / (1.0 + np.exp(-z_total))
            preds = (probs >= 0.5).astype(int)
            acc = float((preds == y).mean()) * 100.0
            self.fit_sample_size = len(X_raw)
            self.fit_accuracy = acc
            self.full_13_accuracy = acc
            self.in_sample_probs = [float(p) for p in probs]
            self.in_sample_df = train_df.loc[X_raw.index].copy()

    def predict(self, raw_features: Dict[str, float]) -> Tuple[float, dict]:
        feats = raw_features.copy()
        raw_f = feats.get("foreign_net", 0.0)
        raw_tx_diff = feats.get("tx_oi_diff", feats.get("tx_foreign_oi", 0.0))

        transformed_f = self.transform_foreign_net_series(raw_f)
        transformed_tx_diff = self.transform_oi_series(raw_tx_diff)

        raw_sbl_5d = feats.get("sbl_5d_chg", 0.0)
        transformed_sbl_5d = self.transform_sbl_series(raw_sbl_5d)

        feats["foreign_net"] = transformed_f
        feats["tx_oi_diff"] = transformed_tx_diff
        feats["sbl_5d_chg"] = transformed_sbl_5d

        if "fx_inter" not in feats:
            mu_f = self.active_scaler.get("foreign_net", {}).get("mu", 0.0)
            sig_f = self.active_scaler.get("foreign_net", {}).get("sigma", 1.0)
            mu_tx = self.active_scaler.get("tx_oi_diff", {}).get("mu", 0.0)
            sig_tx = self.active_scaler.get("tx_oi_diff", {}).get("sigma", 1.0)
            z_f = (transformed_f - mu_f) / (sig_f if sig_f > 0 else 1.0)
            z_tx = (transformed_tx_diff - mu_tx) / (sig_tx if sig_tx > 0 else 1.0)
            feats["fx_inter"] = float(self.compute_directional_interaction(z_f, z_tx))

        if "sox_tsm_inter" not in feats:
            raw_sox = feats.get("sox_ret", 0.0)
            raw_tsm = feats.get("tsm_ret", 0.0)
            mu_sox = self.active_scaler.get("sox_ret", {}).get("mu", 0.0)
            sig_sox = self.active_scaler.get("sox_ret", {}).get("sigma", 1.0)
            mu_tsm = self.active_scaler.get("tsm_ret", {}).get("mu", 0.0)
            sig_tsm = self.active_scaler.get("tsm_ret", {}).get("sigma", 1.0)
            z_sox = (raw_sox - mu_sox) / (sig_sox if sig_sox > 0 else 1.0)
            z_tsm = (raw_tsm - mu_tsm) / (sig_tsm if sig_tsm > 0 else 1.0)
            feats["sox_tsm_inter"] = float(self.compute_directional_interaction(z_sox, z_tsm))

        for k_def in ["gap_ret", "basis_ret", "txf_ah_ret", "ma5_bias", "ma10_bias", "ma5_ma10", "vix_chg", "nsf_active", "nvda_ret", "sbl_5d_chg"]:
            if k_def not in feats:
                feats[k_def] = 0.0

        z = self.active_intercept
        contributions = {}

        for k in self.ALL_FEATURES:
            weight = self.active_beta.get(k, 0.0)
            val = feats.get(k, 0.0)
            mu = self.active_scaler.get(k, {}).get("mu", 0.0)
            sigma = self.active_scaler.get(k, {}).get("sigma", 1.0)
            sigma = sigma if sigma > 0 else 1.0
            z_score = (val - mu) / sigma
            contrib = weight * z_score
            contributions[k] = contrib
            z += contrib

        prob_up = 1.0 / (1.0 + np.exp(-z))
        return prob_up, contributions



# -----------------------------------------------------------------------------
# 3. 臺指期波段留倉會計結算與門檻尋優 (訊號一致則留倉，不連貫時以預測日開盤價平倉；僅停損、不停利)
# -----------------------------------------------------------------------------
def simulate_day_trading_pnl(
    probs_pct: list,
    eval_df: pd.DataFrame,
    threshold_bull: int,
    threshold_bear: int,
    cost_rate_pct: float,
    stop_loss_pct: float = 0.0,
) -> Tuple[list, list, list, list, list, list, int, int]:
    signal_types = []
    positions = []
    gross_returns = []
    friction_costs = []
    net_returns = []
    validation_status = []
    active_signals = 0
    correct_signals = 0

    open_prices = eval_df["TXF_Open"].tolist()
    close_prices = eval_df["TXF_Close"].tolist()
    high_prices = (
        eval_df["TXF_High"].tolist()
        if "TXF_High" in eval_df.columns
        else [max(o, c) for o, c in zip(open_prices, close_prices)]
    )
    low_prices = (
        eval_df["TXF_Low"].tolist()
        if "TXF_Low" in eval_df.columns
        else [min(o, c) for o, c in zip(open_prices, close_prices)]
    )
    n = len(probs_pct)

    prev_pos = 0
    prev_cp = np.nan
    entry_price = np.nan

    for i in range(n):
        p = probs_pct[i]
        op = open_prices[i]
        cp = close_prices[i]
        hp = high_prices[i] if not np.isnan(high_prices[i]) else max(op, cp)
        lp = low_prices[i] if not np.isnan(low_prices[i]) else min(op, cp)
        is_pending = np.isnan(cp) or np.isnan(op)

        # 【進出場雙門檻緩衝 (Hysteresis)】：
        # - 新進場維持嚴格門檻 (多方 p >= threshold_bull，空方 p <= threshold_bear)
        # - 續抱緩衝：前一日已持多單 (prev_pos == 1) 時，只要 p >= 50% 即續抱；前一日已持空單 (prev_pos == -1) 時，只要 p <= 50% 即續抱
        if p >= threshold_bull:
            sig_base = "📈 多方訊號"
            signal_pos = 1
        elif p <= threshold_bear:
            sig_base = "📉 空方訊號"
            signal_pos = -1
        elif prev_pos == 1 and p >= 50.0:
            sig_base = "📈 多方訊號 (緩衝續抱)"
            signal_pos = 1
        elif prev_pos == -1 and p <= 50.0:
            sig_base = "📉 空方訊號 (緩衝續抱)"
            signal_pos = -1
        else:
            sig_base = "⚪ 觀望訊號"
            signal_pos = 0

        # ... 下接既有的部位結算、開盤平倉與純停損檢核邏輯 (保持不變) ...
        if is_pending:
            gross_ret = 0.0
            cost = 0.0
            net_ret = 0.0
            end_pos = 0
            sig = f"{sig_base} (待結算)"
            status = "⏳ 待當日開收盤結算"
        else:
            gross_ret = 0.0
            cost = 0.0
            status_parts = []
            end_pos = signal_pos

            # (A) 處理前期留倉部位在當日 (預測日 i) 開盤之平倉或續抱隔夜段 (Close_{i-1} -> Open_i)
            if prev_pos != 0 and not np.isnan(prev_cp) and prev_cp > 0:
                if prev_pos == 1:
                    overnight_ret = ((op - prev_cp) / prev_cp) * 100.0
                else:
                    overnight_ret = ((prev_cp - op) / prev_cp) * 100.0
                gross_ret += overnight_ret

                if signal_pos != prev_pos:
                    # 預測訊號不連貫：於預測日開盤價平倉 (扣除平倉單趟成本)
                    cost += cost_rate_pct
                    dir_str = "多" if prev_pos == 1 else "空"
                    status_parts.append(f"🔄 開盤平{dir_str}倉({op:,.1f}|隔夜{overnight_ret:+.2f}%)")
                    entry_price = np.nan
                else:
                    # 買賣訊號一致：繼續留倉 (無平倉成本)
                    dir_str = "多" if prev_pos == 1 else "空"
                    status_parts.append(f"🔒 {dir_str}單留倉續抱(隔夜{overnight_ret:+.2f}%)")

            # (B) 處理當日 (預測日 i) 訊號之新進場或續抱盤中段，並檢核「純停損機制（不停利）」
            if signal_pos != 0:
                is_new_entry = (signal_pos != prev_pos)
                if is_new_entry:
                    cost += cost_rate_pct
                    entry_price = op

                # 計算當日盤中最不利價位 (Worst Excursion) 下相對於建倉價之累計報酬率與單日盤中虧損率
                if signal_pos == 1:
                    worst_cum_ret = ((lp - entry_price) / entry_price) * 100.0
                    intraday_worst_ret = ((lp - op) / op) * 100.0
                else:
                    worst_cum_ret = ((entry_price - hp) / entry_price) * 100.0
                    intraday_worst_ret = ((op - hp) / op) * 100.0

                # 【純停損檢核（絕不停利）】：僅當累計損益虧損達 -stop_loss_pct% (或當日盤中虧損達 -stop_loss_pct%) 時才啟動停損平倉！
                if stop_loss_pct > 0 and (
                    worst_cum_ret <= -stop_loss_pct + 1e-9
                    or intraday_worst_ret <= -stop_loss_pct + 1e-9
                ):
                    if worst_cum_ret <= -stop_loss_pct + 1e-9:
                        if signal_pos == 1:
                            stop_price_cum = entry_price * (1.0 - stop_loss_pct / 100.0)
                            stop_price_day = op * (1.0 - stop_loss_pct / 100.0)
                            stop_price = max(stop_price_cum, stop_price_day)
                            # 若開盤跳空已直接跌破停損價，則於開盤價即刻停損
                            stop_price = min(stop_price, op)
                        else:
                            stop_price_cum = entry_price * (1.0 + stop_loss_pct / 100.0)
                            stop_price_day = op * (1.0 + stop_loss_pct / 100.0)
                            stop_price = min(stop_price_cum, stop_price_day)
                            stop_price = max(stop_price, op)
                    else:
                        if signal_pos == 1:
                            stop_price = op * (1.0 - stop_loss_pct / 100.0)
                        else:
                            stop_price = op * (1.0 + stop_loss_pct / 100.0)

                    if signal_pos == 1:
                        intraday_ret = ((stop_price - op) / op) * 100.0
                    else:
                        intraday_ret = ((op - stop_price) / op) * 100.0

                    gross_ret += intraday_ret
                    # 觸發停損平倉，加計出場單趟摩擦成本
                    cost += cost_rate_pct
                    dir_str = "多" if signal_pos == 1 else "空"

                    if is_new_entry:
                        act_prefix = "⚡ 反手進" if prev_pos != 0 else "⚡ 開盤進"
                        status_parts.append(
                            f"{act_prefix}{dir_str}({op:,.1f}) ➔ 🛡️觸及-{stop_loss_pct:.1f}%停損平倉({stop_price:,.1f}|當日{intraday_ret:+.2f}%)"
                        )
                    else:
                        status_parts.append(
                            f"🛡️ 觸及-{stop_loss_pct:.1f}%即刻停損平倉({op:,.1f}→{stop_price:,.1f}|當日{intraday_ret:+.2f}%)"
                        )

                    sig = f"{sig_base} (🛡️-{stop_loss_pct:.1f}%停損平倉)"
                    # 盤中已停損出場，收盤部位歸零
                    end_pos = 0
                    entry_price = np.nan
                else:
                    # 未觸發停損（且策略不設停利），正常持有至收盤讓獲利奔跑
                    if signal_pos == 1:
                        intraday_ret = ((cp - op) / op) * 100.0
                    else:
                        intraday_ret = ((op - cp) / op) * 100.0

                    gross_ret += intraday_ret

                    if is_new_entry:
                        dir_str = "多" if signal_pos == 1 else "空"
                        act_prefix = "⚡ 反手進" if prev_pos != 0 else "⚡ 開盤進"
                        status_parts.append(f"{act_prefix}{dir_str}倉({op:,.1f}→{cp:,.1f}|盤中{intraday_ret:+.2f}%)")
                        sig = f"{sig_base} (反手留倉)" if prev_pos != 0 else f"{sig_base} (新進留倉)"
                    else:
                        status_parts.append(f"盤中續抱({op:,.1f}→{cp:,.1f}|{intraday_ret:+.2f}%)")
                        sig = f"{sig_base} (一致續抱)"
                    end_pos = signal_pos
            else:
                if prev_pos != 0:
                    sig = "⚪ 訊號中斷 (開盤平倉)"
                else:
                    sig = "⚪ 空手觀望"
                    status_parts.append("⚪ 空手無部位")
                end_pos = 0

            net_ret = gross_ret - cost
            status = " ｜ ".join(status_parts)

            # 只要當日持有部位、盤中停損或執行開盤平倉，皆計入有效參與交易日
            if signal_pos != 0 or prev_pos != 0:
                active_signals += 1
                if net_ret > 0:
                    correct_signals += 1

            prev_pos = end_pos
            prev_cp = cp

        signal_types.append(sig)
        positions.append(end_pos if not is_pending else signal_pos)
        gross_returns.append(gross_ret)
        friction_costs.append(cost)
        net_returns.append(net_ret)
        validation_status.append(status)

    return (
        signal_types,
        positions,
        gross_returns,
        friction_costs,
        net_returns,
        validation_status,
        active_signals,
        correct_signals,
    )


def extract_day_trades(
    dates: list,
    open_prices: list,
    close_prices: list,
    positions: list,
    gross_returns: list,
    friction_costs: list,
    net_returns: list,
    high_prices: Optional[list] = None,
    low_prices: Optional[list] = None,
    signal_types: Optional[list] = None,
    stop_loss_pct: float = 0.0,
) -> pd.DataFrame:
    """
    擷取波段留倉逐筆完整交易明細（純停損、不停利機制）：
    - 訊號觸發當日以開盤價 (TXF Open) 建倉；
    - 若後續交易日買賣訊號一致且未觸發停損則留倉續抱（累計持有天數，不設停利讓獲利奔跑）；
    - 若當日部位虧損達 stop_loss_pct% 以上，即刻於停損觸發價位執行「停損平倉」；
    - 當預測訊號不連貫時，以該預測日之開盤價 (TXF Open) 正式平倉結算（若至樣本最後一日仍持倉，則以期末收盤價標記為留倉進行中）。
    """
    trades = []
    n = len(positions)
    active_trade = None
    single_cost = next((c for c in friction_costs if 0 < c <= 0.045), 0.03)

    for i in range(n):
        pos = positions[i]
        d = dates[i]
        op = open_prices[i]
        cp = close_prices[i]
        if np.isnan(op) or np.isnan(cp):
            continue

        hp = high_prices[i] if (high_prices is not None and i < len(high_prices) and not np.isnan(high_prices[i])) else max(op, cp)
        lp = low_prices[i] if (low_prices is not None and i < len(low_prices) and not np.isnan(low_prices[i])) else min(op, cp)
        sig_str = signal_types[i] if (signal_types is not None and i < len(signal_types)) else ""

        if "多方訊號" in sig_str:
            day_sig_dir = 1
        elif "空方訊號" in sig_str:
            day_sig_dir = -1
        else:
            day_sig_dir = pos

        is_intraday_stopped = ("停損平倉" in sig_str)

        # (1) 若前一日有留倉部位，且當日訊號方向改變，先於當日開盤價平倉前筆交易
        if active_trade is not None and day_sig_dir != active_trade["dir"]:
            exit_price = op
            entry_p = active_trade["entry_price"]
            t_dir = active_trade["dir"]
            point_pnl = (exit_price - entry_p) if t_dir == 1 else (entry_p - exit_price)
            g_r = (point_pnl / entry_p) * 100.0
            c_total = single_cost * 2.0
            n_r = g_r - c_total

            trades.append(
                {
                    "交易序號": f"#{len(trades) + 1}",
                    "進場日": active_trade["entry_date"],
                    "平倉日(開盤/停損)": f"{d} (開盤平倉)",
                    "方向": "🟢 波段做多" if t_dir == 1 else "🔴 波段放空",
                    "持有天數": f"{active_trade['hold_days']} 日",
                    "進場價 (TXF Open)": f"{entry_p:,.1f}",
                    "平倉價 (結算價位)": f"{exit_price:,.1f}",
                    "點數損益": f"{point_pnl:+.1f} 點",
                    "波段毛報酬": f"{g_r:+.2f}%",
                    "摩擦成本": f"-{c_total:.3f}%",
                    "實質淨損益": f"{n_r:+.2f}%",
                    "交易結果": "🏆 獲利平倉" if n_r > 0 else "🌧️ 虧損平倉",
                    "_net_val": n_r,
                }
            )
            active_trade = None

        # (2) 若目前無持倉且當日有進場訊號 (day_sig_dir != 0)，建立新部位
        if active_trade is None and day_sig_dir != 0:
            active_trade = {
                "dir": day_sig_dir,
                "entry_date": d,
                "entry_price": op,
                "hold_days": 1,
            }
        elif active_trade is not None and day_sig_dir == active_trade["dir"]:
            active_trade["hold_days"] += 1

        # (3) 檢查當日是否觸發盤中停損平倉
        if active_trade is not None and (is_intraday_stopped or pos == 0) and stop_loss_pct > 0:
            entry_p = active_trade["entry_price"]
            t_dir = active_trade["dir"]

            if t_dir == 1:
                worst_cum = ((lp - entry_p) / entry_p) * 100.0
            else:
                worst_cum = ((entry_p - hp) / entry_p) * 100.0

            if worst_cum <= -stop_loss_pct + 1e-9:
                if t_dir == 1:
                    stop_p = max(entry_p * (1.0 - stop_loss_pct / 100.0), op * (1.0 - stop_loss_pct / 100.0))
                    stop_p = min(stop_p, op)
                else:
                    stop_p = min(entry_p * (1.0 + stop_loss_pct / 100.0), op * (1.0 + stop_loss_pct / 100.0))
                    stop_p = max(stop_p, op)
            else:
                stop_p = op * (1.0 - stop_loss_pct / 100.0) if t_dir == 1 else op * (1.0 + stop_loss_pct / 100.0)

            point_pnl = (stop_p - entry_p) if t_dir == 1 else (entry_p - stop_p)
            g_r = (point_pnl / entry_p) * 100.0
            c_total = single_cost * 2.0
            n_r = g_r - c_total

            trades.append(
                {
                    "交易序號": f"#{len(trades) + 1}",
                    "進場日": active_trade["entry_date"],
                    "平倉日(開盤/停損)": f"{d} (盤中-{stop_loss_pct:.1f}%停損)",
                    "方向": "🟢 波段做多" if t_dir == 1 else "🔴 波段放空",
                    "持有天數": f"{active_trade['hold_days']} 日",
                    "進場價 (TXF Open)": f"{entry_p:,.1f}",
                    "平倉價 (結算價位)": f"{stop_p:,.1f} (停損觸發價)",
                    "點數損益": f"{point_pnl:+.1f} 點",
                    "波段毛報酬": f"{g_r:+.2f}%",
                    "摩擦成本": f"-{c_total:.3f}%",
                    "實質淨損益": f"{n_r:+.2f}%",
                    "交易結果": f"🛡️ 觸及-{stop_loss_pct:.1f}%停損",
                    "_net_val": n_r,
                }
            )
            active_trade = None

    # 若至樣本最後一日仍有未平倉部位，以期末收盤價進行帳面評價
    if active_trade is not None and n > 0:
        last_cp = close_prices[-1]
        last_d = dates[-1]
        entry_p = active_trade["entry_price"]
        t_dir = active_trade["dir"]
        point_pnl = (last_cp - entry_p) if t_dir == 1 else (entry_p - last_cp)
        g_r = (point_pnl / entry_p) * 100.0
        c_open = single_cost
        n_r = g_r - c_open

        trades.append(
            {
                "交易序號": f"#{len(trades) + 1}",
                "進場日": active_trade["entry_date"],
                "平倉日(開盤/停損)": f"{last_d} (留倉中)",
                "方向": "🟢 波段做多" if t_dir == 1 else "🔴 波段放空",
                "持有天數": f"{active_trade['hold_days']} 日 (續抱)",
                "進場價 (TXF Open)": f"{entry_p:,.1f}",
                "平倉價 (結算價位)": f"{last_cp:,.1f} (期末收盤評價)",
                "點數損益": f"{point_pnl:+.1f} 點",
                "波段毛報酬": f"{g_r:+.2f}%",
                "摩擦成本": f"-{c_open:.3f}%",
                "實質淨損益": f"{n_r:+.2f}%",
                "交易結果": "⏳ 留倉浮盈" if n_r > 0 else "⏳ 留倉浮虧",
                "_net_val": n_r,
            }
        )

    return pd.DataFrame(trades)


def optimize_thresholds_fixed_days(
    probs_pct: list,
    eval_df: pd.DataFrame,
    acc_pct: float,
    bull_range: Tuple[int, int] = (60, 75),
    bear_range: Tuple[int, int] = (25, 40),
    step: int = 1,
    cost_rate_pct: float = 0.03,
    enable_stop_loss: bool = True,
) -> Tuple[int, int, float, float, float, float, pd.DataFrame, pd.DataFrame]:
    """
    樣本內門檻優化引擎（全面採取波段操作）：
    - 遍歷所有候選門檻組合 (多方 60%~75%, 空方 25%~40%)
    - 逐格執行波段留倉結算 (extract_day_trades)
    - 優化目標評分：樣本內 [波段累計淨報酬] × [波段實質勝率]
    - 透過 3×3 九宮格總和搜尋高原穩健最佳中心點
    """
    bull_steps = list(range(bull_range[0], bull_range[1] + 1, step))
    bear_steps = list(range(bear_range[0], bear_range[1] + 1, step))
    candidate_stops = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] if enable_stop_loss else [0.0]

    best_bull = 60
    best_bear = 40
    best_stop_loss_pct = 2.0 if enable_stop_loss else 0.0
    best_net_ret = 0.0
    best_3x3_sum_score = -999999.0
    best_score_matrix = pd.DataFrame()
    best_text_matrix = pd.DataFrame()

    dates_list = [d.strftime("%Y-%m-%d") for d in eval_df.index]
    open_list = eval_df["TXF_Open"].tolist()
    close_list = eval_df["TXF_Close"].tolist()
    high_list = eval_df["TXF_High"].tolist() if "TXF_High" in eval_df.columns else open_list
    low_list = eval_df["TXF_Low"].tolist() if "TXF_Low" in eval_df.columns else close_list

    for sl_pct in candidate_stops:
        raw_score_matrix = pd.DataFrame(index=bull_steps, columns=bear_steps, dtype=float)
        net_ret_matrix = pd.DataFrame(index=bull_steps, columns=bear_steps, dtype=float)
        text_matrix = pd.DataFrame(index=bull_steps, columns=bear_steps, dtype=str)

        # 第一階段：計算該停損百分比下每個儲存格的 [波段累計淨報酬 × 波段勝率]
        for bull in bull_steps:
            for bear in bear_steps:
                sigs, positions, gross_rets, costs, net_rets, _, active, correct = simulate_day_trading_pnl(
                    probs_pct, eval_df, bull, bear, cost_rate_pct,
                    stop_loss_pct=sl_pct,
                )
                total_net = sum(net_rets)

                # 提取完整波段交易明細並計算波段勝率
                swing_df = extract_day_trades(
                    dates_list, open_list, close_list, positions,
                    gross_rets, costs, net_rets,
                    high_prices=high_list, low_prices=low_list,
                    signal_types=sigs, stop_loss_pct=sl_pct
                )
                n_swings = len(swing_df)
                win_swings = int((swing_df["_net_val"] > 0).sum()) if (n_swings > 0 and "_net_val" in swing_df.columns) else 0
                swing_win_rate = (win_swings / n_swings * 100.0) if n_swings > 0 else 0.0
                # swing_win_dec = (win_swings / n_swings) if n_swings > 0 else 0.0

                # 優化目標：極大化 
                if n_swings < 10:
                    score = -999.0
                else:
                    # 強烈偏好 75% 以上勝率，給予大幅權重加成
                    win_weight = 2.0 if swing_win_rate >= 75.0 else 1.0
                    score = (swing_win_rate * win_weight) + (total_net * 0.1)

                raw_score_matrix.loc[bull, bear] = score
                net_ret_matrix.loc[bull, bear] = total_net
                text_matrix.loc[bull, bear] = (
                    f"評分:{score:+.1f}<br>淨:{total_net:+.1f}%<br>(波段勝:{swing_win_rate:.0f}%|{n_swings}筆)"
                )

        # 第二階段：搜尋 3x3 九宮格總和最大之穩健中心點
        n_rows = len(bull_steps)
        n_cols = len(bear_steps)
        score_matrix = raw_score_matrix.round(2).copy()

        sl_best_bull = 60
        sl_best_bear = 40
        sl_best_net_ret = 0.0
        sl_best_3x3_sum = -999999.0

        if n_rows >= 3 and n_cols >= 3:
            for i in range(1, n_rows - 1):
                for j in range(1, n_cols - 1):
                    bull_c = bull_steps[i]
                    bear_c = bear_steps[j]
                    block_3x3 = raw_score_matrix.iloc[i - 1 : i + 2, j - 1 : j + 2]
                    block_sum = float(block_3x3.values.sum())

                    orig_txt = text_matrix.loc[bull_c, bear_c]
                    text_matrix.loc[bull_c, bear_c] = f"3x3合:{block_sum:+.1f}<br>{orig_txt}"

                    if (block_sum > sl_best_3x3_sum) or (
                        abs(block_sum - sl_best_3x3_sum) < 1e-9
                        and raw_score_matrix.loc[bull_c, bear_c] > raw_score_matrix.loc[sl_best_bull, sl_best_bear]
                    ):
                        sl_best_3x3_sum = block_sum
                        sl_best_bull = bull_c
                        sl_best_bear = bear_c
                        sl_best_net_ret = float(net_ret_matrix.loc[bull_c, bear_c])
        else:
            for bull in bull_steps:
                for bear in bear_steps:
                    sc = float(raw_score_matrix.loc[bull, bear])
                    if sc > sl_best_3x3_sum:
                        sl_best_3x3_sum = sc
                        sl_best_bull = bull
                        sl_best_bear = bear
                        sl_best_net_ret = float(net_ret_matrix.loc[bull, bear])

        if (sl_best_3x3_sum > best_3x3_sum_score) or (
            abs(sl_best_3x3_sum - best_3x3_sum_score) < 1e-9 and sl_best_net_ret > best_net_ret
        ):
            best_3x3_sum_score = sl_best_3x3_sum
            best_bull = sl_best_bull
            best_bear = sl_best_bear
            best_stop_loss_pct = sl_pct
            best_net_ret = sl_best_net_ret
            best_score_matrix = score_matrix.copy()
            best_text_matrix = text_matrix.copy()

    return (
        best_bull,
        best_bear,
        best_stop_loss_pct,
        acc_pct,
        best_net_ret,
        best_3x3_sum_score,
        best_score_matrix,
        best_text_matrix,
    )

def render_joint_threshold_heatmap(
    score_matrix: pd.DataFrame,
    text_matrix: pd.DataFrame,
    best_bull: int,
    best_bear: int,
    sample_days: int = 490,
    best_stop_loss_pct: float = 0.0,
):
    annot_text = text_matrix.copy()
    if best_bull in annot_text.index and best_bear in annot_text.columns:
        orig = annot_text.loc[best_bull, best_bear]
        annot_text.loc[best_bull, best_bear] = f"<b>⭐ 3x3最佳中心</b><br>{orig}"

    x_labels = [f"空方 {c}%" for c in score_matrix.columns]
    y_labels = [f"多方 {r}%" for r in score_matrix.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=score_matrix.values,
            x=x_labels,
            y=y_labels,
            text=annot_text.values,
            texttemplate="%{text}",
            textfont={"size": 9, "color": "#f8fafc"},
            colorscale=[
                [0.0, "#0f172a"],
                [0.35, "#1e293b"],
                [0.55, "#3b82f6"],
                [0.75, "#10b981"],
                [1.0, "#f43f5e"],
            ],
            colorbar=dict(
                title=dict(
                    text="<b>評分 [淨利×波段勝率]</b>",
                    font=dict(color="#f8fafc"),
                ),
                tickfont=dict(color="#94a3b8"),
            ),
            hoverongaps=False,
        )
    )

    if best_bull in score_matrix.index and best_bear in score_matrix.columns:
        row_idx = list(score_matrix.index).index(best_bull)
        col_idx = list(score_matrix.columns).index(best_bear)
        fig.add_shape(
            type="rect",
            x0=max(col_idx - 1.5, -0.5),
            x1=min(col_idx + 1.5, len(score_matrix.columns) - 0.5),
            y0=max(row_idx - 1.5, -0.5),
            y1=min(row_idx + 1.5, len(score_matrix.index) - 0.5),
            line=dict(color="#fbbf24", width=3, dash="dash"),
            fillcolor="rgba(251, 191, 36, 0.08)",
        )

    sl_title_tag = f" | 最佳停損: -{best_stop_loss_pct:.1f}% (不停利)" if best_stop_loss_pct > 0 else " | 無停損 (不停利)"
    fig.update_layout(
        title=f"<b>前 {sample_days} 交易日訓練樣本門檻優化熱力圖 (目標: 3×3 九宮格內 Σ[波段淨利 × 波段勝率] 最大之中心點{sl_title_tag})</b>",
        xaxis=dict(
            title="<b>空方波段防禦發動信心門檻 (P(Up) ≤ 門檻)</b>",
            gridcolor="#334155",
        ),
        yaxis=dict(
            title="<b>多方波段進攻發動信心門檻 (P(Up) ≥ 門檻)</b>",
            gridcolor="#334155",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_gap_scatter(
    dates: list,
    gaps: list,
    intraday_rets: list,
    filled_status: list,
):
    clean_data = []
    for d, g, r, f in zip(dates, gaps, intraday_rets, filled_status):
        try:
            g_float = float(g)
            r_float = float(r)
            if not np.isnan(g_float) and not np.isnan(r_float):
                clean_data.append((str(d), g_float, r_float, bool(f)))
        except (ValueError, TypeError):
            continue

    fig = go.Figure()

    if len(clean_data) < 3:
        fig.add_annotation(
            text="⚠️ 有效跳空與盤中走勢樣本不足（需至少 3 筆），暫無法繪製散布圖",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#94a3b8"),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    d_list = [item[0] for item in clean_data]
    g_arr = np.array([item[1] for item in clean_data], dtype=float)
    r_arr = np.array([item[2] for item in clean_data], dtype=float)
    f_list = [item[3] for item in clean_data]

    point_colors = ["#f43f5e" if r >= 0 else "#10b981" for r in r_arr]
    hover_texts = [
        f"<b>交易日:</b> {d}<br>"
        f"<b>開盤跳空 (Gap):</b> {g:+.2f}%<br>"
        f"<b>盤中走勢 (O to C):</b> {r:+.2f}%<br>"
        f"<b>缺口回補:</b> {'✅ 完全回補' if fl else '❌ 未回補'}"
        for d, g, r, fl in zip(d_list, g_arr, r_arr, f_list)
    ]

    fig.add_trace(
        go.Scatter(
            x=g_arr,
            y=r_arr,
            mode="markers+text",
            marker=dict(
                size=12,
                color=point_colors,
                line=dict(width=1.5, color="#ffffff"),
                opacity=0.85,
            ),
            text=[f"{d[-5:]}" for d in d_list],
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text",
            name="歷史交易日",
        )
    )

    trend_desc = "震盪平緩"
    if np.std(g_arr) > 1e-6 and np.std(r_arr) > 1e-6:
        try:
            slope, intercept = np.polyfit(g_arr, r_arr, 1)
            corr_mat = np.corrcoef(g_arr, r_arr)
            correlation = (
                corr_mat[0, 1] if not np.isnan(corr_mat[0, 1]) else 0.0
            )
            r_squared = correlation**2

            x_line = np.linspace(min(g_arr) - 0.2, max(g_arr) + 0.2, 50)
            y_line = slope * x_line + intercept

            trend_desc = "順勢延續動能" if slope > 0 else "均值回歸 (Fade the Gap)"
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="#38bdf8", width=2, dash="dash"),
                    name=f"OLS 迴歸線 (斜率={slope:+.2f}, R²={r_squared:.2f})",
                )
            )
        except Exception:
            pass

    fig.add_hline(y=0, line_width=1, line_color="#64748b", line_dash="solid")
    fig.add_vline(x=0, line_width=1, line_color="#64748b", line_dash="solid")

    fig.update_layout(
        title=(
            "<b>跳空幅度 vs. 盤中開盤至收盤漲跌散布圖 (趨勢特性:"
            f" {trend_desc})</b>"
        ),
        xaxis=dict(
            title="<b>開盤跳空幅度 Gap (%)</b>",
            ticksuffix="%",
            gridcolor="#334155",
            zerolinecolor="#64748b",
        ),
        yaxis=dict(
            title="<b>開盤進場至收盤漲跌幅 (%)</b>",
            ticksuffix="%",
            gridcolor="#334155",
            zerolinecolor="#64748b",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_gauge(prob_up: float, target_date: Optional[datetime.date] = None):
    date_title = f"{target_date.strftime('%Y/%m/%d')} " if target_date else ""
    title_text = f"<b>加權指數 {date_title}上漲勝率預測 P(Up)</b>"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_up * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": title_text,
                "font": {"size": 20, "color": "#f8fafc"},
            },
            number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#94a3b8",
                },
                "bar": {"color": "#f43f5e" if prob_up >= 0.5 else "#10b981"},
                "bgcolor": "#1e293b",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 45], "color": "rgba(16, 185, 129, 0.3)"},
                    {"range": [45, 55], "color": "rgba(148, 163, 184, 0.2)"},
                    {"range": [55, 100], "color": "rgba(244, 63, 94, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.75,
                    "value": prob_up * 100,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig



def render_waterfall(
    contributions: dict,
    tag_extra: str = "",
    selected_features: Optional[List[str]] = None,
):
    label_map = {
        "foreign_net": f"1. 外資現貨買賣超{tag_extra}",
        "tx_oi_diff": f"2. 單日未平倉增減變動量{tag_extra}",
        "fx_inter": "3. 外資期現幾何共振項",
        "usdtwd_ret": "4. 美元兌台幣變動 (USD/TWD)",
        "sox_ret": "5. 美股費半指數 (SOX)",
        "tsm_ret": "6. 台積電 ADR (TSM)",
        "sox_tsm_inter": "7. 費半×台積電ADR幾何共振項",
        "vix_chg": "8. 美股 VIX 恐慌指數變動 (ΔVIX)",
        "nvda_ret": "9. 輝達股票漲跌幅 (NVDA %)",
        "basis_ret": "10. 期現貨價差率 (Basis %)",
        "txf_ah_ret": "11. 台指期盤後夜盤漲跌幅 (%)",
        "nsf_active": "12. 國安基金進場護盤 (Dummy)",
        # --- 技術分析相關變數區塊 ---
        "gap_ret": "13. [技術] 開盤跳空幅度 (Gap %)",
        "ma5_bias": "14. [技術] 5日均線乖離>0 (Dummy)",
        "ma10_bias": "15. [技術] 10日均線乖離>0 (Dummy)",
        "ma5_ma10": "16. [技術] 5日>10日均線 (Dummy)",
        # --- 借券放空籌碼面 ---
        "sbl_5d_chg": f"17. [籌碼] 近5日借券賣出餘額變動{tag_extra}",
    }
    if selected_features is not None and len(selected_features) > 0:
        active_keys = [k for k in contributions.keys() if k in selected_features]
    else:
        active_keys = list(contributions.keys())

    labels = [label_map.get(k, k) for k in active_keys]
    values = [contributions[k] for k in active_keys]
    colors = ["#f43f5e" if v >= 0 else "#10b981" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=(
            f"<b>入選 {len(active_keys)} 大關鍵特徵因子對開盤進場 Log-Odds 邊際貢獻 (Beta × Z-Score)</b>"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="邊際貢獻 (大於 0 偏多，小於 0 偏空)",
            gridcolor="#334155",
            zerolinecolor="#64748b",
        ),
        yaxis=dict(autorange="reversed"),
        font={"color": "#f8fafc"},
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# -----------------------------------------------------------------------------
# 5. 產出 520 交易日真實歷史數據、績效摘要與原生 Excel 試算表
# -----------------------------------------------------------------------------
def export_520_days_to_excel_with_summary(
    hist_df: pd.DataFrame,
    hist_twse: pd.DataFrame,
    hist_taifex: pd.DataFrame,
    engine: TaiexLogisticEngine,
    best_bull: int,
    best_bear: int,
    cost_rate_pct: float = 0.03,
    best_stop_loss_pct: float = 0.0,
    hist_sbl: Optional[pd.DataFrame] = None,
) -> Tuple[bytes, str, str]:
    export_df = hist_df.copy()
    export_df.index = pd.to_datetime(export_df.index).normalize()

    # 1. 嚴格對齊現貨三大法人買賣超 (外資、投信、自營商)
    if (
        hist_twse is not None
        and not hist_twse.empty
        and "foreign_net" in hist_twse.columns
    ):
        twse_clean = hist_twse[~hist_twse.index.duplicated(keep="last")].copy()
        twse_clean.index = pd.to_datetime(twse_clean.index).normalize()
        export_df["外資現貨買賣超(億元)"] = twse_clean["foreign_net"].reindex(export_df.index)
        export_df["投信現貨買賣超(億元)"] = (
            twse_clean["trust_net"].reindex(export_df.index)
            if "trust_net" in twse_clean.columns
            else np.nan
        )
        export_df["自營商現貨買賣超(億元)"] = (
            twse_clean["dealer_net"].reindex(export_df.index)
            if "dealer_net" in twse_clean.columns
            else np.nan
        )
    else:
        export_df["外資現貨買賣超(億元)"] = np.nan
        export_df["投信現貨買賣超(億元)"] = np.nan
        export_df["自營商現貨買賣超(億元)"] = np.nan

    # 2. 嚴格對齊臺指期三大法人未平倉淨額與單日未平倉增減變動量 (口)
    if (
        hist_taifex is not None
        and not hist_taifex.empty
        and "tx_foreign_oi" in hist_taifex.columns
    ):
        taifex_clean = hist_taifex[~hist_taifex.index.duplicated(keep="last")].copy()
        taifex_clean.index = pd.to_datetime(taifex_clean.index).normalize()
        s_aligned_oi = taifex_clean["tx_foreign_oi"].reindex(export_df.index)
        s_aligned_oi_diff = s_aligned_oi.ffill().diff().fillna(0.0)

        missing_oi_count = s_aligned_oi.isna().sum()
        if missing_oi_count > 0:
            st.warning(f"⚠️ 外資期指留倉有 {missing_oi_count} 個交易日查無官方紀錄，已標註為未公告 (NaN)。")

        export_df["外資期指未平倉淨額(口)"] = s_aligned_oi
        export_df["外資期指單日未平倉增減變動量(口)"] = s_aligned_oi_diff
        export_df["投信期指未平倉淨額(口)"] = (
            taifex_clean["tx_trust_oi"].reindex(export_df.index)
            if "tx_trust_oi" in taifex_clean.columns
            else np.nan
        )
        export_df["自營商期指未平倉淨額(口)"] = (
            taifex_clean["tx_dealer_oi"].reindex(export_df.index)
            if "tx_dealer_oi" in taifex_clean.columns
            else np.nan
        )
    else:
        export_df["外資期指未平倉淨額(口)"] = np.nan
        export_df["外資期指單日未平倉增減變動量(口)"] = 0.0
        export_df["投信期指未平倉淨額(口)"] = np.nan
        export_df["自營商期指未平倉淨額(口)"] = np.nan

    # 2-B. 嚴格對齊整體借券賣出餘額與近 5 日累計變動 (張)
    if (
        hist_sbl is not None
        and not hist_sbl.empty
        and "sbl_balance" in hist_sbl.columns
    ):
        sbl_clean = hist_sbl[~hist_sbl.index.duplicated(keep="last")].copy()
        sbl_clean.index = pd.to_datetime(sbl_clean.index).normalize()
        s_aligned_sbl = sbl_clean["sbl_balance"].reindex(export_df.index)
        s_aligned_sbl_5d = s_aligned_sbl.ffill().diff(5).fillna(0.0)
        export_df["整體借券賣出餘額(張)"] = s_aligned_sbl
        export_df["近5日借券賣出餘額累計變動(張)"] = s_aligned_sbl_5d
    else:
        export_df["整體借券賣出餘額(張)"] = np.nan
        export_df["近5日借券賣出餘額累計變動(張)"] = 0.0

    # 補齊潛在空值
    for us_c in ["usdtwd_ret", "sox_ret", "tsm_ret", "nvda_ret", "vix_close", "vix_chg", "us_date", "basis_ret", "txf_ah_ret", "ma5_bias_pct", "ma10_bias_pct", "ma5_bias", "ma10_bias", "ma5_ma10"]:
        if us_c in export_df.columns:
            export_df[us_c] = export_df[us_c].ffill().bfill()

    # 計算外資期現幾何共振項與費半×TSM幾何共振項
    f_trans_col = engine.transform_foreign_net_series(export_df["外資現貨買賣超(億元)"].ffill().fillna(0.0))
    tx_diff_trans_col = engine.transform_oi_series(export_df["外資期指單日未平倉增減變動量(口)"].ffill().fillna(0.0))
    mu_f = engine.active_scaler.get("foreign_net", {}).get("mu", 0.0)
    sig_f = engine.active_scaler.get("foreign_net", {}).get("sigma", 1.0)
    mu_tx = engine.active_scaler.get("tx_oi_diff", {}).get("mu", 0.0)
    sig_tx = engine.active_scaler.get("tx_oi_diff", {}).get("sigma", 1.0)
    z_f_col = (f_trans_col - mu_f) / (sig_f if sig_f > 0 else 1.0)
    z_tx_col = (tx_diff_trans_col - mu_tx) / (sig_tx if sig_tx > 0 else 1.0)
    export_df["外資期現幾何共振項"] = engine.compute_directional_interaction(z_f_col, z_tx_col).ffill().bfill()

    mu_sox = engine.active_scaler.get("sox_ret", {}).get("mu", 0.0)
    sig_sox = engine.active_scaler.get("sox_ret", {}).get("sigma", 1.0)
    mu_tsm = engine.active_scaler.get("tsm_ret", {}).get("mu", 0.0)
    sig_tsm = engine.active_scaler.get("tsm_ret", {}).get("sigma", 1.0)
    z_sox_col = (export_df["sox_ret"] - mu_sox) / (sig_sox if sig_sox > 0 else 1.0)
    z_tsm_col = (export_df["tsm_ret"] - mu_tsm) / (sig_tsm if sig_tsm > 0 else 1.0)
    export_df["費半×台積電ADR幾何共振項"] = engine.compute_directional_interaction(z_sox_col, z_tsm_col).ffill().bfill()

    export_df["nsf_active"] = [float(check_is_nsf_active(d)) for d in export_df.index]
    export_df["國安基金護盤期間(1=是,0=否)"] = export_df["nsf_active"].astype(int)

    sample_tags = ["前490日訓練樣本"] * min(490, len(export_df))
    if len(export_df) > 490:
        sample_tags.extend(["後30日純樣本外測試"] * (len(export_df) - 490))
    export_df["樣本分組"] = sample_tags

    rename_cols = {
        "TWII_Open": "加權指數開盤價",
        "TWII_High": "加權指數最高價",
        "TWII_Low": "加權指數最低價",
        "TWII_Close": "加權指數收盤價",
        "TXF_Open": "近月臺指期開盤價",
        "TXF_High": "近月臺指期最高價",
        "TXF_Low": "近月臺指期最低價",
        "TXF_Close": "近月臺指期收盤價",
        "TXF_Settle": "近月臺指期結算價",
        "TXF_Volume": "近月臺指期成交量(口)",
        "TXF_AH_Open": "近月臺指期盤後夜盤開盤價",
        "TXF_AH_High": "近月臺指期盤後夜盤最高價",
        "TXF_AH_Low": "近月臺指期盤後夜盤最低價",
        "TXF_AH_Close": "近月臺指期盤後夜盤收盤價",
        "TXF_AH_Volume": "近月臺指期盤後夜盤成交量(口)",
        "txf_ah_ret": "台指期盤後夜盤漲跌幅(%)",
        "basis_ret": "期現貨價差率(%)",
        "gap_ret": "加權開盤跳空幅度(%)",
        "ma5_bias_pct": "加權指數5日均線乖離率(%)",
        "ma10_bias_pct": "加權指數10日均線乖離率(%)",
        "ma5_bias": "5日均線正乖離虛擬變數(1=MA5乖離>0)",
        "ma10_bias": "10日均線正乖離虛擬變數(1=MA10乖離>0)",
        "ma5_ma10": "均線多頭排列虛擬變數(1=MA5>MA10)",
        "usdtwd_ret": "美元兌台幣變動(%)",
        "sox_ret": "美股費半漲跌(%)",
        "tsm_ret": "台積電ADR漲跌(%)",
        "nvda_ret": "輝達NVDA漲跌(%)",
        "vix_close": "美股VIX收盤",
        "vix_chg": "美股VIX變動(ΔVIX)",
        "ret_open_to_close": "加權指數開盤至收盤漲跌幅(%)",
        "txf_day_trade_ret": "臺指期當日當沖毛報酬(%)",
        "us_date": "對齊美股交易日",
    }
    export_df = export_df.rename(columns=rename_cols)
    export_df.index.name = "交易日期"

    cols_order = [
        "樣本分組",
        "加權指數開盤價",
        "加權指數最高價",
        "加權指數最低價",
        "加權指數收盤價",
        "加權開盤跳空幅度(%)",
        "加權指數5日均線乖離率(%)",
        "5日均線正乖離虛擬變數(1=MA5乖離>0)",
        "加權指數10日均線乖離率(%)",
        "10日均線正乖離虛擬變數(1=MA10乖離>0)",
        "均線多頭排列虛擬變數(1=MA5>MA10)",
        "近月臺指期開盤價",
        "近月臺指期最高價",
        "近月臺指期最低價",
        "近月臺指期收盤價",
        "近月臺指期結算價",
        "近月臺指期成交量(口)",
        "期現貨價差率(%)",
        "近月臺指期盤後夜盤開盤價",
        "近月臺指期盤後夜盤最高價",
        "近月臺指期盤後夜盤最低價",
        "近月臺指期盤後夜盤收盤價",
        "近月臺指期盤後夜盤成交量(口)",
        "台指期盤後夜盤漲跌幅(%)",
        "臺指期當日當沖毛報酬(%)",
        "外資現貨買賣超(億元)",
        "整體借券賣出餘額(張)",
        "近5日借券賣出餘額累計變動(張)",
        "投信現貨買賣超(億元)",
        "自營商現貨買賣超(億元)",
        "外資期指未平倉淨額(口)",
        "外資期指單日未平倉增減變動量(口)",
        "外資期現幾何共振項",
        "投信期指未平倉淨額(口)",
        "自營商期指未平倉淨額(口)",
        "美元兌台幣變動(%)",
        "對齊美股交易日",
        "美股費半漲跌(%)",
        "台積電ADR漲跌(%)",
        "輝達NVDA漲跌(%)",
        "費半×台積電ADR幾何共振項",
        "美股VIX收盤",
        "美股VIX變動(ΔVIX)",
        "國安基金護盤期間(1=是,0=否)",
    ]
    avail_cols = [c for c in cols_order if c in export_df.columns]
    data_sheet_df = export_df[avail_cols]

    # 3. 樣本內（前 490 日）波段留倉會計結算、逐筆明細與結算流水帳
    curve_records = []
    in_trades_display = pd.DataFrame()
    in_ledger_df = pd.DataFrame()

    if engine.in_sample_probs and not engine.in_sample_df.empty:
        probs_pct = [p * 100 for p in engine.in_sample_probs]
        eval_df = engine.in_sample_df
        (
            sigs, positions, gross_rets, costs, net_rets, validation_status, active, correct
        ) = simulate_day_trading_pnl(
            probs_pct, eval_df, best_bull, best_bear, cost_rate_pct,
            stop_loss_pct=best_stop_loss_pct,
        )

        in_dates_list = [d.strftime("%Y-%m-%d") for d in eval_df.index]
        in_open_list = eval_df["TXF_Open"].tolist()
        in_close_list = eval_df["TXF_Close"].tolist()
        in_high_list = eval_df["TXF_High"].tolist() if "TXF_High" in eval_df.columns else in_open_list
        in_low_list = eval_df["TXF_Low"].tolist() if "TXF_Low" in eval_df.columns else in_close_list

        # (A) 提取樣本內逐筆波段留倉明細 (In-Sample Swing Trades)
        in_trades_df = extract_day_trades(
            in_dates_list, in_open_list, in_close_list, positions,
            gross_rets, costs, net_rets,
            high_prices=in_high_list, low_prices=in_low_list,
            signal_types=sigs, stop_loss_pct=best_stop_loss_pct
        )
        if not in_trades_df.empty:
            in_trades_display = in_trades_df.drop(columns=["_net_val"], errors="ignore")

        # (B) 建立樣本內波段留倉市價結算流水帳 (含美股同日對齊、國安基金狀態與重大事件)
        pos_labels = [
            ("🟢 多單留倉 (+1)" if p == 1 else ("🔴 空單留倉 (-1)" if p == -1 else "⚪ 空手無部位 (0)"))
            for p in positions
        ]

        us_align_labels = []
        for d in eval_df.index:
            matched_row = hist_df.loc[d] if d in hist_df.index else None
            if matched_row is not None:
                usd = matched_row.get("us_date", "-")
                ev = matched_row.get("us_event", "")
                us_align_labels.append(f"{usd}" + (f" ({ev})" if ev else ""))
            else:
                us_align_labels.append("-")

        nsf_status_labels = [
            "🛡️ 護盤期間(1)" if check_is_nsf_active(d.date()) == 1 else "未護盤(0)"
            for d in eval_df.index
        ]

        in_ledger_df = pd.DataFrame({
            "樣本類別": ["樣本內訓練 (In-Sample)"] * len(in_dates_list),
            "目標交易日 (T)": in_dates_list,
            "國安基金": nsf_status_labels,
            "近月期開盤 (TXF Open)": [f"{op:,.1f}" if not np.isnan(op) else "-" for op in in_open_list],
            "近月期收盤 (TXF Close)": [f"{cp:,.1f}" if not np.isnan(cp) else "-" for cp in in_close_list],
            "對齊美股結算日": us_align_labels,
            "樣本內預測勝率": [f"{p:.1f}%" for p in probs_pct],
            "策略訊號": sigs,
            "收盤留倉部位": pos_labels,
            "會計執行動作 (開盤平倉/留倉續抱)": validation_status,
            "單日波段毛損益": [f"{gr:+.2f}%" if not np.isnan(gr) else "尚未結算" for gr in gross_rets],
            "當日摩擦成本": [f"-{c:.3f}%" if c > 0 else "0.000%" for c in costs],
            "單日實質淨損益": [f"{nr:+.2f}%" if not np.isnan(nr) else "尚未結算" for nr in net_rets],
        })

        total_swings = len(in_trades_df)
        winning_swings = int((in_trades_df["_net_val"] > 0).sum()) if (total_swings > 0 and "_net_val" in in_trades_df.columns) else 0
        swing_win_rate = (winning_swings / total_swings * 100.0) if total_swings > 0 else 0.0

        total_days = len(eval_df)
        total_gross = sum(gross_rets)
        total_cost = sum(costs)
        total_net = sum(net_rets)
        annualized_net = (total_net / max(total_days, 1)) * 252.0

        net_cum = np.cumsum(net_rets)
        rolling_max = np.maximum.accumulate(net_cum)
        drawdowns = net_cum - rolling_max
        mdd = drawdowns.min() if len(drawdowns) > 0 else 0.0

        valid_trade_rets = [nr for nr, gr, c in zip(net_rets, gross_rets, costs) if (abs(gr) > 1e-9 or c > 0)]
        if len(valid_trade_rets) > 1:
            ret_series = pd.Series(valid_trade_rets) / 100.0
            rf_daily = (1.5 / 100.0) / 252.0
            excess_mean = (ret_series - rf_daily).mean()
            ret_std = ret_series.std()
            sharpe = (excess_mean / ret_std * np.sqrt(252)) if ret_std > 0 else 0.0
            gains = sum([r for r in valid_trade_rets if r > 0])
            losses = abs(sum([r for r in valid_trade_rets if r < 0]))
            profit_factor = (gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)
            calmar = (annualized_net / abs(mdd)) if abs(mdd) > 0 else (annualized_net if annualized_net > 0 else 0.0)
        else:
            sharpe = 0.0
            profit_factor = 0.0
            calmar = 0.0

        for d_str, nr, cr in zip(in_dates_list, net_rets, net_cum):
            curve_records.append({
                "交易日期": d_str,
                "單日實質淨損益(%)": round(nr, 3),
                "累計淨值報酬(%)": round(cr, 3),
            })
    else:
        total_swings = 0
        winning_swings = 0
        swing_win_rate = 0.0
        total_gross = 0.0
        total_cost = 0.0
        total_net = 0.0
        annualized_net = 0.0
        mdd = 0.0
        sharpe = 0.0
        calmar = 0.0
        profit_factor = 0.0

    target_score_val = swing_win_rate
    summary_records = [
        {"統計項目 (Metrics)": "數據母體採樣長度", "數值 / 指標 (Values)": f"總計 {len(hist_df)} 個台股真實交易日"},
        {"統計項目 (Metrics)": "候選解釋變數母體與自動篩選", "數值 / 指標 (Values)": f"自 17 個候選因子中自動篩選保留 {len(engine.selected_features)} 個高影響力關鍵變數 (剔除 {len(engine.eliminated_features)} 個雜訊變數)"},
        {"統計項目 (Metrics)": "入選關鍵解釋變數清單 (Selected Features)", "數值 / 指標 (Values)": ", ".join(engine.selected_features)},
        {"統計項目 (Metrics)": "模型訓練樣本範圍 (In-Sample)", "數值 / 指標 (Values)": f"前 {len(eval_df) if engine.in_sample_probs else 0} 個台股真實交易日"},
        {"統計項目 (Metrics)": "純樣本外測試範圍 (Out-of-Sample)", "數值 / 指標 (Values)": "後 30 個台股真實交易日 (嚴格盲測)"},
        {"統計項目 (Metrics)": "多方波段發動門檻 (3x3 最佳中心)", "數值 / 指標 (Values)": f"{best_bull}%"},
        {"統計項目 (Metrics)": "空方波段防禦門檻 (3x3 最佳中心)", "數值 / 指標 (Values)": f"{best_bear}%"},
        {"統計項目 (Metrics)": "最佳化停損百分比 (Stop-Loss %, 不停利)", "數值 / 指標 (Values)": f"-{best_stop_loss_pct:.1f}%" if best_stop_loss_pct > 0 else "不設停損"},
        {"統計項目 (Metrics)": "樣本內波段總交易筆數 (Total Swings)", "數值 / 指標 (Values)": f"{total_swings} 筆"},
        {"統計項目 (Metrics)": "樣本內獲利波段 / 虧損波段筆數", "數值 / 指標 (Values)": f"{winning_swings} 筆 / {total_swings - winning_swings} 筆"},
        {"統計項目 (Metrics)": "樣本內波段實質勝率 (Swing Win Rate)", "數值 / 指標 (Values)": f"{swing_win_rate:.2f}%"},
        {"統計項目 (Metrics)": "樣本內波段實質累積淨報酬 (Net Return)", "數值 / 指標 (Values)": f"{total_net:+.2f}%"},
        {"統計項目 (Metrics)": "優化目標分數 [波段勝率 (%)]", "數值 / 指標 (Values)": f"{target_score_val:+.2f}"},
        {"統計項目 (Metrics)": "樣本內年化實質淨報酬率 (Annualized Return)", "數值 / 指標 (Values)": f"{annualized_net:+.2f}%"},
        {"統計項目 (Metrics)": "樣本內最大淨值回撤 (Max Drawdown, MDD)", "數值 / 指標 (Values)": f"{mdd:.2f}%"},
        {"統計項目 (Metrics)": "樣本內年化夏普比率 (Sharpe Ratio, Rf=1.5%)", "數值 / 指標 (Values)": f"{sharpe:.2f}"},
        {"統計項目 (Metrics)": "樣本內卡瑪比率 (Calmar Ratio, 年化/|MDD|)", "數值 / 指標 (Values)": f"{calmar:.2f}"},
        {"統計項目 (Metrics)": "樣本內獲利因子 (Profit Factor, 總盈/總虧)", "數值 / 指標 (Values)": f"{profit_factor:.2f}" if profit_factor < 90 else "∞ (無虧損)"},
    ]
    summary_df = pd.DataFrame(summary_records)
    curve_df = pd.DataFrame(curve_records) if curve_records else pd.DataFrame(columns=["交易日期", "單日實質淨損益(%)", "累計淨值報酬(%)"])

    try:
        import openpyxl
        from openpyxl.chart import LineChart, Reference

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # 1. 績效統計摘要表
            summary_df.to_excel(writer, sheet_name="490日波段留倉績效摘要", index=False)
            
            # 2. 逐筆波段留倉交易明細
            if not in_trades_display.empty:
                in_trades_display.to_excel(writer, sheet_name="樣本內波段逐筆交易明細", index=False)
            
            # 3. 市價結算流水帳
            if not in_ledger_df.empty:
                in_ledger_df.to_excel(writer, sheet_name="樣本內波段市價結算流水帳", index=False)
                
            # 4. 淨值走勢數據
            if not curve_df.empty:
                curve_df.to_excel(writer, sheet_name="波段留倉淨值曲線數據", index=False)
                
            # 5. 520 日台股真實歷史與因子行情明細
            data_sheet_df.to_excel(writer, sheet_name="520日台股真實交易數據")

            if not curve_df.empty:
                wb = writer.book
                ws_summary = wb["490日波段留倉績效摘要"]
                ws_curve = wb["波段留倉淨值曲線數據"]

                chart = LineChart()
                chart.title = f"前490日臺指期波段留倉累計淨值走勢圖 (Sharpe: {sharpe:.2f}, MDD: {mdd:.2f}%)"
                chart.style = 2
                chart.legend = None

                chart.y_axis.title = "累積淨報酬率 (%)"
                chart.y_axis.delete = False
                chart.y_axis.tickLblPos = "nextTo"
                chart.y_axis.numFmt = "0.0"

                chart.x_axis.title = "交易日期"
                chart.x_axis.delete = False
                chart.x_axis.tickLblPos = "low"
                chart.x_axis.tickLblSkip = 20
                chart.x_axis.tickMarkSkip = 20

                chart.width = 24
                chart.height = 14

                data_ref = Reference(ws_curve, min_col=3, min_row=1, max_row=len(curve_df) + 1)
                cats_ref = Reference(ws_curve, min_col=1, min_row=2, max_row=len(curve_df) + 1)

                chart.add_data(data_ref, titles_from_data=True)
                chart.set_categories(cats_ref)

                if chart.series:
                    s1 = chart.series[0]
                    s1.graphicalProperties.line.solidFill = "0284C7"
                    s1.graphicalProperties.line.width = 22000

                ws_summary.add_chart(chart, "E2")

        return (
            output.getvalue(),
            "xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        summary_text = "# ===== 520日數據 (前490日訓練 / 後30日樣本外) 績效統計摘要表 =====\n"
        for _, r in summary_df.iterrows():
            summary_text += f"# {r['統計項目 (Metrics)']}: {r['數值 / 指標 (Values)']}\n"
        summary_text += "# =========================================================\n\n"
        csv_body = data_sheet_df.to_csv(encoding="utf-8-sig")
        final_csv_bytes = (summary_text + csv_body).encode("utf-8-sig")
        return (
            final_csv_bytes,
            "csv",
            "text/csv",
        )


# -----------------------------------------------------------------------------
# 6. Streamlit 介面佈局與資料載入
# -----------------------------------------------------------------------------
pipeline = MarketDataPipeline()

st.sidebar.header("🕹️ 17因子特徵監控與波段留倉模擬器")

auth_col1, auth_col2 = st.sidebar.columns([3, 2])
with auth_col1:
    current_user = st.session_state.get("current_user", "使用者")
    st.caption(f"🟢 使用者: **{current_user}**")
with auth_col2:
    if st.button("登出", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["current_user"] = None
        st.rerun()

st.sidebar.markdown("---")

with st.sidebar.expander(
    "💾 本地磁碟快取狀態面板 (520日 Parquet Cache - 含夜盤數據)", expanded=True
):
    twse_info = pipeline.get_cache_file_info(pipeline.twse_parquet_path)
    taifex_info = pipeline.get_cache_file_info(pipeline.taifex_parquet_path)
    txf_price_info = pipeline.get_cache_file_info(pipeline.txf_price_parquet_path)
    sbl_info = pipeline.get_cache_file_info(pipeline.sbl_parquet_path)

    st.markdown("**證交所現貨買賣超快取** (`twse_chip_520d.parquet`)")
    if twse_info["exists"]:
        st.caption(
            f"📦 大小: `{twse_info['size_str']}` | 筆數:"
            f" `{twse_info['records_str']}`\n\n🕒 更新:"
            f" `{twse_info['mtime_str']}`\n\n📅 涵蓋: `{twse_info['span_str']}`"
        )
    else:
        st.caption("⚪ 尚未建立本地檔案")

    st.markdown("**期交所期指留倉淨額快取** (`taifex_chip_520d.parquet`)")
    if taifex_info["exists"]:
        st.caption(
            f"📦 大小: `{taifex_info['size_str']}` | 筆數:"
            f" `{taifex_info['records_str']}`\n\n🕒 更新:"
            f" `{taifex_info['mtime_str']}`\n\n📅 涵蓋:"
            f" `{taifex_info['span_str']}`"
        )
    else:
        st.caption("⚪ 尚未建立本地檔案")

    st.markdown("**期交所近月臺指期日盤+盤後夜盤快取** (`txf_prices_520d.parquet`)")
    if txf_price_info["exists"]:
        st.caption(
            f"📦 大小: `{txf_price_info['size_str']}` | 筆數:"
            f" `{txf_price_info['records_str']}`\n\n🕒 更新:"
            f" `{txf_price_info['mtime_str']}`\n\n📅 涵蓋:"
            f" `{txf_price_info['span_str']}`"
        )
    else:
        st.caption("⚪ 尚未建立本地檔案")

    st.markdown("**證交所整體借券賣出餘額快取** (`twse_sbl_520d.parquet`)")
    if sbl_info["exists"]:
        st.caption(
            f"📦 大小: `{sbl_info['size_str']}` | 筆數:"
            f" `{sbl_info['records_str']}`\n\n🕒 更新:"
            f" `{sbl_info['mtime_str']}`\n\n📅 涵蓋:"
            f" `{sbl_info['span_str']}`"
        )
    else:
        st.caption("⚪ 尚未建立本地檔案")

    if twse_info["exists"] or taifex_info["exists"] or txf_price_info["exists"] or sbl_info["exists"]:
        current_user = st.session_state.get("current_user", "")
        is_admin = (current_user == "admin")

        if is_admin:
            if st.button("🗑️ 清除本地 Parquet 快取", use_container_width=True):
                gc.collect()
                gc.collect()

                cache_files = [
                    pipeline.twse_parquet_path,
                    pipeline.taifex_parquet_path,
                    pipeline.txf_price_parquet_path,
                    pipeline.sbl_parquet_path,
                ]
                
                deleted_list = []
                failed_list = []

                for f_path in cache_files:
                    if os.path.exists(f_path):
                        deleted = False
                        for attempt in range(3):
                            try:
                                os.remove(f_path)
                                deleted = True
                                deleted_list.append(f_path)
                                break
                            except Exception:
                                time.sleep(0.15)
                                gc.collect()

                        if not deleted:
                            try:
                                with open(f_path, "wb") as f:
                                    f.truncate(0)
                                deleted_list.append(f"{f_path} (已強制清空內容)")
                            except Exception as e:
                                failed_list.append(f"{f_path} ({str(e)})")

                keys_to_clear = [
                    "cached_data",
                    "last_target_date",
                    "slider_f_net",
                    "slider_t_net",
                    "slider_tx_oi_diff",
                    "slider_usdtwd_r",
                    "slider_sox_r",
                    "slider_tsm_r",
                    "slider_nvda_r",
                    "slider_vix_chg",
                    "slider_gap_r",
                    "slider_basis_r",
                    "slider_txf_ah_r",
                    "slider_ma5_bias",
                    "slider_ma10_bias",
                    "slider_ma5_ma10",
                    "slider_sbl_5d_chg",
                    "slider_nsf_active",
                ]
                for k in keys_to_clear:
                    st.session_state.pop(k, None)

                if failed_list:
                    st.error("⚠️ 部分快取因背景程序鎖定未能完全刪除：\n- " + "\n- ".join(failed_list))
                else:
                    st.success("✅ 已徹底清除四個本地 Parquet 快取與記憶體資料！下次點選擷取將全面重新抓取。")
                
                time.sleep(0.5)
                st.rerun()
        else:
            st.caption("🔒 **權限限制**：僅系統管理者 (`admin`) 可執行磁碟快取重置。")

st.sidebar.markdown("---")

fixed_weight_option = st.sidebar.radio(
    "⚖️ 是否採固定權重",
    ["是 (採固定專家權重)", "否 (依解釋變數自動篩選機制動態配適)"],
    index=1,
    help="預設為「是」：直接採用固定專家權重；若選擇「否」，則依下方解釋變數自動篩選機制動態配適。",
)
use_fixed_weights = fixed_weight_option.startswith("是")

chip_transform_mode = st.sidebar.radio(
    "📐 外資現期貨籌碼特徵變換",
    ["原始線性水準項", "雙對稱對數變換 (Signed-Log)"],
    index=0,
    help="預設啟用「原始線性水準項」：直接採用外資現貨買賣超（億元）與期指單日未平倉變動量 (ΔOI, 口) 之水準項進行建模；亦可切換回「雙對稱對數變換 (Signed-Log)」進行比對。",
)

feature_select_option = st.sidebar.radio(
    "🧬 解釋變數自動篩選機制",
    ["自動選取高影響力關鍵變數 (Stepwise + RFE + L1)", "保留全部 17 個解釋變數"],
    index=0,
    help="預設啟用「自動選取高影響力關鍵變數」：透過雙向逐步迴歸 (Stepwise)、遞迴特徵消除 (RFE) 與 L1 Lasso 正則化，自動剔除對預測漲跌無助益或具共線性雜訊之變數，僅留下對預測成功率較有影響的核心解釋變數！",
)
auto_select_flag = feature_select_option.startswith("自動選取")

# -----------------------------------------------------------------------------
# 停損設定：將預設 index 由 0 改為 1 (預設為「需停損」)
# -----------------------------------------------------------------------------
stop_loss_option = st.sidebar.radio(
    "🛡️ 是否需停損設定 (純停損、不停利)",
    ["不需停損", "需停損 (優化自動設定最佳停損%)"],
    index=0,  # 預設不需停損
    help="預設為「需停損」：優化過程將自動於候選停損比例 (0.5% ~ 5.0%) 中搜尋使 3×3 九宮格目標分數最大之最佳停損百分比。",
)
enable_stop_loss_flag = stop_loss_option.startswith("需停損")

engine = TaiexLogisticEngine(
    transform_mode=chip_transform_mode,
    auto_select_features=auto_select_flag,
)

DATA_WINDOW_DAYS = 520
TRAIN_SAMPLE_DAYS = 490
OOS_TEST_DAYS = 30

selected_target_date = st.sidebar.date_input(
    "選擇目標預測交易日 (T)", datetime.date.today()
)

is_holiday, holiday_reason = check_is_taiwan_holiday(selected_target_date)
if is_holiday:
    st.sidebar.error(f"⚠️ 所選日期為 **{holiday_reason}**，台灣股市未開盤！")
    st.error(
        f"🛑 **目標交易日 ({selected_target_date}) 為【{holiday_reason}】**。非交易日市場休市，無需計算盤前預測機率，請於左側選單選擇一般交易日。"
    )
    st.stop()

base_prev_date = pipeline.get_previous_weekday(selected_target_date)

btn_col1, btn_col2 = st.sidebar.columns([3, 2])
with btn_col1:
    fetch_btn = st.button("🚀 擷取當日數據", use_container_width=True)
with btn_col2:
    reset_btn = st.button("↺ 還原特徵", use_container_width=True)

st.sidebar.caption(
    f"📌 **波段留倉、雙門檻緩衝與純停損規則（不停利）**：以 **{selected_target_date}** 近月臺指期開盤價進場，**採雙門檻緩衝（多單 p ≥ 50% 續抱、空單 p ≤ 50% 續抱），跌破/突破 50% 中軸或訊號反轉時以預測日開盤價平倉；若設定「需停損」，優化器將自動尋優最佳停損百分比，於盤中觸及停損門檻時即刻停損平倉**。"
)


def safe_clamp(val, min_v, max_v, default_v=0.0):
    try:
        f_val = float(val)
        if np.isnan(f_val):
            return float(default_v)
        return float(max(min(f_val, max_v), min_v))
    except Exception:
        return float(default_v)


def reset_slider_states(
    f_net, t_net, tx_oi_diff, usdtwd_r, sox_r, tsm_r, vix_chg, gap_r=0.0, basis_r=0.0, txf_ah_r=0.0, ma5_b=0, nsf_act=0, nvda_r=0.0, ma10_b=0, ma5_ma10_v=0, sbl_5d_c=0.0
):
    st.session_state["slider_f_net"] = safe_clamp(f_net, -1200.0, 1200.0, 0.0)
    st.session_state["slider_t_net"] = safe_clamp(t_net, -300.0, 300.0, 0.0)
    st.session_state["slider_tx_oi_diff"] = safe_clamp(
        tx_oi_diff, -40000.0, 40000.0, 0.0
    )
    st.session_state["slider_usdtwd_r"] = safe_clamp(usdtwd_r, -2.0, 2.0, 0.0)
    st.session_state["slider_sox_r"] = safe_clamp(sox_r, -15.0, 15.0, 0.0)
    st.session_state["slider_tsm_r"] = safe_clamp(tsm_r, -15.0, 15.0, 0.0)
    st.session_state["slider_nvda_r"] = safe_clamp(nvda_r, -20.0, 20.0, 0.0)
    st.session_state["slider_vix_chg"] = safe_clamp(vix_chg, -20.0, 20.0, 0.0)
    st.session_state["slider_basis_r"] = safe_clamp(basis_r, -5.0, 5.0, 0.0)
    st.session_state["slider_txf_ah_r"] = safe_clamp(txf_ah_r, -10.0, 10.0, 0.0)
    st.session_state["slider_nsf_active"] = 1 if int(nsf_act) == 1 else 0
    # --- 技術分析相關變數區塊 ---
    st.session_state["slider_gap_r"] = safe_clamp(gap_r, -10.0, 10.0, 0.0)
    st.session_state["slider_ma5_bias"] = 1 if int(ma5_b) == 1 else 0
    st.session_state["slider_ma10_bias"] = 1 if int(ma10_b) == 1 else 0
    st.session_state["slider_ma5_ma10"] = 1 if int(ma5_ma10_v) == 1 else 0
    st.session_state["slider_sbl_5d_chg"] = safe_clamp(sbl_5d_c, -1500000.0, 1500000.0, 0.0)


need_fetch = (
    "cached_data" not in st.session_state
    or fetch_btn
    or st.session_state.get("last_target_date") != selected_target_date
)

if need_fetch:
    with st.spinner(
        "正在極速擷取 520 交易日真實歷史數據（前490天訓練，後30天OOS樣本外檢驗，含近月台指期盤後夜盤）..."
    ):
        twse_df, twse_actual_date, twse_msg = (
            pipeline.fetch_twse_institutional_with_fallback(
                base_prev_date
            )
        )
        taifex_df, taifex_actual_date, taifex_msg = (
            pipeline.fetch_taifex_oi_with_fallback(base_prev_date)
        )

        us_df, us_actual_date, us_msg = pipeline.fetch_us_market(
            base_prev_date, days_back=40
        )

        hist_df = pipeline.fetch_historical_training_and_verification_data(
            selected_target_date, exact_train_days=DATA_WINDOW_DAYS
        )

        hist_chip_start = selected_target_date - datetime.timedelta(
            days=int(DATA_WINDOW_DAYS * 1.65) + 60
        )
        hist_chip_end = selected_target_date

        tw_trading_dates = hist_df.index.tolist() if not hist_df.empty else None

        hist_twse_df = pipeline.fetch_twse_historical_range_parquet(
            hist_chip_start, hist_chip_end, target_trading_dates=tw_trading_dates
        )

        hist_taifex_df = pipeline.fetch_taifex_historical_range_parquet(
            hist_chip_start, hist_chip_end, target_trading_dates=tw_trading_dates
        )

        hist_sbl_df = pipeline.fetch_sbl_historical_range_parquet(
            hist_chip_start, hist_chip_end, target_trading_dates=tw_trading_dates
        )

        if hist_twse_df.empty or hist_taifex_df.empty:
            api_twse, api_taifex = pipeline.fetch_finmind_chips_history(
                hist_chip_start, hist_chip_end
            )
            if hist_twse_df.empty and not api_twse.empty:
                hist_twse_df = api_twse
                try:
                    hist_twse_df.to_parquet(pipeline.twse_parquet_path)
                except Exception:
                    pass

            if hist_taifex_df.empty and not api_taifex.empty:
                hist_taifex_df = api_taifex
                try:
                    hist_taifex_df.to_parquet(pipeline.taifex_parquet_path)
                except Exception:
                    pass

        f_net = (
            float(twse_df["foreign_net"].iloc[0])
            if not twse_df.empty and "foreign_net" in twse_df
            else 0.0
        )
        t_net = (
            float(twse_df["trust_net"].iloc[0])
            if not twse_df.empty and "trust_net" in twse_df
            else 0.0
        )
        tx_oi = (
            float(taifex_df["tx_foreign_oi"].iloc[0])
            if not taifex_df.empty and "tx_foreign_oi" in taifex_df
            else 0.0
        )

        # 計算外資期指單日未平倉增減變動量 (ΔOI)
        tx_oi_diff = 0.0
        if not hist_taifex_df.empty and "tx_foreign_oi" in hist_taifex_df.columns:
            prior_tfx = hist_taifex_df[hist_taifex_df.index < pd.to_datetime(selected_target_date)].dropna(subset=["tx_foreign_oi"])
            if len(prior_tfx) >= 2:
                tx_oi_diff = float(prior_tfx["tx_foreign_oi"].iloc[-1] - prior_tfx["tx_foreign_oi"].iloc[-2])

        # 計算近 5 日整體借券賣出餘額累計變動 (張, T-1 減 T-6)
        sbl_5d_chg_latest = 0.0
        sbl_actual_date = base_prev_date
        sbl_msg = "查無數據"
        if not hist_sbl_df.empty and "sbl_balance" in hist_sbl_df.columns:
            prior_sbl = hist_sbl_df[hist_sbl_df.index < pd.to_datetime(selected_target_date)].dropna(subset=["sbl_balance"])
            if len(prior_sbl) >= 6:
                sbl_5d_chg_latest = float(prior_sbl["sbl_balance"].iloc[-1] - prior_sbl["sbl_balance"].iloc[-6])
                sbl_actual_date = prior_sbl.index[-1].date()
                sbl_msg = "正常取得" if sbl_actual_date == base_prev_date else f"原日前推至有效交易日 ({sbl_actual_date})"
            elif len(prior_sbl) >= 2:
                sbl_5d_chg_latest = float(prior_sbl["sbl_balance"].iloc[-1] - prior_sbl["sbl_balance"].iloc[0])
                sbl_actual_date = prior_sbl.index[-1].date()
                sbl_msg = f"樣本不足6日，採近 {len(prior_sbl)} 日累計變動 ({sbl_actual_date})"

        latest_us = us_df.iloc[-1] if not us_df.empty else None
        usdtwd_r = float(latest_us["usdtwd_ret"]) if latest_us is not None and "usdtwd_ret" in latest_us else 0.0
        sox_r = float(latest_us["sox_ret"]) if latest_us is not None else 0.0
        tsm_r = float(latest_us["tsm_ret"]) if latest_us is not None else 0.0
        nvda_r = float(latest_us["nvda_ret"]) if latest_us is not None and "nvda_ret" in latest_us else 0.0
        vix_c = float(latest_us["vix_close"]) if latest_us is not None else 18.0
        vix_chg_val = float(latest_us["vix_chg"]) if latest_us is not None and "vix_chg" in latest_us else 0.0

        # 嚴格取前一交易日 (T-1) 之技術分析特徵 (跳空、ma5_bias 虛擬變數、ma10_bias 虛擬變數、ma5_ma10 虛擬變數)
        # 夜盤價差率與夜盤報酬率取預測日已結算數據 (遇缺以前一日取代)
        gap_r = 0.0
        basis_r = 0.0
        txf_ah_r = 0.0
        ma5_b = 0
        ma10_b = 0
        ma5_ma10_v = 0
        ma5_b_pct = 0.0
        ma10_b_pct = 0.0
        if not hist_df.empty:
            target_p = hist_df[hist_df.index == pd.to_datetime(selected_target_date)]
            prior_hist = hist_df[hist_df.index < pd.to_datetime(selected_target_date)]
            if not target_p.empty and not np.isnan(target_p.iloc[-1].get("basis_ret", np.nan)):
                basis_r = float(target_p.iloc[-1]["basis_ret"])
                txf_ah_r = float(target_p.iloc[-1]["txf_ah_ret"])
            elif not prior_hist.empty:
                basis_r = float(prior_hist.iloc[-1].get("basis_ret", 0.0))
                txf_ah_r = float(prior_hist.iloc[-1].get("txf_ah_ret", 0.0))

            if not prior_hist.empty:
                last_p = prior_hist.iloc[-1]
                gap_r = float(last_p.get("gap_ret", 0.0))
                ma5_b = int(last_p.get("ma5_bias", 0))
                ma10_b = int(last_p.get("ma10_bias", 0))
                ma5_ma10_v = int(last_p.get("ma5_ma10", 0))
                ma5_b_pct = float(last_p.get("ma5_bias_pct", 0.0))
                ma10_b_pct = float(last_p.get("ma10_bias_pct", 0.0))
            else:
                gap_r = round((sox_r * 0.4 + tsm_r * 0.6) * 0.35, 2)

        nsf_act = check_is_nsf_active(base_prev_date)
        us_macro_event = US_MACRO_EVENTS_MAP.get(us_actual_date, "")

        st.session_state["cached_data"] = {
            "target_date": selected_target_date,
            "base_feature_date": base_prev_date,
            "twse_actual_date": twse_actual_date,
            "taifex_actual_date": taifex_actual_date,
            "us_actual_date": us_actual_date,
            "us_macro_event": us_macro_event,
            "twse_msg": twse_msg,
            "taifex_msg": taifex_msg,
            "us_msg": us_msg,
            "f_net": f_net,
            "t_net": t_net,
            "tx_oi": tx_oi,
            "tx_oi_diff": tx_oi_diff,
            "usdtwd_r": usdtwd_r,
            "sox_r": sox_r,
            "tsm_r": tsm_r,
            "nvda_r": nvda_r,
            "vix_c": vix_c,
            "vix_chg": vix_chg_val,
            "gap_r": gap_r,
            "basis_r": basis_r,
            "txf_ah_r": txf_ah_r,
            "ma5_b": ma5_b,
            "ma10_b": ma10_b,
            "ma5_ma10_v": ma5_ma10_v,
            "ma5_b_pct": ma5_b_pct,
            "ma10_b_pct": ma10_b_pct,
            "sbl_5d_chg": sbl_5d_chg_latest,
            "sbl_actual_date": sbl_actual_date,
            "sbl_msg": sbl_msg,
            "nsf_act": nsf_act,
            "hist_df": hist_df,
            "hist_twse_df": hist_twse_df,
            "hist_taifex_df": hist_taifex_df,
            "hist_sbl_df": hist_sbl_df,
        }
        st.session_state["last_target_date"] = selected_target_date
        reset_slider_states(
            f_net, t_net, tx_oi_diff, usdtwd_r, sox_r, tsm_r, vix_chg_val, gap_r, basis_r, txf_ah_r, ma5_b, nsf_act, nvda_r, ma10_b, ma5_ma10_v, sbl_5d_chg_latest
        )

elif reset_btn:
    c = st.session_state["cached_data"]
    reset_slider_states(
        c["f_net"],
        c["t_net"],
        c.get("tx_oi_diff", 0.0),
        c.get("usdtwd_r", 0.0),
        c["sox_r"],
        c["tsm_r"],
        c.get("vix_chg", 0.0),
        c.get("gap_r", 0.0),
        c.get("basis_r", 0.0),
        c.get("txf_ah_r", 0.0),
        c.get("ma5_b", 0),
        c.get("nsf_act", check_is_nsf_active(c["target_date"])),
        c.get("nvda_r", 0.0),
        c.get("ma10_b", 0),
        c.get("ma5_ma10_v", 0),
        c.get("sbl_5d_chg", 0.0),
    )
    st.toast("已還原至真實特徵數值！", icon="🔄")

data = st.session_state["cached_data"]
target_dt = pd.to_datetime(data["target_date"])

if "hist_df" in data and not data["hist_df"].empty:
    data["hist_df"]["nsf_active"] = [
        float(check_is_nsf_active(d)) for d in data["hist_df"].index
    ]
data["nsf_act"] = check_is_nsf_active(data.get("base_feature_date", base_prev_date))

# -----------------------------------------------------------------------------
# 核心架構：前 490 天訓練尋優，後 30 天做純樣本外 (OOS) 盲測
# -----------------------------------------------------------------------------
cost_rate_setting = 0.03
hist_df_full = data["hist_df"]

if len(hist_df_full) >= DATA_WINDOW_DAYS:
    train_sample_df = hist_df_full.iloc[:TRAIN_SAMPLE_DAYS].copy()
    oos_eval_df = hist_df_full.iloc[TRAIN_SAMPLE_DAYS:].copy()
else:
    train_sample_df = hist_df_full.iloc[:-OOS_TEST_DAYS].copy() if len(hist_df_full) > OOS_TEST_DAYS else hist_df_full.copy()
    oos_eval_df = hist_df_full.tail(min(OOS_TEST_DAYS, len(hist_df_full))).copy()

if not use_fixed_weights and not train_sample_df.empty:
    engine.fit_model_on_sample(
        train_sample=train_sample_df,
        hist_twse=data.get("hist_twse_df"),
        hist_taifex=data.get("hist_taifex_df"),
        hist_sbl=data.get("hist_sbl_df"),
    )
else:
    engine.reset_to_static(
        train_sample=train_sample_df,
        hist_twse=data.get("hist_twse_df"),
        hist_taifex=data.get("hist_taifex_df"),
        hist_sbl=data.get("hist_sbl_df"),
    )

if len(engine.in_sample_probs) >= 40:
    in_sample_probs_pct = [p * 100 for p in engine.in_sample_probs]
    (
        joint_best_bull,
        joint_best_bear,
        joint_best_stop_loss_pct,
        joint_best_acc,
        joint_best_net_ret,
        joint_best_score,
        joint_score_matrix,
        joint_text_matrix,
    ) = optimize_thresholds_fixed_days(
        probs_pct=in_sample_probs_pct,
        eval_df=engine.in_sample_df,
        acc_pct=engine.fit_accuracy if engine.fit_accuracy is not None else 50.0,
        bull_range=(60, 75),
        bear_range=(25, 40),
        step=1,
        cost_rate_pct=cost_rate_setting,
        enable_stop_loss=enable_stop_loss_flag,
    )
else:
    joint_best_bull = 60
    joint_best_bear = 40
    joint_best_stop_loss_pct = 2.0 if enable_stop_loss_flag else 0.0
    joint_best_acc = 50.0
    joint_best_net_ret = 0.0
    joint_best_score = 0.0
    joint_score_matrix = pd.DataFrame()
    joint_text_matrix = pd.DataFrame()

# -----------------------------------------------------------------------------
# 計算樣本內波段實質勝率 (供頂部橫幅呈現，取代原本無門檻的二元分類準確率)
# -----------------------------------------------------------------------------
in_total_swings = 0
in_win_swings = 0
in_swing_win_rate = 0.0
if len(engine.in_sample_probs) >= 40 and not engine.in_sample_df.empty:
    _sigs_in, _pos_in, _g_in, _c_in, _n_in, _, _, _ = simulate_day_trading_pnl(
        in_sample_probs_pct,
        engine.in_sample_df,
        joint_best_bull,
        joint_best_bear,
        cost_rate_setting,
        stop_loss_pct=joint_best_stop_loss_pct,
    )
    _high_list = engine.in_sample_df["TXF_High"].tolist() if "TXF_High" in engine.in_sample_df.columns else engine.in_sample_df["TXF_Open"].tolist()
    _low_list = engine.in_sample_df["TXF_Low"].tolist() if "TXF_Low" in engine.in_sample_df.columns else engine.in_sample_df["TXF_Close"].tolist()
    _in_trades = extract_day_trades(
        [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in engine.in_sample_df.index],
        engine.in_sample_df["TXF_Open"].tolist(),
        engine.in_sample_df["TXF_Close"].tolist(),
        _pos_in, _g_in, _c_in, _n_in,
        high_prices=_high_list, low_prices=_low_list,
        signal_types=_sigs_in,
        stop_loss_pct=joint_best_stop_loss_pct,
    )
    if not _in_trades.empty and "_net_val" in _in_trades.columns:
        in_total_swings = len(_in_trades)
        in_win_swings = int((_in_trades["_net_val"] > 0).sum())
        in_swing_win_rate = (in_win_swings / in_total_swings * 100.0) if in_total_swings > 0 else 0.0

if "slider_f_net" not in st.session_state or "slider_tx_oi_diff" not in st.session_state or "slider_ma10_bias" not in st.session_state or "slider_sbl_5d_chg" not in st.session_state:
    reset_slider_states(
        data["f_net"],
        data["t_net"],
        data.get("tx_oi_diff", 0.0),
        data.get("usdtwd_r", 0.0),
        data["sox_r"],
        data["tsm_r"],
        data.get("vix_chg", 0.0),
        data.get("gap_r", 0.0),
        data.get("basis_r", 0.0),
        data.get("txf_ah_r", 0.0),
        data.get("ma5_b", 0),
        data.get("nsf_act", check_is_nsf_active(data["target_date"])),
        data.get("nvda_r", 0.0),
        data.get("ma10_b", 0),
        data.get("ma5_ma10_v", 0),
        data.get("sbl_5d_chg", 0.0),
    )

st.sidebar.caption("💡 可微調 17 項升級因子數值進行即時敏感度測試：")
st.sidebar.markdown("**🏦 籌碼、國際美股、期貨夜盤與政策面**")

sim_foreign_net = st.sidebar.slider(
    "1. 外資現貨買賣超 (億元)",
    min_value=-1200.0,
    max_value=1200.0,
    value=st.session_state["slider_f_net"],
    step=1.0,
    key="slider_f_net",
)
sim_tx_oi_diff = st.sidebar.slider(
    "2. 單日未平倉增減變動量 ΔOI (口)",
    min_value=-40000.0,
    max_value=40000.0,
    value=st.session_state["slider_tx_oi_diff"],
    step=50.0,
    key="slider_tx_oi_diff",
    help="外資臺指期未平倉淨額之單日增減變動口數 (T-1 口數減 T-2 口數)",
)
sim_usdtwd_ret = st.sidebar.slider(
    "4. 美元兌台幣 (USD/TWD) 漲跌幅 (%)",
    min_value=-2.0,
    max_value=2.0,
    value=st.session_state.get("slider_usdtwd_r", 0.0),
    step=0.05,
    key="slider_usdtwd_r",
    help="正值代表美元升值／台幣貶值（資金流出偏空）；負值代表台幣升值（熱錢流入偏多）",
)
sim_sox_ret = st.sidebar.slider(
    "5. 費半指數漲跌幅 (%)",
    min_value=-15.0,
    max_value=15.0,
    value=st.session_state["slider_sox_r"],
    step=0.1,
    key="slider_sox_r",
)
sim_tsm_ret = st.sidebar.slider(
    "6. 台積電 ADR 漲跌幅 (%)",
    min_value=-15.0,
    max_value=15.0,
    value=st.session_state["slider_tsm_r"],
    step=0.1,
    key="slider_tsm_r",
)
sim_vix_chg = st.sidebar.slider(
    "8. VIX 恐慌指數變動 (ΔVIX)",
    min_value=-20.0,
    max_value=20.0,
    value=st.session_state.get("slider_vix_chg", 0.0),
    step=0.05,
    key="slider_vix_chg",
    help="美股 VIX 恐慌指數單日增減點數 (正值代表恐慌升溫偏空，負值代表恐慌降溫偏多)",
)
sim_nvda_ret = st.sidebar.slider(
    "9. 輝達 (NVDA) 股票漲跌幅 (%)",
    min_value=-20.0,
    max_value=20.0,
    value=st.session_state.get("slider_nvda_r", 0.0),
    step=0.1,
    key="slider_nvda_r",
    help="美股輝達 (NVIDIA, NVDA) 前一交易日收盤漲跌幅 (%)，反映全球 AI 與台灣伺服器/半導體供應鏈動能。",
)
sim_basis_ret = st.sidebar.slider(
    "10. 期現貨價差率 Basis (%)",
    min_value=-5.0,
    max_value=5.0,
    value=st.session_state.get("slider_basis_r", 0.0),
    step=0.02,
    key="slider_basis_r",
    help="(預測日近月臺指期夜盤收盤價 - 前一交易日加權指數收盤價) / 前一交易日加權指數收盤價 × 100%",
)
sim_txf_ah_ret = st.sidebar.slider(
    "11. 台指期盤後夜盤漲跌幅 (%)",
    min_value=-10.0,
    max_value=10.0,
    value=st.session_state.get("slider_txf_ah_r", 0.0),
    step=0.05,
    key="slider_txf_ah_r",
    help="近月臺指期盤後交易時段 (15:00 ~ 次日05:00) 收盤相較於夜盤開盤之漲跌幅 (%)",
)
sim_nsf_active = st.sidebar.selectbox(
    "12. 🛡️ 國安基金進場護盤期間 (虛擬變數 0/1)",
    options=[0, 1],
    format_func=lambda x: "1 (國安基金護盤期間)" if x == 1 else "0 (未護盤期間)",
    key="slider_nsf_active",
    help="第9次及第10次國安基金進場護盤全期間皆設為 1（第9次: 2025/4/9-2026/1/12；第10次: 2026/11/9起持續護盤中），其餘未護盤期間設為 0。",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📈 技術分析相關解釋變數區塊 (T-1 前一交易日確立)**")

sim_gap_ret = st.sidebar.slider(
    "13. 開盤跳空幅度 Gap (%)",
    min_value=-10.0,
    max_value=10.0,
    value=st.session_state.get("slider_gap_r", 0.0),
    step=0.05,
    key="slider_gap_r",
    help="前一交易日加權指數開盤相較於前日收盤之跳空漲跌幅 (%)",
)
sim_ma5_bias = st.sidebar.selectbox(
    "14. 5日均線乖離虛擬變數 (T-1: ma5_bias > 0)",
    options=[0, 1],
    format_func=lambda x: "1 (前一日 ma5_bias > 0，站上5日線)" if x == 1 else "0 (前一日 ma5_bias ≤ 0，跌破5日線)",
    key="slider_ma5_bias",
    help="虛擬變數：預測日前一交易日加權指數收盤之 5 日均線乖離率 > 0 設為 1，否則為 0。",
)
sim_ma10_bias = st.sidebar.selectbox(
    "15. 10日均線乖離虛擬變數 (T-1: ma10_bias > 0)",
    options=[0, 1],
    format_func=lambda x: "1 (前一日 ma10_bias > 0，站上10日線)" if x == 1 else "0 (前一日 ma10_bias ≤ 0，跌破10日線)",
    key="slider_ma10_bias",
    help="虛擬變數：預測日前一交易日加權指數收盤之 10 日均線乖離率 > 0 設為 1，否則為 0。",
)
sim_ma5_ma10 = st.sidebar.selectbox(
    "16. 均線多頭排列虛擬變數 (T-1: MA5 > MA10)",
    options=[0, 1],
    format_func=lambda x: "1 (前一日 MA5 > MA10，短均線多頭排列)" if x == 1 else "0 (前一日 MA5 ≤ MA10，短均線空頭排列)",
    key="slider_ma5_ma10",
    help="虛擬變數：預測日前一交易日加權指數 5 日均線大於 10 日均線 (MA5 > MA10) 設為 1，否則為 0。",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📉 借券放空籌碼面擴充變數 (T-1 前一交易日確立)**")
sim_sbl_5d_chg = st.sidebar.slider(
    "17. 近 5 日借券賣出餘額累計變動 (張)",
    min_value=-1500000.0,
    max_value=1500000.0,
    value=st.session_state.get("slider_sbl_5d_chg", 0.0),
    step=1000.0,
    key="slider_sbl_5d_chg",
    help="證交所 TWT93U 整體市場借券賣出當日餘額之近 5 個交易日累計增減張數 (T-1 減 T-6)。正值代表借券賣出餘額增加（偏空避險/放空），負值代表借券回補（偏多）。",
)

trans_sim_f = engine.transform_foreign_net_series(sim_foreign_net)
trans_sim_tx_diff = engine.transform_oi_series(sim_tx_oi_diff)

# 1. 計算外資期現幾何共振項: sign(Z_f + Z_dOI) * |Z_f * Z_dOI|
mu_f = engine.active_scaler.get("foreign_net", {}).get("mu", 0.0)
sig_f = engine.active_scaler.get("foreign_net", {}).get("sigma", 1.0)
mu_tx = engine.active_scaler.get("tx_oi_diff", {}).get("mu", 0.0)
sig_tx = engine.active_scaler.get("tx_oi_diff", {}).get("sigma", 1.0)
current_z_f = (trans_sim_f - mu_f) / (sig_f if sig_f > 0 else 1.0)
current_z_tx = (trans_sim_tx_diff - mu_tx) / (sig_tx if sig_tx > 0 else 1.0)
current_fx_inter = float(engine.compute_directional_interaction(current_z_f, current_z_tx))

# 2. 計算 [美股費半]*[台積電ADR] 幾何共振項: sign(Z_sox + Z_tsm) * |Z_sox * Z_tsm|
mu_sox = engine.active_scaler.get("sox_ret", {}).get("mu", 0.0)
sig_sox = engine.active_scaler.get("sox_ret", {}).get("sigma", 1.0)
mu_tsm = engine.active_scaler.get("tsm_ret", {}).get("mu", 0.0)
sig_tsm = engine.active_scaler.get("tsm_ret", {}).get("sigma", 1.0)
current_z_sox = (sim_sox_ret - mu_sox) / (sig_sox if sig_sox > 0 else 1.0)
current_z_tsm = (sim_tsm_ret - mu_tsm) / (sig_tsm if sig_tsm > 0 else 1.0)
current_sox_tsm_inter = float(engine.compute_directional_interaction(current_z_sox, current_z_tsm))

current_features = {
    "foreign_net": sim_foreign_net,
    "tx_oi_diff": sim_tx_oi_diff,
    "fx_inter": current_fx_inter,
    "usdtwd_ret": sim_usdtwd_ret,
    "sox_ret": sim_sox_ret,
    "tsm_ret": sim_tsm_ret,
    "sox_tsm_inter": current_sox_tsm_inter,
    "vix_chg": sim_vix_chg,
    "nvda_ret": sim_nvda_ret,
    "basis_ret": sim_basis_ret,
    "txf_ah_ret": sim_txf_ah_ret,
    "nsf_active": float(sim_nsf_active),
    # --- 技術分析相關變數區塊 ---
    "gap_ret": sim_gap_ret,
    "ma5_bias": float(sim_ma5_bias),
    "ma10_bias": float(sim_ma10_bias),
    "ma5_ma10": float(sim_ma5_ma10),
    # --- 借券放空籌碼面 ---
    "sbl_5d_chg": float(sim_sbl_5d_chg),
}

prob_up, contributions = engine.predict(current_features)

# -----------------------------------------------------------------------------
# 7. 主視覺顯示 (Dashboard View)
# -----------------------------------------------------------------------------
st.title("🎯 TAIEX 台灣加權指數次日漲跌機率監控儀表 (17因子-含近5日借券賣出餘額變動升級版)")

if use_fixed_weights:
    m_desc = (
        f"採固定專家權重 (17因子預設權重 | 訓練樣本: 前 {len(train_sample_df)} 營業日 | "
        f"波段實質勝率: {in_swing_win_rate:.1f}% [{in_win_swings}/{in_total_swings}筆] | 樣本外測試: 最近 30 個營業日)"
    )
elif engine.fit_accuracy is not None:
    if engine.auto_select_features:
        m_desc = (
            f"自動精選 {len(engine.selected_features)}/{len(engine.ALL_FEATURES)} 個關鍵解釋變數動態 MLE 配適 "
            f"(訓練樣本: 前 {len(train_sample_df)} 營業日 | 精選後波段實質勝率: {in_swing_win_rate:.1f}% [{in_win_swings}/{in_total_swings}筆] | "
            f"樣本外測試: 最近 30 個營業日)"
        )
    else:
        m_desc = (
            f"全 17 因子動態 MLE 配適 (訓練樣本: 前 {len(train_sample_df)} 營業日 | "
            f"波段實質勝率: {in_swing_win_rate:.1f}% [{in_win_swings}/{in_total_swings}筆] | 樣本外測試: 最近 30 個營業日)"
        )
else:
    m_desc = "採固定專家權重 (17因子)" 

fallback_warnings = []
if data["twse_actual_date"] != data["base_feature_date"]:
    fallback_warnings.append(
        f"TWSE 三大法人數據順延至 `{data['twse_actual_date']}`（台灣市場休市）"
    )
if data["taifex_actual_date"] != data["base_feature_date"]:
    fallback_warnings.append(
        f"TAIFEX 期貨留倉數據順延至 `{data['taifex_actual_date']}`（台灣市場休市）"
    )
if data["us_actual_date"] != data["base_feature_date"]:
    fallback_warnings.append(
        f"美股行情數據順延至 `{data['us_actual_date']}`（美股休市/回溯）"
    )

if fallback_warnings:
    st.info(
        "ℹ️ **交易日因天然災害/休市自動向前校正**：\n- "
        + "\n- ".join(fallback_warnings)
    )

if data.get("us_macro_event"):
    st.warning(
        f"🔔 **重大總經事件關注**：`{data['us_actual_date']}` 市場公布"
        f" **【{data['us_macro_event']}】**，請留意跨市場行情震盪對台股開盤之溢出效應！"
    )

nsf_banner_str = (
    "🟢 **國安基金進場護盤期間 (Dummy = 1)**"
    if sim_nsf_active == 1
    else "⚪ **非護盤期間 (Dummy = 0)**"
)

sl_banner_desc = (
    f"啟用停損（優化自動設定最佳停損：-{joint_best_stop_loss_pct:.1f}%，交易策略：不停利）"
    if enable_stop_loss_flag
    else "不需停損（交易策略：不停利、不設停損）"
)

st.markdown(
    f"""
> **系統運行狀態與跨市場因果檢驗 (17 變量 Logistic 升級模型)**  
> • **目標預測交易日 (T)**：`{data['target_date']}` | **模型權重模式**：`{m_desc}`  
> • **交易執行標的**：**臺指近月期貨 (TXF Front-Month)** | **執行策略**：**開盤價進場，訊號一致留倉、不連貫以預測日開盤平倉；{sl_banner_desc}**  
> • **架構切分**：**數據採樣 520 交易日 $\\rightarrow$ 前 490 日訓練尋優 (3×3最佳中心多方門檻: {joint_best_bull}% | 空方門檻: {joint_best_bear}% | 最佳停損: {f'-{joint_best_stop_loss_pct:.1f}%' if enable_stop_loss_flag else '無'})；後 30 日為純樣本外 (OOS) 盲測**  
> • **國安基金護盤狀態**：{nsf_banner_str} （第9次護盤期間：`2025/04/09 ~ 2026/01/12`；第10次護盤期間：`2026/11/09 起持續進行中` 皆設為 1，其餘為 0）  
> • **證券交易所 (TWSE)**：外資現貨買賣超 (擷取日期: `{data['twse_actual_date']}` | `狀態: {data['twse_msg']}`)  
> • **期貨交易所 (TAIFEX)**：外資單日未平倉增減量 ΔOI、期現貨價差率、近月台指期盤後夜盤漲跌幅 (擷取日期: `{data['taifex_actual_date']}` | `狀態: {data['taifex_msg']}`)  
> • **國際美股與匯率特徵 (yfinance)**：費半、TSM ADR、輝達 (NVDA)、費半×TSM幾何共振項、VIX變動 (ΔVIX)、USD/TWD (擷取日期: `{data['us_actual_date']}` | `狀態: {data['us_msg']}`)  
> • **技術分析特徵區塊 (T-1 前一交易日確立)**：開盤跳空幅度 (Gap %)、5日均線正乖離虛擬變數 (`ma5_bias > 0`)、10日均線正乖離虛擬變數 (`ma10_bias > 0`)、5日/10日均線多頭排列虛擬變數 (`MA5 > MA10`)  
> • **借券放空籌碼面 (TWSE TWT93U, T-1)**：近 5 日整體借券賣出餘額累計變動張數 (擷取日期: `{data.get('sbl_actual_date', data['twse_actual_date'])}` | `狀態: {data.get('sbl_msg', '已載入')}`)
"""
)

# -----------------------------------------------------------------------------
# 8. 提供 520 交易日真實歷史數據與「績效統計摘要表＋淨值走勢圖」下載
# -----------------------------------------------------------------------------
if not data["hist_df"].empty:
    current_user = st.session_state.get("current_user", "")
    allowed_download_users = ["admin", "analyst01", "manager"]

    if current_user in allowed_download_users:
        file_bytes, file_ext, mime_type = export_520_days_to_excel_with_summary(
            hist_df=data["hist_df"],
            hist_twse=data.get("hist_twse_df"),
            hist_taifex=data.get("hist_taifex_df"),
            engine=engine,
            best_bull=joint_best_bull,
            best_bear=joint_best_bear,
            cost_rate_pct=cost_rate_setting,
            best_stop_loss_pct=joint_best_stop_loss_pct,
            hist_sbl=data.get("hist_sbl_df"),
        )
        st.download_button(
            label=f"📥 下載 520 交易日台美數據 (含17因子明細、近5日借券賣出餘額變動、技術分析虛擬變數、盤後夜盤、績效摘要與淨值走勢圖) ({file_ext.upper()})",
            data=file_bytes,
            file_name=f"TAIEX_US_520_Days_17Factors_{data['target_date']}.{file_ext}",
            mime=mime_type,
            help="下載包含「前490日波段留倉績效摘要與原生淨值走勢圖」、「波段留倉淨值曲線數據」與「520日真實交易數據明細(含17因子、借券餘額與夜盤數據)」之完整試算表",
            use_container_width=True,
        )
    else:
        st.info(f"🔒 **權限限制**：您目前的身分 (`{current_user}`) 僅開放儀表板瀏覽權限，請升級帳號以解鎖 Excel 數據下載功能。")

st.markdown("---")

col_gauge, col_waterfall = st.columns([5, 5])

with col_gauge:
    st.plotly_chart(
        render_gauge(prob_up, target_date=data["target_date"]),
        use_container_width=True,
    )
    if prob_up >= (joint_best_bull / 100.0):
        st.markdown(
            "<h3 style='text-align: center;' class='text-bullish'>策略指令：開盤做多 / 多單續抱臺指期"
            f" (勝率 {prob_up*100:.1f}% ≥ {joint_best_bull}%)</h3>",
            unsafe_allow_html=True,
        )
    elif prob_up <= (joint_best_bear / 100.0):
        st.markdown(
            "<h3 style='text-align: center;' class='text-bearish'>策略指令：開盤放空 / 空單續抱臺指期"
            f" (勝率 {prob_up*100:.1f}% ≤ {joint_best_bear}%)</h3>",
            unsafe_allow_html=True,
        )
    elif prob_up >= 0.50:
        st.markdown(
            "<h3 style='text-align: center; color: #fbbf24;'>策略指令：緩衝區間 — 既有多單續抱 / 既有空單開盤平倉 / 空手觀望"
            f" (勝率 {prob_up*100:.1f}% 介於 50%~{joint_best_bull}%)</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h3 style='text-align: center; color: #94a3b8;'>策略指令：緩衝區間 — 既有空單續抱 / 既有多單開盤平倉 / 空手觀望"
            f" (勝率 {prob_up*100:.1f}% 介於 {joint_best_bear}%~50%)</h3>",
            unsafe_allow_html=True,
        )
    st.caption(f"⭐ **目標交易日 ({data['target_date']}) 為前 490 日模型訓練完成後、30 筆純樣本外測試之最新預測（訊號一致則留倉，不連貫則以預測日開盤價平倉）**。")

with col_waterfall:
    oi_tag = " (Signed-Log)" if "Signed-Log" in chip_transform_mode else " (水準項)"
    st.plotly_chart(
        render_waterfall(
            contributions,
            tag_extra=oi_tag,
            selected_features=engine.selected_features,
        ),
        use_container_width=True,
    )
    if engine.auto_select_features and len(engine.eliminated_features) > 0:
        st.caption(
            f"🧬 **自動特徵篩選結果**：已自 17 個候選變數中自動保留 **{len(engine.selected_features)} 個高影響力關鍵變數**，並將其餘 **{len(engine.eliminated_features)} 個低貢獻/共線性雜訊變數** 權重歸零剔除（訓練準確率：`{engine.full_13_accuracy:.1f}%` $\rightarrow$ **`{engine.fit_accuracy:.1f}%`**）。"
        )

with st.expander(
    f"🔍 檢視自動篩選後入選之 {len(engine.selected_features)} 個關鍵解釋變數與 Beta 權重常模 (前 {TRAIN_SAMPLE_DAYS} 交易日訓練樣本)"
):
    current_user = st.session_state.get("current_user", "")
    allowed_weight_users = ["admin", "analyst01", "manager"]

    if current_user in allowed_weight_users:
        feature_name_map = {
            "foreign_net": f"1. 外資現貨買賣超 (foreign_net, {'Signed-Log' if 'Signed-Log' in chip_transform_mode else '水準項'})",
            "tx_oi_diff": f"2. 單日未平倉增減變動量 (tx_oi_diff, {'Signed-Log' if 'Signed-Log' in chip_transform_mode else '水準項'})",
            "fx_inter": "3. 外資期現幾何共振項 (fx_inter: sign(Z1+Z2)*|Z1*Z2|)",
            "usdtwd_ret": "4. 美元兌台幣變動 (usdtwd_ret %)",
            "sox_ret": "5. 美股費半漲跌幅 (sox_ret %)",
            "tsm_ret": "6. 台積電 ADR 漲跌幅 (tsm_ret %)",
            "sox_tsm_inter": "7. 費半×台積電ADR幾何共振項 (sox_tsm_inter: sign(Z1+Z2)*|Z1*Z2|)",
            "vix_chg": "8. 美股 VIX 恐慌指數變動 (vix_chg ΔVIX)",
            "nvda_ret": "9. 輝達股票漲跌幅 (nvda_ret %)",
            "basis_ret": "10. 期現貨價差率 (basis_ret %)",
            "txf_ah_ret": "11. 台指期盤後夜盤漲跌幅 (txf_ah_ret %)",
            "nsf_active": "12. 國安基金進場護盤虛擬變數 (nsf_active: 1=護盤期間, 0=其餘)",
            # --- 技術分析相關變數區塊 ---
            "gap_ret": "13. [技術分析] 開盤跳空幅度 (gap_ret %, T-1)",
            "ma5_bias": "14. [技術分析] 5日均線正乖離虛擬變數 (ma5_bias: T-1 >0 則 1, 否則 0)",
            "ma10_bias": "15. [技術分析] 10日均線正乖離虛擬變數 (ma10_bias: T-1 >0 則 1, 否則 0)",
            "ma5_ma10": "16. [技術分析] 均線多頭排列虛擬變數 (ma5_ma10: T-1 MA5>MA10 則 1, 否則 0)",
            # --- 借券放空籌碼面 ---
            "sbl_5d_chg": f"17. [借券籌碼] 近 5 日借券賣出餘額累計變動 (sbl_5d_chg, {'Signed-Log' if 'Signed-Log' in chip_transform_mode else '水準項'}, 張, T-1)",
        }
        w_df = pd.DataFrame([
            {
                "特徵因子代碼與名稱": feature_name_map.get(k, k),
                "篩選狀態": "✅ 入選關鍵變數" if k in engine.selected_features else "⚪ 已自動剔除 (Beta=0)",
                "最佳化 Beta 權重": round(engine.active_beta.get(k, 0.0), 4),
                "影響力排序 (|Beta|)": round(abs(engine.active_beta.get(k, 0.0)), 4),
                "常模均值 (μ)": round(
                    engine.active_scaler.get(k, {}).get("mu", 0.0), 3
                ),
                "常模標準差 (σ)": round(
                    engine.active_scaler.get(k, {}).get("sigma", 1.0), 3
                ),
            }
            for k in engine.ALL_FEATURES
        ])
        st.dataframe(w_df, hide_index=True, use_container_width=True)
    else:
        st.info(f"🔒 **權限限制**：您目前的身分 (`{current_user}`) 無權限檢視模型內部 Beta 權重與常模參數，請以授權帳號登入。")

st.subheader(f"📊 盤前與開盤已知候選特徵數值一覽 (自動入選 {len(engine.selected_features)} 個關鍵變數 | 剔除 {len(engine.eliminated_features)} 個低貢獻變數)")

def get_bull_delta_color(contrib_val: float) -> str:
    return "inverse" if contrib_val >= 0 else "normal"

st.markdown("##### 🏦 籌碼面、國際美股、期貨夜盤與政策面特徵 (12 項)")
row1_m1, row1_m2, row1_m3, row1_m4 = st.columns(4)
row2_m5, row2_m6, row2_m7, row2_m8 = st.columns(4)
row3_m9, row3_m10, row3_m11, row3_m12 = st.columns(4)

# -----------------------------------------------------------------------------
# 第 1 列：外資籌碼與匯率 (1, 2, 3, 4)
# -----------------------------------------------------------------------------
with row1_m1:
    f_disp = f"{sim_foreign_net:+.1f} 億"
    if "Signed-Log" in chip_transform_mode:
        f_disp += f" (s-ln={trans_sim_f:.2f})"
    
    b_f = engine.active_beta.get('foreign_net', 0.0)
    arr_f = "▲" if sim_foreign_net >= 0 else "▼"
    c_f = contributions.get('foreign_net', 0.0)

    st.metric(
        "1. 外資現貨買賣超",
        f_disp,
        delta=f"{arr_f} Beta: {b_f:+.2f}",
        delta_color=get_bull_delta_color(c_f),
    )
    st.caption(f"來源: TWSE ({data['twse_actual_date']})")

with row1_m2:
    oi_diff_status = "單日增持多單" if sim_tx_oi_diff >= 0 else "單日增持空單"
    val_disp = f"{int(sim_tx_oi_diff):+,} 口"
    if "Signed-Log" in chip_transform_mode:
        val_disp += f" (s-ln={trans_sim_tx_diff:.2f})"
    
    b_tx = engine.active_beta.get('tx_oi_diff', 0.0)
    arr_tx = "▲" if sim_tx_oi_diff >= 0 else "▼"
    c_tx = contributions.get('tx_oi_diff', 0.0)

    st.metric(
        "2. 單日未平倉增減變動量",
        val_disp,
        delta=f"{arr_tx} {oi_diff_status} (Beta: {b_tx:+.2f})",
        delta_color=get_bull_delta_color(c_tx),
    )
    st.caption(f"來源: TAIFEX ΔOI ({data['taifex_actual_date']})")

with row1_m3:
    inter_status = (
        "雙多幾何共振"
        if current_fx_inter > 0.2
        else ("雙空幾何共振" if current_fx_inter < -0.2 else "中性平衡")
    )
    b_fx = engine.active_beta.get('fx_inter', 0.0)
    arr_fx = "▲" if current_fx_inter >= 0 else "▼"
    c_fx = contributions.get('fx_inter', 0.0)

    st.metric(
        "3. 外資期現幾何共振項",
        f"{current_fx_inter:+.2f}",
        delta=f"{arr_fx} {inter_status} (Beta: {b_fx:+.2f})",
        delta_color=get_bull_delta_color(c_fx),
    )
    st.caption("sign(Z現貨+ZΔOI) × |Z現貨×ZΔOI|")

with row1_m4:
    fx_status = "台幣貶值 (流出)" if sim_usdtwd_ret > 0 else ("台幣升值 (流入)" if sim_usdtwd_ret < 0 else "平盤持平")
    b_usdtwd = engine.active_beta.get('usdtwd_ret', 0.0)
    arr_fx_flow = "▲" if sim_usdtwd_ret >= 0 else "▼"
    c_usdtwd = contributions.get('usdtwd_ret', 0.0)

    st.metric(
        "4. USD/TWD 匯率變動",
        f"{sim_usdtwd_ret:+.2f}%",
        delta=f"{arr_fx_flow} {fx_status} (Beta: {b_usdtwd:+.2f})",
        delta_color=get_bull_delta_color(c_usdtwd),
    )
    st.caption("來源: yfinance (TWD=X)")

# -----------------------------------------------------------------------------
# 第 2 列：國際美股與恐慌波動 (5, 6, 7, 8)
# -----------------------------------------------------------------------------
with row2_m5:
    ev_tag = f" [{data['us_macro_event']}]" if data.get("us_macro_event") else ""
    sox_status = "費半收紅" if sim_sox_ret > 0 else ("費半收黑" if sim_sox_ret < 0 else "平盤")
    b_sox = engine.active_beta.get('sox_ret', 0.0)
    arr_sox = "▲" if sim_sox_ret >= 0 else "▼"
    c_sox = contributions.get('sox_ret', 0.0)

    st.metric(
        "5. 美股費半 (SOX)",
        f"{sim_sox_ret:+.2f}%",
        delta=f"{arr_sox} {sox_status} (Beta: {b_sox:+.2f})",
        delta_color=get_bull_delta_color(c_sox),
    )
    st.caption(f"來源: yfinance ({data['us_actual_date']}){ev_tag}")

with row2_m6:
    tsm_status = "美股 ADR 上漲" if sim_tsm_ret > 0 else ("美股 ADR 下跌" if sim_tsm_ret < 0 else "平盤持平")
    b_tsm = engine.active_beta.get('tsm_ret', 0.0)
    arr_tsm = "▲" if sim_tsm_ret >= 0 else "▼"
    c_tsm = contributions.get('tsm_ret', 0.0)

    st.metric(
        "6. 台積電 ADR (TSM)",
        f"{sim_tsm_ret:+.2f}%",
        delta=f"{arr_tsm} {tsm_status} (Beta: {b_tsm:+.2f})",
        delta_color=get_bull_delta_color(c_tsm),
    )
    st.caption(f"來源: yfinance ({data['us_actual_date']})")

with row2_m7:
    st_inter_status = (
        "半導體雙多共振"
        if current_sox_tsm_inter > 0.2
        else ("半導體雙空共振" if current_sox_tsm_inter < -0.2 else "中性波動")
    )
    b_soxtsm = engine.active_beta.get('sox_tsm_inter', 0.0)
    arr_soxtsm = "▲" if current_sox_tsm_inter >= 0 else "▼"
    c_soxtsm = contributions.get('sox_tsm_inter', 0.0)

    st.metric(
        "7. 費半×TSM 幾何共振項",
        f"{current_sox_tsm_inter:+.2f}",
        delta=f"{arr_soxtsm} {st_inter_status} (Beta: {b_soxtsm:+.2f})",
        delta_color=get_bull_delta_color(c_soxtsm),
    )
    st.caption("sign(Z費半+ZTSM) × |Z費半×ZTSM|")

with row2_m8:
    vix_status = "恐慌升溫" if sim_vix_chg > 0 else ("恐慌降溫" if sim_vix_chg < 0 else "波動持平")
    b_vix = engine.active_beta.get('vix_chg', 0.0)
    arr_vix = "▲" if sim_vix_chg >= 0 else "▼"
    c_vix = contributions.get('vix_chg', 0.0)

    st.metric(
        "8. VIX 恐慌指數變動 (ΔVIX)",
        f"{sim_vix_chg:+.2f}",
        delta=f"{arr_vix} {vix_status} (Beta: {b_vix:+.2f})",
        delta_color=get_bull_delta_color(c_vix),
    )
    st.caption(f"來源: yfinance ΔVIX ({data['us_actual_date']})")

# -----------------------------------------------------------------------------
# 第 3 列：輝達、期現貨夜盤與國安基金 (9, 10, 11, 12)
# -----------------------------------------------------------------------------
with row3_m9:
    nvda_status = "輝達上漲" if sim_nvda_ret > 0 else ("輝達下跌" if sim_nvda_ret < 0 else "平盤持平")
    b_nvda = engine.active_beta.get('nvda_ret', 0.0)
    arr_nvda = "▲" if sim_nvda_ret >= 0 else "▼"
    c_nvda = contributions.get('nvda_ret', 0.0)

    st.metric(
        "9. 輝達股票漲跌幅 (NVDA)",
        f"{sim_nvda_ret:+.2f}%",
        delta=f"{arr_nvda} {nvda_status} (Beta: {b_nvda:+.2f})",
        delta_color=get_bull_delta_color(c_nvda),
    )
    st.caption(f"來源: yfinance ({data['us_actual_date']})")

with row3_m10:
    basis_status = "正價差 (偏多)" if sim_basis_ret > 0 else ("逆價差 (避險)" if sim_basis_ret < 0 else "價差持平")
    b_basis = engine.active_beta.get('basis_ret', 0.0)
    arr_basis = "▲" if sim_basis_ret >= 0 else "▼"
    c_basis = contributions.get('basis_ret', 0.0)

    st.metric(
        "10. 期現貨價差率 (Basis)",
        f"{sim_basis_ret:+.2f}%",
        delta=f"{arr_basis} {basis_status} (Beta: {b_basis:+.2f})",
        delta_color=get_bull_delta_color(c_basis),
    )
    st.caption("(預測日夜盤收盤 - 前日加權收盤) / 前日加權收盤")

with row3_m11:
    ah_status = "夜盤收紅" if sim_txf_ah_ret > 0 else ("夜盤收黑" if sim_txf_ah_ret < 0 else "夜盤平盤")
    b_ah = engine.active_beta.get('txf_ah_ret', 0.0)
    arr_ah = "▲" if sim_txf_ah_ret >= 0 else "▼"
    c_ah = contributions.get('txf_ah_ret', 0.0)

    st.metric(
        "11. 台指期盤後夜盤漲跌幅",
        f"{sim_txf_ah_ret:+.2f}%",
        delta=f"{arr_ah} {ah_status} (Beta: {b_ah:+.2f})",
        delta_color=get_bull_delta_color(c_ah),
    )
    st.caption("來源: TAIFEX 盤後時段 (Parquet快取)")

with row3_m12:
    nsf_label = "護盤期間 (1)" if sim_nsf_active == 1 else "未護盤 (0)"
    b_nsf = engine.active_beta.get('nsf_active', 0.0)
    arr_nsf = "▲" if b_nsf >= 0 else "▼"
    c_nsf = contributions.get('nsf_active', 0.0)

    st.metric(
        "12. 國安基金護盤變數",
        nsf_label,
        delta=f"{arr_nsf} Beta: {b_nsf:+.2f}",
        delta_color=get_bull_delta_color(c_nsf),
    )
    st.caption("第9次:25/4/9-26/1/12 | 第10次:26/11/9起")

# -----------------------------------------------------------------------------
# 獨立區塊（第 4 列）：技術分析相關解釋變數 (13, 14, 15, 16) — 嚴格取預測日前一交易日 (T-1)
# -----------------------------------------------------------------------------
st.markdown("##### 📈 技術分析相關解釋變數區塊 (4 項 — 皆為預測日前一交易日 T-1 確立數據)")
tech_m1, tech_m2, tech_m3, tech_m4 = st.columns(4)

with tech_m1:
    gap_status = "跳空開高" if sim_gap_ret > 0 else ("跳空開低" if sim_gap_ret < 0 else "平盤開出")
    b_gap = engine.active_beta.get('gap_ret', 0.0)
    arr_gap = "▲" if sim_gap_ret >= 0 else "▼"
    c_gap = contributions.get('gap_ret', 0.0)

    st.metric(
        "13. 開盤跳空幅度 (Gap)",
        f"{sim_gap_ret:+.2f}%",
        delta=f"{arr_gap} {gap_status} (Beta: {b_gap:+.2f})",
        delta_color=get_bull_delta_color(c_gap),
    )
    st.caption("T-1: (開盤 - 前收) / 前收")

with tech_m2:
    ma5_label = "站上5日線 (1)" if sim_ma5_bias == 1 else "跌破5日線 (0)"
    b_ma5 = engine.active_beta.get('ma5_bias', 0.0)
    arr_ma5 = "▲" if sim_ma5_bias == 1 else "▼"
    c_ma5 = contributions.get('ma5_bias', 0.0)

    st.metric(
        "14. 5日均線乖離虛擬變數",
        ma5_label,
        delta=f"{arr_ma5} Beta: {b_ma5:+.2f}",
        delta_color=get_bull_delta_color(c_ma5),
    )
    st.caption(f"T-1: ma5_bias > 0 則 1 (實測: {data.get('ma5_b_pct', 0.0):+.2f}%)")

with tech_m3:
    ma10_label = "站上10日線 (1)" if sim_ma10_bias == 1 else "跌破10日線 (0)"
    b_ma10 = engine.active_beta.get('ma10_bias', 0.0)
    arr_ma10 = "▲" if sim_ma10_bias == 1 else "▼"
    c_ma10 = contributions.get('ma10_bias', 0.0)

    st.metric(
        "15. 10日均線乖離虛擬變數",
        ma10_label,
        delta=f"{arr_ma10} Beta: {b_ma10:+.2f}",
        delta_color=get_bull_delta_color(c_ma10),
    )
    st.caption(f"T-1: ma10_bias > 0 則 1 (實測: {data.get('ma10_b_pct', 0.0):+.2f}%)")

with tech_m4:
    ma510_label = "均線多頭 (1)" if sim_ma5_ma10 == 1 else "均線空頭 (0)"
    b_ma510 = engine.active_beta.get('ma5_ma10', 0.0)
    arr_ma510 = "▲" if sim_ma5_ma10 == 1 else "▼"
    c_ma510 = contributions.get('ma5_ma10', 0.0)

    st.metric(
        "16. 5日/10日均線排列變數",
        ma510_label,
        delta=f"{arr_ma510} Beta: {b_ma510:+.2f}",
        delta_color=get_bull_delta_color(c_ma510),
    )
    st.caption("T-1: if MA5 > MA10 then 1 else 0")

# -----------------------------------------------------------------------------
# 獨立區塊（第 5 列）：借券放空籌碼面擴充解釋變數 (17) — 嚴格取預測日前一交易日 (T-1)
# -----------------------------------------------------------------------------
st.markdown("##### 📉 借券放空籌碼面擴充變數 (1 項 — 證交所 TWT93U 預測日前一交易日 T-1 確立)")
sbl_m1, _, _, _ = st.columns(4)
with sbl_m1:
    sbl_status = "借券賣出增加 (偏空)" if sim_sbl_5d_chg > 0 else ("借券餘額回補 (偏多)" if sim_sbl_5d_chg < 0 else "餘額持平")
    trans_sim_sbl = engine.transform_sbl_series(sim_sbl_5d_chg)
    sbl_disp = f"{int(sim_sbl_5d_chg):+,} 張"
    if "Signed-Log" in chip_transform_mode:
        sbl_disp += f" (s-ln={trans_sim_sbl:.2f})"
    b_sbl = engine.active_beta.get('sbl_5d_chg', 0.0)
    arr_sbl = "▲" if sim_sbl_5d_chg >= 0 else "▼"
    c_sbl = contributions.get('sbl_5d_chg', 0.0)

    st.metric(
        "17. 近5日借券賣出餘額變動",
        sbl_disp,
        delta=f"{arr_sbl} {sbl_status} (Beta: {b_sbl:+.2f})",
        delta_color=get_bull_delta_color(c_sbl),
    )
    st.caption(f"來源: TWSE TWT93U ({data.get('sbl_actual_date', data['twse_actual_date'])})")

# -----------------------------------------------------------------------------
# 9. 臺指期波段留倉策略 (訊號一致留倉、不連貫開盤平倉) 過往 30 個台股交易日純樣本外 (OOS) 結算
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader(
    "📈 臺指期波段留倉 (買賣訊號一致留倉 | 訊號不連貫以預測日開盤價平倉) 過往 30 個台股交易日純樣本外 (OOS) 結算"
)
st.caption(
    "🛡️ **無偏估計說明**：本段 30 個交易日（第 491 至 520 筆資料）**完全未參與 17 變量模型的任何訓練與門檻尋優**，純粹用於驗證前 490 天模型的真實樣本外預測能力。"
)

hist_twse = data.get("hist_twse_df", pd.DataFrame())
hist_taifex = data.get("hist_taifex_df", pd.DataFrame())
hist_sbl = data.get("hist_sbl_df", pd.DataFrame())

if not oos_eval_df.empty and len(oos_eval_df) >= 5:
    ten_days_df = oos_eval_df.tail(OOS_TEST_DAYS).copy()
    calc_probs = []
    decision_us_dates = []
    decision_us_events = []

    for i in range(len(ten_days_df)):
        current_date_idx = ten_days_df.index[i]

        # 交易日當日 (current_date_idx, T) 的開盤投資決策：
        # 1. 台股日內、夜盤、價差、乖離與籌碼特徵：嚴格取自前一交易日 (prior_rows, index < current_date_idx, T-1)
        # 2. 美股隔夜特徵：curr_row 已於資料管線嚴格對齊 us_raw.index < dt (台灣 T 日凌晨 04:00 剛收盤之 T-1 美股)
        curr_row = ten_days_df.loc[current_date_idx]
        prior_rows = hist_df_full[hist_df_full.index < current_date_idx]
        if not prior_rows.empty:
            prev_tw_row = prior_rows.iloc[-1]
            prev_date_idx = prior_rows.index[-1]
        else:
            prev_tw_row = curr_row
            prev_date_idx = current_date_idx

        f_val = 5.0
        if not hist_twse.empty and "foreign_net" in hist_twse.columns:
            matched_twse = hist_twse[hist_twse.index < current_date_idx].dropna(subset=["foreign_net"])
            if not matched_twse.empty:
                f_val = float(matched_twse.iloc[-1]["foreign_net"])

        tx_oi_diff_val = 0.0
        if not hist_taifex.empty and "tx_foreign_oi" in hist_taifex.columns:
            matched_taifex = hist_taifex[hist_taifex.index < current_date_idx].dropna(subset=["tx_foreign_oi"])
            if len(matched_taifex) >= 2:
                tx_oi_diff_val = float(matched_taifex.iloc[-1]["tx_foreign_oi"] - matched_taifex.iloc[-2]["tx_foreign_oi"])

        sbl_5d_chg_val = 0.0
        if not hist_sbl.empty and "sbl_balance" in hist_sbl.columns:
            matched_sbl = hist_sbl[hist_sbl.index < current_date_idx].dropna(subset=["sbl_balance"])
            if len(matched_sbl) >= 6:
                sbl_5d_chg_val = float(matched_sbl.iloc[-1]["sbl_balance"] - matched_sbl.iloc[-6]["sbl_balance"])
            elif len(matched_sbl) >= 2:
                sbl_5d_chg_val = float(matched_sbl.iloc[-1]["sbl_balance"] - matched_sbl.iloc[0]["sbl_balance"])

        nsf_val = float(check_is_nsf_active(prev_date_idx))

        hist_feat = {
            "foreign_net": f_val,
            "tx_oi_diff": tx_oi_diff_val,
            "usdtwd_ret": float(curr_row.get("usdtwd_ret", 0.0)),
            "sox_ret": float(curr_row["sox_ret"]),
            "tsm_ret": float(curr_row["tsm_ret"]),
            "nvda_ret": float(curr_row.get("nvda_ret", 0.0)),
            "vix_chg": float(curr_row.get("vix_chg", 0.0)),
            "basis_ret": float(curr_row.get("basis_ret", prev_tw_row.get("basis_ret", 0.0))),
            "txf_ah_ret": float(curr_row.get("txf_ah_ret", prev_tw_row.get("txf_ah_ret", 0.0))),
            "nsf_active": nsf_val,
            # --- 技術分析相關變數區塊 (嚴格取 T-1 前一交易日數據) ---
            "gap_ret": float(prev_tw_row.get("gap_ret", 0.0)),
            "ma5_bias": float(prev_tw_row.get("ma5_bias", 0.0)),
            "ma10_bias": float(prev_tw_row.get("ma10_bias", 0.0)),
            "ma5_ma10": float(prev_tw_row.get("ma5_ma10", 0.0)),
            # --- 借券放空籌碼面 (嚴格取 T-1 前一交易日確立數據) ---
            "sbl_5d_chg": sbl_5d_chg_val,
        }

        p, _ = engine.predict(hist_feat)
        calc_probs.append(round(p, 4))
        decision_us_dates.append(curr_row.get("us_date", prev_date_idx.date()))
        decision_us_events.append(curr_row.get("us_event", ""))

    ten_days_df["Prob_Up"] = calc_probs
    date_labels = ten_days_df.index.strftime("%Y-%m-%d").tolist()
    probs_pct = [p * 100 for p in ten_days_df["Prob_Up"]]

    txf_open_prices = ten_days_df["TXF_Open"].tolist()
    txf_close_prices = ten_days_df["TXF_Close"].tolist()
    gaps_list = ten_days_df["gap_ret"].tolist()
    filled_list = ten_days_df["gap_filled"].tolist()
    us_dates_list = decision_us_dates
    us_events_list = decision_us_events

    valid_gaps = [g for g in gaps_list if not np.isnan(g)]
    valid_fills = [f for f in filled_list if not np.isnan(f)]

    avg_gap = np.mean(valid_gaps) if valid_gaps else 0.0
    max_gap_up = max([g for g in valid_gaps if g > 0], default=0.0)
    max_gap_down = min([g for g in valid_gaps if g < 0], default=0.0)
    gap_fill_rate = (
        (sum(valid_fills) / len(valid_fills) * 100.0)
        if valid_fills
        else 0.0
    )

    cont_count = 0
    valid_pairs = 0
    for g, r in zip(gaps_list, ten_days_df["ret_open_to_close"].tolist()):
        if not np.isnan(g) and not np.isnan(r) and abs(g) >= 0.05:
            valid_pairs += 1
            if (g > 0 and r > 0) or (g < 0 and r < 0):
                cont_count += 1
    cont_rate = (
        (cont_count / valid_pairs * 100.0) if valid_pairs > 0 else 0.0
    )

    st.markdown(
        "##### 📌 樣本外 30 個台股交易日開盤跳空統計看板 (Gap Statistics)"
    )
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.metric(
            "平均跳空幅度 (Mean Gap)",
            f"{avg_gap:+.2f}%",
            delta="偏向跳空開高" if avg_gap >= 0 else "偏向跳空開低",
            delta_color="normal" if avg_gap >= 0 else "inverse",
        )
    with g2:
        st.metric(
            "最大向上 / 向下跳空",
            f"{max_gap_up:+.2f}% / {max_gap_down:+.2f}%",
            delta=f"極端跨距: {max_gap_up - max_gap_down:.2f}%",
            delta_color="off",
        )
    with g3:
        st.metric(
            "盤中缺口完全回補率",
            f"{gap_fill_rate:.1f}%",
            delta=f"30 日中回補 {sum(valid_fills)} 天",
            delta_color="normal" if gap_fill_rate >= 60 else "off",
        )
    with g4:
        st.metric(
            "跳空同向延續率",
            f"{cont_rate:.1f}%",
            delta=f"順勢慣性 (有效 {valid_pairs} 天)",
            delta_color="normal" if cont_rate >= 50 else "inverse",
        )

    with st.expander(
        "📊 檢視跳空幅度 vs. 盤中漲跌散布圖與 OLS 迴歸線 (Gap Scatter Plot)",
        expanded=False,
    ):
        scatter_fig = render_gap_scatter(
            date_labels,
            gaps_list,
            ten_days_df["ret_open_to_close"].tolist(),
            filled_list,
        )
        if scatter_fig is not None:
            st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("---")


    ctrl_col0, ctrl_col1, ctrl_col2, ctrl_col_sl, ctrl_col3 = st.columns([2.6, 2.5, 2.5, 2.5, 2.2])
    with ctrl_col0:
        auto_tune_thresholds = st.checkbox(
            "⚡ 自動套用前 490 日最佳門檻與停損%",
            value=True,
            help=f"套用前 {TRAIN_SAMPLE_DAYS} 交易日訓練樣本下，所有 3×3 九宮格範圍內 Σ(準確度×波段留倉淨利) 最大之最佳中心門檻與最佳停損百分比",
        )
    with ctrl_col1:
        if auto_tune_thresholds:
            threshold_bull = joint_best_bull
            st.metric(
                "🎯 多方波段門檻 (3×3中心)",
                f"{threshold_bull}%",
                delta=f"3×3九宮格總分: {joint_best_score:+.2f}",
            )
        else:
            threshold_bull = st.slider(
                "🎯 多方波段發動門檻 (%)", 60, 75, max(60, min(75, joint_best_bull)), step=1
            )
    with ctrl_col2:
        if auto_tune_thresholds:
            threshold_bear = joint_best_bear
            st.metric(
                "🛡️ 空方波段門檻 (3×3中心)",
                f"{threshold_bear}%",
                delta=f"前{TRAIN_SAMPLE_DAYS}日淨利: {joint_best_net_ret:+.1f}%",
            )
        else:
            threshold_bear = st.slider(
                "🛡️ 空方波段防禦門檻 (%)", 25, 40, max(25, min(40, joint_best_bear)), step=1
            )
    with ctrl_col_sl:
        if not enable_stop_loss_flag:
            active_stop_loss_pct = 0.0
            st.metric(
                "🛑 停損設定 (不停利)",
                "不需停損",
                delta="左側已設為不需停損",
                delta_color="off",
            )
        elif auto_tune_thresholds:
            active_stop_loss_pct = joint_best_stop_loss_pct
            st.metric(
                "🛑 優化最佳停損% (不停利)",
                f"-{active_stop_loss_pct:.1f}%",
                delta="自動尋優最佳停損百分比",
                delta_color="normal",
            )
        else:
            active_stop_loss_pct = st.slider(
                "🛑 手動停損百分比 (%)",
                min_value=0.5,
                max_value=5.0,
                value=float(joint_best_stop_loss_pct if joint_best_stop_loss_pct > 0 else 2.0),
                step=0.5,
            )
    with ctrl_col3:
        cost_rate_pct = st.number_input(
            "💸 單趟摩擦成本率 (%)",
            min_value=0.00,
            max_value=0.20,
            value=0.03,
            step=0.005,
            format="%.3f",
            help="包含期交稅、手續費與進出滑價（建倉與平倉各扣單趟成本，留倉續抱期間不重複扣費）",
        )

    with st.expander(
        f"🔥 檢視前 {TRAIN_SAMPLE_DAYS} 交易日訓練樣本門檻評分熱力圖 (目標: 3×3 九宮格內 Σ(準確度×波段留倉淨利) 最大之中心點)",
        expanded=False,
    ):
        if not joint_score_matrix.empty:
            st.plotly_chart(
                render_joint_threshold_heatmap(
                    joint_score_matrix,
                    joint_text_matrix,
                    joint_best_bull,
                    joint_best_bear,
                    sample_days=TRAIN_SAMPLE_DAYS,
                    best_stop_loss_pct=active_stop_loss_pct,
                ),
                use_container_width=True,
            )

    sl_rule_text = (
        f"當盤中部位虧損達 **-{active_stop_loss_pct:.1f}%** 以上時即刻啟動停損平倉（**策略固定不停利**）"
        if active_stop_loss_pct > 0
        else "**不設停損且不停利**，全數依訊號不連貫於預測日開盤平倉"
    )
    st.caption(
        f"💡 **純樣本外波段留倉會計基準（含雙門檻緩衝）**：以 **前 490 天訓練得出之 3×3 九宮格最佳中心門檻 ({threshold_bull}% / {threshold_bear}%)** 於最近 30 個營業日執行：**開盤價進場，持有多單時 p ≥ 50% 續抱、持有空單時 p ≤ 50% 續抱（免重複扣費），跌破/突破 50% 中軸或反轉時以預測日開盤價平倉；{sl_rule_text}**（單趟進場或平倉各扣除 **{cost_rate_pct:.3f}%** 摩擦成本）。"
    )

    (
        signal_types,
        positions,
        gross_returns,
        friction_costs,
        net_returns,
        validation_status,
        active_signals,
        correct_signals,
    ) = simulate_day_trading_pnl(
        probs_pct,
        ten_days_df,
        threshold_bull,
        threshold_bear,
        cost_rate_pct,
        stop_loss_pct=active_stop_loss_pct,
    )

    txf_high_prices = (
        ten_days_df["TXF_High"].tolist()
        if "TXF_High" in ten_days_df.columns
        else [max(o, c) for o, c in zip(txf_open_prices, txf_close_prices)]
    )
    txf_low_prices = (
        ten_days_df["TXF_Low"].tolist()
        if "TXF_Low" in ten_days_df.columns
        else [min(o, c) for o, c in zip(txf_open_prices, txf_close_prices)]
    )

    trades_df = extract_day_trades(
        date_labels,
        txf_open_prices,
        txf_close_prices,
        positions,
        gross_returns,
        friction_costs,
        net_returns,
        high_prices=txf_high_prices,
        low_prices=txf_low_prices,
        signal_types=signal_types,
        stop_loss_pct=active_stop_loss_pct,
    )

    bar_colors = ["#f43f5e" if r >= 0 else "#10b981" for r in gross_returns]
    bar_opacities = [0.85 if (pos != 0 or abs(r) > 1e-9 or c > 0) else 0.3 for pos, r, c in zip(positions, gross_returns, friction_costs)]

    gross_cum = [0.0]
    net_cum = [0.0]

    for g_r, n_r in zip(gross_returns, net_returns):
        gross_cum.append(gross_cum[-1] + g_r)
        net_cum.append(net_cum[-1] + n_r)

    gross_equity = gross_cum[1:]
    net_equity = net_cum[1:]

    net_series = pd.Series(net_equity)
    rolling_max = np.maximum.accumulate(net_series)
    drawdowns = net_series - rolling_max
    max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0.0

    total_net_return = net_equity[-1]
    total_gross_return = gross_equity[-1]
    total_cost_spent = sum(friction_costs)

    # 樣本外勝率改採波段實質淨損益來計算（實質淨損益 > 0 的波段次數佔波段總次數的百分比）
    total_swing_trades = len(trades_df) if not trades_df.empty else 0
    winning_swing_trades = (
        int((trades_df["_net_val"] > 0).sum())
        if (not trades_df.empty and "_net_val" in trades_df.columns)
        else 0
    )
    win_rate = (
        (winning_swing_trades / total_swing_trades * 100.0)
        if total_swing_trades > 0
        else 0.0
    )

    valid_net_returns = [
        nr for nr, gr, c in zip(net_returns, gross_returns, friction_costs) if (abs(gr) > 1e-9 or c > 0)
    ]
    sample_n = len(valid_net_returns)
    rf_daily = (1.5 / 100.0) / 252.0

    if sample_n > 1:
        ret_series = pd.Series(valid_net_returns) / 100.0
        ret_std = ret_series.std()
        excess_mean = (ret_series - rf_daily).mean()
        sharpe_ratio = (
            (excess_mean / ret_std * np.sqrt(252)) if ret_std > 0 else 0.0
        )

        annualized_net_return = (total_net_return / max(float(len(ten_days_df)), 1.0)) * 252.0
        calmar_ratio = (
            (annualized_net_return / abs(max_drawdown))
            if abs(max_drawdown) > 0
            else (annualized_net_return if annualized_net_return > 0 else 0.0)
        )

        gains = sum([r for r in valid_net_returns if r > 0])
        losses = abs(sum([r for r in valid_net_returns if r < 0]))
        profit_factor = (
            (gains / losses)
            if losses > 0
            else (99.0 if gains > 0 else 0.0)
        )
    else:
        sharpe_ratio = 0.0
        calmar_ratio = 0.0
        profit_factor = 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "30日樣本外波段勝率",
            f"{win_rate:.1f}%" if total_swing_trades > 0 else "無觸發",
            delta=f"共 {total_swing_trades} 筆波段 / 獲利 {winning_swing_trades} 筆 (實質淨損益>0)",
            delta_color="normal" if win_rate >= 60 else "inverse",
        )
    with k2:
        st.metric(
            "樣本外累積淨報酬 (扣成本)",
            f"{total_net_return:+.2f}%",
            delta=f"毛利: {total_gross_return:+.2f}%",
            delta_color="normal" if total_net_return >= 0 else "inverse",
        )
    with k3:
        st.metric(
            "倉位與純停損機制 (不停利)",
            f"一致留倉 / -{active_stop_loss_pct:.1f}%停損" if active_stop_loss_pct > 0 else "一致留倉 / 不設停損",
            delta=f"虧損達-{active_stop_loss_pct:.1f}%即停損(不停利)" if active_stop_loss_pct > 0 else "訊號不連貫開盤平倉(不停利)",
            delta_color="off",
        )
    with k4:
        st.metric(
            "累計手續與稅費損耗",
            f"-{total_cost_spent:.3f}%",
            delta=f"有效參與日數: {active_signals} 日",
            delta_color="off",
        )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric(
            "樣本外最大回撤 (MDD)",
            f"{max_drawdown:.2f}%",
            delta="純樣本外盲測",
            delta_color="inverse",
        )
    with r2:
        st.metric(
            "樣本外年化夏普比率",
            f"{sharpe_ratio:.2f}",
            delta="報酬波動比 (Rf=1.5%)",
            delta_color="normal" if sharpe_ratio >= 1.0 else "off",
        )
    with r3:
        st.metric(
            "樣本外卡瑪比率",
            f"{calmar_ratio:.2f}",
            delta="年化淨報酬 / |MDD|",
            delta_color="normal" if calmar_ratio >= 2.0 else "off",
        )
    with r4:
        pf_str = (
            f"{profit_factor:.2f}" if profit_factor < 90 else "∞ (無虧損)"
        )
        st.metric(
            "樣本外獲利因子",
            pf_str,
            delta="總獲利 / 總虧損",
            delta_color="normal" if profit_factor >= 1.5 else "off",
        )

    fig_verify = go.Figure()

    fig_verify.add_trace(
        go.Scatter(
            x=date_labels,
            y=probs_pct,
            mode="lines+markers+text",
            name="樣本外預測勝率 P(Up)",
            line=dict(color="#f59e0b", width=3),
            text=[f"{p:.1f}%" for p in probs_pct],
            textposition="top center",
            yaxis="y1",
        )
    )

    fig_verify.add_trace(
        go.Bar(
            x=date_labels,
            y=gross_returns,
            name="臺指期單日波段毛損益 (%) [訊號一致留倉、不連貫開盤平倉]",
            marker=dict(color=bar_colors, opacity=bar_opacities),
            text=[
                (
                    f"{r:+.2f}%"
                    if not np.isnan(r) and abs(r) > 0.001
                    else ("0.00%" if not np.isnan(r) else "待結算")
                )
                for r in gross_returns
            ],
            textposition="auto",
            yaxis="y2",
        )
    )

    fig_verify.add_hline(
        y=threshold_bull,
        line_dash="dot",
        line_color="rgba(244, 63, 94, 0.7)",
        annotation_text=f"多方波段門檻 ({threshold_bull}%)",
        annotation_position="bottom right",
        annotation_font_color="#f43f5e",
    )
    fig_verify.add_hline(
        y=threshold_bear,
        line_dash="dot",
        line_color="rgba(16, 185, 129, 0.7)",
        annotation_text=f"空方波段門檻 ({threshold_bear}%)",
        annotation_position="top right",
        annotation_font_color="#10b981",
    )

    fig_verify.update_layout(
        title=(
            "<b>最近 30 個台股交易日【純樣本外預測勝率 vs 臺指期波段留倉損益】疊加比對</b>"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        xaxis=dict(title="樣本外交易日 (T)", gridcolor="#334155"),
        yaxis=dict(
            title="樣本外預測勝率 P(Up)",
            range=[0, 105],
            gridcolor="#334155",
            ticksuffix="%",
        ),
        yaxis2=dict(
            title="當日近月臺指期波段損益 (%)",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            ticksuffix="%",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=370,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_verify, use_container_width=True)

    fig_equity = go.Figure()

    fig_equity.add_trace(
        go.Scatter(
            x=date_labels,
            y=net_equity,
            mode="lines+markers",
            name="樣本外累積淨報酬 (扣除摩擦成本)",
            line=dict(color="#38bdf8", width=3),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.15)",
        )
    )

    fig_equity.add_trace(
        go.Scatter(
            x=date_labels,
            y=gross_equity,
            mode="lines",
            name="樣本外毛報酬 (未扣摩擦成本)",
            line=dict(color="#818cf8", width=1.8, dash="dot"),
        )
    )

    fig_equity.add_hline(
        y=0, line_width=1, line_color="#64748b", line_dash="solid"
    )

    fig_equity.update_layout(
        title=(
            "<b>最近 30 日純樣本外波段留倉累積模擬淨值曲線 (OOS Equity Curve) | Sharpe:"
            f" {sharpe_ratio:.2f} | Calmar: {calmar_ratio:.2f} | MDD:"
            f" {max_drawdown:.2f}%</b>"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        xaxis=dict(title="樣本外交易日 (T)", gridcolor="#334155"),
        yaxis=dict(
            title="累積報酬率 (%)",
            gridcolor="#334155",
            ticksuffix="%",
            zeroline=True,
            zerolinecolor="#64748b",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=330,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    if not trades_df.empty:
        st.markdown("##### 📋 最近 30 日純樣本外逐筆波段留倉交易明細 (OOS Swing-Trades | 訊號不連貫以預測日開盤價平倉)")
        display_trades = trades_df.drop(
            columns=["_net_val"], errors="ignore"
        )
        st.dataframe(
            display_trades, use_container_width=True, hide_index=True
        )

    st.markdown(
        "##### 📑 最近 30 日純樣本外波段留倉市價結算流水帳 (含美股同日對齊、國安基金狀態與重大事件)"
    )
    pos_labels = [
        (
            "🟢 多單留倉 (+1)"
            if p == 1
            else ("🔴 空單留倉 (-1)" if p == -1 else "⚪ 空手無部位 (0)")
        )
        for p in positions
    ]

    us_align_labels = (
        [
            f"{usd}" + (f" ({ev})" if ev else "")
            for usd, ev in zip(us_dates_list, us_events_list)
        ]
        if len(us_dates_list) == len(date_labels)
        else ["-"] * len(date_labels)
    )

    nsf_status_labels = [
        "🛡️ 護盤期間(1)" if check_is_nsf_active(pd.to_datetime(d).date()) == 1 else "未護盤(0)"
        for d in date_labels
    ]

    table_df = pd.DataFrame({
        "樣本類別": ["純樣本外測試 (OOS)"] * len(date_labels),
        "目標交易日 (T)": date_labels,
        "國安基金": nsf_status_labels,
        "近月期開盤 (TXF Open)": [
            f"{op:,.1f}" if not np.isnan(op) else "-"
            for op in txf_open_prices
        ],
        "近月期收盤 (TXF Close)": [
            f"{cp:,.1f}" if not np.isnan(cp) else "-"
            for cp in txf_close_prices
        ],
        "對齊美股結算日": us_align_labels,
        "樣本外預測勝率": [f"{p:.1f}%" for p in probs_pct],
        "策略訊號": signal_types,
        "收盤留倉部位": pos_labels,
        "會計執行動作 (開盤平倉/留倉續抱)": validation_status,
        "單日波段毛損益": [
            f"{gr:+.2f}%" if not np.isnan(gr) else "尚未結算"
            for gr in gross_returns
        ],
        "當日摩擦成本": [
            f"-{c:.3f}%" if c > 0 else "0.000%" for c in friction_costs
        ],
        "單日實質淨損益": [
            f"{nr:+.2f}%" if not np.isnan(nr) else "尚未結算"
            for nr in net_returns
        ],
    })
    st.dataframe(table_df, use_container_width=True, hide_index=True)
else:
    st.warning("歷史交易日資料不足，無法繪製純樣本外測試比對圖。")