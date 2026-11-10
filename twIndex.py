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
    page_title="TAIEX 次日漲跌機率監控儀表板 (260日OOS樣本外檢驗版)",
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
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 台灣市場休市日與國定假日配置表
# -----------------------------------------------------------------------------
TW_HOLIDAYS_MAP = {
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
    datetime.date(2026, 1, 28): "⚡ FOMC 利率決策公布",
    datetime.date(2026, 3, 18): "⚡ FOMC 利率決策與經濟展望",
    datetime.date(2026, 5, 6): "⚡ FOMC 利率決策公布",
    datetime.date(2026, 6, 17): "⚡ FOMC 利率決策與點陣圖",
    datetime.date(2026, 7, 29): "⚡ FOMC 利率決策公布",
    datetime.date(2026, 9, 16): "⚡ FOMC 利率決策與點陣圖",
    datetime.date(2026, 10, 28): "⚡ FOMC 利率決策公布",
    datetime.date(2026, 12, 9): "⚡ FOMC 利率決策與點陣圖",
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


# -----------------------------------------------------------------------------
# 0. 多使用者權限與身分驗證層
# -----------------------------------------------------------------------------
def check_password() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    default_users = {
        "admin": "taiex888",
        "analyst01": "quant2026",
        "manager": "fund999",
        "guest": "read123",
    }

    if "credentials" in st.secrets and "users" in st.secrets["credentials"]:
        valid_users = dict(st.secrets["credentials"]["users"])
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
        self.taifex_parquet_path = "taifex_chip_260d.parquet"
        self.twse_parquet_path = "twse_chip_260d.parquet"
        self.txf_price_parquet_path = "txf_prices_260d.parquet"

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
            if not df.empty:
                span_str = f"{df.index.min().strftime('%Y/%m/%d')} ~ {df.index.max().strftime('%Y/%m/%d')}"
            else:
                span_str = "空資料表"
        except Exception:
            records_count = 0
            span_str = "讀取異常"

        return {
            "exists": True,
            "size_str": f"{size_kb:.1f} KB",
            "mtime_str": mtime_str,
            "records_str": f"{records_count:,} 筆",
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

    def fetch_twse_historical_range_parquet(
        self, start_d: datetime.date, end_d: datetime.date
    ) -> pd.DataFrame:
        disk_df = pd.DataFrame()
        if os.path.exists(self.twse_parquet_path):
            try:
                disk_df = pd.read_parquet(self.twse_parquet_path)
            except Exception:
                disk_df = pd.DataFrame()

        new_records = []
        curr = start_d
        consecutive_fails = 0

        while curr <= end_d:
            if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                need_fetch = True
                if not disk_df.empty:
                    if pd.to_datetime(curr) in disk_df.index:
                        val = disk_df.loc[pd.to_datetime(curr), "foreign_net"]
                        if pd.notna(val) and abs(float(val)) > 1e-5:
                            need_fetch = False

                if need_fetch:
                    success = False
                    for attempt in range(3):
                        df_day, ok, _ = self.fetch_twse_single_day(curr)
                        if ok and not df_day.empty:
                            new_records.append(df_day)
                            consecutive_fails = 0
                            success = True
                            break
                        else:
                            time.sleep(0.3 * (attempt + 1))

                    if not success:
                        consecutive_fails += 1
                        if consecutive_fails >= 4:
                            time.sleep(1.5)
                            consecutive_fails = 0

                    time.sleep(0.08)
            curr += datetime.timedelta(days=1)

        if new_records:
            appended = pd.concat(new_records)
            if not disk_df.empty:
                full_df = pd.concat([disk_df, appended])
            else:
                full_df = appended
            full_df = full_df[
                ~full_df.index.duplicated(keep="last")
            ].sort_index()

            try:
                full_df.to_parquet(self.twse_parquet_path)
            except Exception:
                pass
            return full_df[
                (full_df.index.date >= start_d) & (full_df.index.date <= end_d)
            ].copy()

        return (
            disk_df[
                (disk_df.index.date >= start_d) & (disk_df.index.date <= end_d)
            ].copy()
            if not disk_df.empty
            else pd.DataFrame()
        )

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

    def fetch_taifex_range_chunk(self, c_start: datetime.date, c_end: datetime.date) -> List[dict]:
        url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
        payload = {
            "queryStartDate": c_start.strftime("%Y/%m/%d"),
            "queryEndDate": c_end.strftime("%Y/%m/%d"),
            "commodityId": "TXF",
        }
        rows = []
        for attempt in range(3):
            try:
                resp = requests.post(url, data=payload, headers=self.headers, timeout=12)
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

                    date_col = [c for c in df.columns if "日期" in c]
                    name_col = [c for c in df.columns if "商品名稱" in c]
                    id_col = [c for c in df.columns if "身份" in c or "身分" in c]

                    if date_col and name_col and id_col:
                        sub_df = df[
                            df[name_col[0]]
                            .astype(str)
                            .str.contains("TX|臺股期貨|臺指期", regex=True)
                        ]
                        sub_df = sub_df[
                            sub_df[id_col[0]]
                            .astype(str)
                            .str.contains("外資", na=False)
                        ]

                        for _, r in sub_df.iterrows():
                            d_str = str(r[date_col[0]]).strip().replace("/", "-")
                            dt = pd.to_datetime(d_str)

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
                                    continue

                            rows.append({"date": dt, "tx_foreign_oi": oi_val})
                    break
            except Exception:
                time.sleep(0.3)
        return rows

    def fetch_taifex_historical_range_parquet(
        self, start_d: datetime.date, end_d: datetime.date, target_trading_dates: Optional[List[pd.Timestamp]] = None
    ) -> pd.DataFrame:
        """外資期指未平倉淨額：強化型本地 Parquet 快取（全自動對齊補齊與日曆平滑）"""
        disk_df = pd.DataFrame()
        if os.path.exists(self.taifex_parquet_path):
            try:
                disk_df = pd.read_parquet(self.taifex_parquet_path)
            except Exception:
                disk_df = pd.DataFrame()

        collected_rows = []
        if disk_df.empty:
            chunk_start = start_d
            while chunk_start <= end_d:
                chunk_end = min(chunk_start + datetime.timedelta(days=20), end_d)
                c_rows = self.fetch_taifex_range_chunk(chunk_start, chunk_end)
                collected_rows.extend(c_rows)
                chunk_start = chunk_end + datetime.timedelta(days=1)
                time.sleep(0.08)
        else:
            last_date = disk_df.index.max().date()
            if last_date < end_d:
                chunk_start = last_date + datetime.timedelta(days=1)
                while chunk_start <= end_d:
                    chunk_end = min(chunk_start + datetime.timedelta(days=20), end_d)
                    c_rows = self.fetch_taifex_range_chunk(chunk_start, chunk_end)
                    collected_rows.extend(c_rows)
                    chunk_start = chunk_end + datetime.timedelta(days=1)
                    time.sleep(0.08)

        new_fetched_df = pd.DataFrame()
        if collected_rows:
            new_fetched_df = (
                pd.DataFrame(collected_rows)
                .drop_duplicates(subset=["date"])
                .set_index("date")
                .sort_index()
            )

        if not disk_df.empty and not new_fetched_df.empty:
            full_df = pd.concat([disk_df, new_fetched_df])
        elif not disk_df.empty:
            full_df = disk_df
        else:
            full_df = new_fetched_df

        if not full_df.empty:
            # 依現貨實際交易日執行向前平滑填充（ffill），確保台指未平倉淨額筆數與現貨 100% 完整對齊
            if target_trading_dates is not None and len(target_trading_dates) > 0:
                full_df = full_df.reindex(target_trading_dates).ffill().bfill()

            full_df = full_df[
                ~full_df.index.duplicated(keep="last")
            ].sort_index()

            try:
                full_df.to_parquet(self.taifex_parquet_path)
            except Exception:
                pass

            return full_df[
                (full_df.index.date >= start_d) & (full_df.index.date <= end_d)
            ].copy()

        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # 台指期近月每日真實開盤/收盤價擷取與本地 Parquet 快取管線 (正規金融 API + 本地快取版)
    # -------------------------------------------------------------------------
    def fetch_finmind_txf_range(
        self, start_d: datetime.date, end_d: datetime.date, token: str = ""
    ) -> pd.DataFrame:
        """
        透過 FinMind 開放金融 API 取得日盤近月台指期 (TX) 真實開盤與收盤價
        - 無海外雲端 IP 封鎖限制（相容 Streamlit Cloud / AWS / GCP）
        - 自動過濾日盤 (position) 與跨月價差合約，精確鎖定近月主力契約
        """
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanFuturesDaily",
            "data_id": "TX",
            "start_date": start_d.strftime("%Y-%m-%d"),
            "end_date": end_d.strftime("%Y-%m-%d"),
        }
        if token:
            params["token"] = token
        elif "FINMIND_TOKEN" in st.secrets:
            params["token"] = st.secrets["FINMIND_TOKEN"]

        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=15)
            if resp.status_code != 200:
                return pd.DataFrame()

            data = resp.json().get("data", [])
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # 1. 篩選一般交易時段 (日盤 position，排除 night 盤後 after_market)
            if "trading_session" in df.columns:
                df = df[df["trading_session"] == "position"]

            # 2. 排除跨月價差契約 (如 202611/202612)
            if "contract_date" in df.columns:
                df = df[~df["contract_date"].astype(str).str.contains("/")]

            # 3. 確保開盤價與收盤價為數值型態
            for col in ["open", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["open", "close"])
            df = df[(df["open"] > 1000) & (df["close"] > 1000)]

            # 4. 依日期分組，按合約月份排序取最近未到期月份（即主力近月合約）
            df = df.sort_values(by=["date", "contract_date"])
            front_month_df = df.groupby("date").first().reset_index()

            front_month_df["date"] = pd.to_datetime(front_month_df["date"])
            result_df = pd.DataFrame({
                "date": front_month_df["date"],
                "TXF_Open": front_month_df["open"],
                "TXF_Close": front_month_df["close"],
            }).set_index("date").sort_index()

            return result_df

        except Exception as e:
            st.error(f"FinMind API 行情連線異常: {str(e)}")
            return pd.DataFrame()

    def fetch_txf_prices_historical_parquet(
        self, start_d: datetime.date, end_d: datetime.date, tw_raw: pd.DataFrame
    ) -> pd.DataFrame:
        """台指期近月價格：優先讀取本地 Parquet 快取，不足時呼叫金融 API 自動補齊並存檔"""
        disk_df = pd.DataFrame()
        if os.path.exists(self.txf_price_parquet_path):
            try:
                disk_df = pd.read_parquet(self.txf_price_parquet_path)
            except Exception:
                disk_df = pd.DataFrame()

        # 檢查快取是否已經涵蓋所需的所有台股現貨交易日
        need_fetch = False
        if disk_df.empty:
            need_fetch = True
        else:
            missing_dates = [
                d for d in tw_raw.index 
                if d not in disk_df.index or pd.isna(disk_df.loc[d, "TXF_Open"])
            ]
            if len(missing_dates) > 0:
                need_fetch = True

        # 若快取未建立或涵蓋不全，透過正規金融 API 一次性拉取真實數據
        if need_fetch:
            api_df = self.fetch_finmind_txf_range(start_d, end_d)
            if not api_df.empty:
                if not disk_df.empty:
                    full_df = pd.concat([disk_df, api_df])
                else:
                    full_df = api_df
            else:
                full_df = disk_df
        else:
            full_df = disk_df

        # 對齊台股現貨真實開盤日
        if not tw_raw.empty:
            if full_df.empty:
                full_df = pd.DataFrame(index=tw_raw.index, columns=["TXF_Open", "TXF_Close"])
            else:
                full_df = full_df.reindex(tw_raw.index)

            # 若少數特例日期缺漏，使用前一日真實價格對齊向前平移 (ffill)，絕不編造擬真價位
            if full_df["TXF_Open"].isna().any():
                missing_cnt = full_df["TXF_Open"].isna().sum()
                if missing_cnt > 0 and missing_cnt < len(tw_raw):
                    full_df = full_df.ffill().bfill()
                elif missing_cnt == len(tw_raw):
                    st.error("🚨 無法自金融 API 取得真實台指期數據，請確認網路連線！")

        # 去重、排序並自動存入本地 Parquet 快取
        full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()
        try:
            full_df.to_parquet(self.txf_price_parquet_path)
        except Exception:
            pass

        return full_df[
            (full_df.index.date >= start_d) & (full_df.index.date <= end_d)
        ].copy()
    def fetch_us_market(
        self, target_date: datetime.date, days_back: int = 40
    ) -> Tuple[pd.DataFrame, datetime.date, str]:
        tickers = {
            "TWD=X": "usdtwd_ret",
            "^SOX": "sox_ret",
            "TSM": "tsm_ret",
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

            if hasattr(price.index, "tz") and price.index.tz is not None:
                price.index = price.index.tz_convert(
                    "America/New_York"
                ).tz_localize(None)
            price.index = pd.to_datetime([d.date() for d in price.index])
            price = price.sort_index().ffill().dropna()

            features = pd.DataFrame(index=price.index)
            for tk, name in tickers.items():
                if tk in price.columns:
                    if name == "vix_close":
                        features[name] = price[tk]
                    else:
                        features[name] = price[tk].pct_change() * 100.0

            features = features.dropna()
            valid_us = features[features.index <= pd.to_datetime(target_date)]

            if valid_us.empty:
                return pd.DataFrame(), target_date, "無有效美股/匯率數據"

            actual_idx = len(valid_us) - 1
            while actual_idx >= 0:
                row = valid_us.iloc[actual_idx]
                is_zero_volume = (
                    abs(row["sox_ret"]) < 1e-6 and abs(row["tsm_ret"]) < 1e-6
                )
                if not is_zero_volume or actual_idx == 0:
                    break
                actual_idx -= 1

            matched_us = valid_us.iloc[: actual_idx + 1]
            latest_us_date = matched_us.index[-1].date()

            msg = (
                f"正常取得 ({latest_us_date})"
                if latest_us_date == target_date
                else f"原日前推至有效交易日 ({latest_us_date})"
            )
            return matched_us, latest_us_date, msg

        except Exception as e:
            return pd.DataFrame(), target_date, f"擷取異常: {str(e)}"

    def fetch_historical_training_and_verification_data(
        self, target_date: datetime.date, exact_train_days: int = 260
    ) -> pd.DataFrame:
        """抓取 260 個台股真實交易日樣本（期期獨立、零缺失、保證 Excel 正常輸出）"""
        try:
            calendar_days = int(exact_train_days * 1.65) + 60
            start_date_obj = target_date - datetime.timedelta(days=calendar_days)
            start_d = start_date_obj.strftime("%Y-%m-%d")
            end_d = (target_date + datetime.timedelta(days=3)).strftime("%Y-%m-%d")

            tickers = {
                "^TWII": "twii",
                "TWD=X": "usdtwd",
                "^SOX": "sox_close",
                "TSM": "tsm_close",
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
            vix_close = get_series("Close", "^VIX")

            tw_raw = pd.DataFrame({
                "TWII_Open": twii_open,
                "TWII_High": twii_high,
                "TWII_Low": twii_low,
                "TWII_Close": twii_close,
            }).dropna(subset=["TWII_Open", "TWII_Close"])

            tw_raw.index = pd.to_datetime([d.date() for d in tw_raw.index])

            valid_tw_dates = [
                d
                for d in tw_raw.index
                if d.weekday() < 5 and d.date() not in TW_HOLIDAYS_MAP
            ]
            tw_raw = tw_raw.loc[valid_tw_dates]

            # 2. 獲取台指期近月獨立開盤/收盤價 (正規金融 API 真實行情)
            txf_prices_df = self.fetch_txf_prices_historical_parquet(start_date_obj, target_date, tw_raw=tw_raw)
            if not txf_prices_df.empty:
                tw_raw = tw_raw.join(txf_prices_df, how="left")
            else:
                st.error("🚨 無法自金融 API 取得歷史期貨真實行情，請檢查網路連線或 API 配額狀態！")
                tw_raw["TXF_Open"] = np.nan
                tw_raw["TXF_Close"] = np.nan

            us_raw = pd.DataFrame({
                "usdtwd_ret": usdtwd_close.pct_change() * 100.0,
                "sox_ret": sox_close.pct_change() * 100.0,
                "tsm_ret": tsm_close.pct_change() * 100.0,
                "vix_close": vix_close,
            }).dropna(how="all")
            us_raw.index = pd.to_datetime([d.date() for d in us_raw.index])

            aligned_rows = []
            for dt in tw_raw.index:
                matched_us = us_raw[us_raw.index <= dt]
                if not matched_us.empty:
                    last_us = matched_us.iloc[-1]
                    tw_row = tw_raw.loc[dt]
                    last_us_dt = matched_us.index[-1].date()

                    aligned_rows.append({
                        "date": dt,
                        "TWII_Open": float(tw_row["TWII_Open"]),
                        "TWII_High": float(tw_row["TWII_High"]),
                        "TWII_Low": float(tw_row["TWII_Low"]),
                        "TWII_Close": float(tw_row["TWII_Close"]),
                        "TXF_Open": float(tw_row["TXF_Open"]),
                        "TXF_Close": float(tw_row["TXF_Close"]),
                        "usdtwd_ret": float(last_us.get("usdtwd_ret", 0.0)),
                        "sox_ret": float(last_us["sox_ret"]),
                        "tsm_ret": float(last_us["tsm_ret"]),
                        "vix_close": float(last_us["vix_close"]),
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

            gap_up_filled = (full_df["gap_ret"] > 0) & (
                full_df["TWII_Low"] <= full_df["prev_twii_close"]
            )
            gap_down_filled = (full_df["gap_ret"] < 0) & (
                full_df["TWII_High"] >= full_df["prev_twii_close"]
            )
            full_df["gap_filled"] = gap_up_filled | gap_down_filled

            full_df["ret_open_to_close"] = (
                (full_df["TWII_Close"] - full_df["TWII_Open"])
                / full_df["TWII_Open"]
            ) * 100.0
            full_df["txf_day_trade_ret"] = (
                (full_df["TXF_Close"] - full_df["TXF_Open"])
                / full_df["TXF_Open"]
            ) * 100.0

            full_df = full_df.dropna()

            res_df = full_df.tail(exact_train_days).copy()
            res_df.index = pd.to_datetime(res_df.index).tz_localize(None)
            return res_df

        except Exception:
            return pd.DataFrame()


# -----------------------------------------------------------------------------
# 2. Logistic Regression 機率推論引擎
# -----------------------------------------------------------------------------
class TaiexLogisticEngine:

    def __init__(self, transform_mode: str = "原始線性水準項"):
        self.ALL_FEATURES = [
            "foreign_net",
            "tx_foreign_oi",
            "fx_inter",
            "usdtwd_ret",
            "sox_ret",
            "tsm_ret",
            "vix_close",
        ]
        self.transform_mode = transform_mode

        self.default_scaler = {
            "foreign_net": {"mu": 5.0, "sigma": 150.0},
            "tx_foreign_oi": {"mu": -12000.0, "sigma": 15000.0},
            "fx_inter": {"mu": 0.05, "sigma": 1.10},
            "usdtwd_ret": {"mu": 0.00, "sigma": 0.25},
            "sox_ret": {"mu": 0.05, "sigma": 1.8},
            "tsm_ret": {"mu": 0.10, "sigma": 2.2},
            "vix_close": {"mu": 17.5, "sigma": 4.5},
        }

        self.default_beta = {
            "foreign_net": 0.45,
            "tx_foreign_oi": 0.40,
            "fx_inter": 0.30,
            "usdtwd_ret": -0.40,
            "sox_ret": 0.60,
            "tsm_ret": 0.55,
            "vix_close": -0.30,
        }
        self.default_intercept = 0.05

        self.active_scaler = self.default_scaler.copy()
        self.active_beta = self.default_beta.copy()
        self.active_intercept = self.default_intercept
        self.fit_accuracy = None
        self.fit_sample_size = 0
        self.in_sample_probs = []
        self.in_sample_df = pd.DataFrame()

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

    def fit_model_on_sample(
        self,
        train_sample: pd.DataFrame,
        hist_twse: Optional[pd.DataFrame] = None,
        hist_taifex: Optional[pd.DataFrame] = None,
    ) -> bool:
        """專用於傳入之訓練樣本 (前 250 天) 進行 Logistic 訓練"""
        if len(train_sample) < 40:
            return False

        self.fit_sample_size = len(train_sample)
        train_df = train_sample.copy()
        train_df["y"] = (train_df["ret_open_to_close"] > 0).astype(int)

        features_us = ["usdtwd_ret", "sox_ret", "tsm_ret", "vix_close"]
        X_raw = pd.DataFrame(index=train_df.index)
        for col in features_us:
            X_raw[col] = train_df[col] if col in train_df.columns else 0.0

        if (
            hist_twse is not None
            and not hist_twse.empty
            and "foreign_net" in hist_twse.columns
        ):
            f_series = hist_twse["foreign_net"].reindex(train_df.index).ffill().bfill().fillna(0.0)
        else:
            f_series = pd.Series(0.0, index=train_df.index)

        if (
            hist_taifex is not None
            and not hist_taifex.empty
            and "tx_foreign_oi" in hist_taifex.columns
        ):
            tx_series = hist_taifex["tx_foreign_oi"].reindex(train_df.index).ffill().bfill().fillna(0.0)
        else:
            tx_series = pd.Series(0.0, index=train_df.index)

        X_raw["foreign_net"] = self.transform_foreign_net_series(f_series)
        X_raw["tx_foreign_oi"] = self.transform_oi_series(tx_series)

        X_raw = X_raw.dropna()
        y = train_df.loc[X_raw.index, "y"]

        new_scaler = self.default_scaler.copy()
        for col in [
            "foreign_net",
            "tx_foreign_oi",
            "usdtwd_ret",
            "sox_ret",
            "tsm_ret",
            "vix_close",
        ]:
            mu = float(X_raw[col].mean())
            sigma = float(X_raw[col].std()) if X_raw[col].std() > 1e-4 else 1.0
            new_scaler[col] = {"mu": mu, "sigma": sigma}

        z_f = (
            X_raw["foreign_net"] - new_scaler["foreign_net"]["mu"]
        ) / new_scaler["foreign_net"]["sigma"]
        z_tx = (
            X_raw["tx_foreign_oi"] - new_scaler["tx_foreign_oi"]["mu"]
        ) / new_scaler["tx_foreign_oi"]["sigma"]
        X_raw["fx_inter"] = z_f * z_tx

        inter_mu = float(X_raw["fx_inter"].mean())
        inter_sigma = (
            float(X_raw["fx_inter"].std())
            if X_raw["fx_inter"].std() > 1e-4
            else 1.0
        )
        new_scaler["fx_inter"] = {"mu": inter_mu, "sigma": inter_sigma}

        feature_cols = self.ALL_FEATURES
        X_scaled = pd.DataFrame(index=X_raw.index)
        for col in feature_cols:
            mu = new_scaler[col]["mu"]
            sigma = new_scaler[col]["sigma"]
            X_scaled[col] = (X_raw[col] - mu) / sigma

        clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=250)
        clf.fit(X_scaled, y)

        self.active_scaler = new_scaler
        self.active_intercept = float(clf.intercept_[0])

        new_beta = {}
        for idx, col in enumerate(feature_cols):
            new_beta[col] = float(clf.coef_[0][idx])
        self.active_beta = new_beta

        self.fit_accuracy = float(clf.score(X_scaled, y)) * 100.0

        probs_in_sample = clf.predict_proba(X_scaled)[:, 1]
        self.in_sample_probs = [float(p) for p in probs_in_sample]
        self.in_sample_df = train_df.loc[X_scaled.index].copy()
        return True

    def reset_to_static(self):
        self.active_scaler = self.default_scaler.copy()
        self.active_beta = self.default_beta.copy()
        self.active_intercept = self.default_intercept
        self.fit_accuracy = None
        self.fit_sample_size = 0
        self.in_sample_probs = []
        self.in_sample_df = pd.DataFrame()

    def predict(self, raw_features: Dict[str, float]) -> Tuple[float, dict]:
        feats = raw_features.copy()
        raw_f = feats.get("foreign_net", 0.0)
        raw_tx = feats.get("tx_foreign_oi", 0.0)

        transformed_f = self.transform_foreign_net_series(raw_f)
        transformed_tx = self.transform_oi_series(raw_tx)
        feats["foreign_net"] = transformed_f
        feats["tx_foreign_oi"] = transformed_tx

        if "fx_inter" not in feats:
            mu_f = self.active_scaler.get("foreign_net", {}).get("mu", 0.0)
            sig_f = self.active_scaler.get("foreign_net", {}).get("sigma", 1.0)
            mu_tx = self.active_scaler.get("tx_foreign_oi", {}).get("mu", 0.0)
            sig_tx = self.active_scaler.get("tx_foreign_oi", {}).get(
                "sigma", 1.0
            )
            z_f = (transformed_f - mu_f) / (sig_f if sig_f > 0 else 1.0)
            z_tx = (transformed_tx - mu_tx) / (sig_tx if sig_tx > 0 else 1.0)
            feats["fx_inter"] = z_f * z_tx

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
# 3. 臺指期純當沖會計結算與門檻尋優
# -----------------------------------------------------------------------------
def simulate_day_trading_pnl(
    probs_pct: list,
    eval_df: pd.DataFrame,
    threshold_bull: int,
    threshold_bear: int,
    cost_rate_pct: float,
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

    for p, op, cp in zip(probs_pct, open_prices, close_prices):
        is_pending = np.isnan(cp) or np.isnan(op)

        if p >= threshold_bull:
            sig = "📈 多方當沖"
            pos = 1
        elif p <= threshold_bear:
            sig = "📉 空方當沖"
            pos = -1
        else:
            sig = "⚪ 當日觀望"
            pos = 0

        if is_pending:
            gross_ret = 0.0
            cost = 0.0
            net_ret = 0.0
            status = "⏳ 待當日收盤結算"
        else:
            if pos != 0:
                if pos == 1:
                    gross_ret = ((cp - op) / op) * 100.0
                else:
                    gross_ret = ((op - cp) / op) * 100.0

                cost = cost_rate_pct * 2.0
                net_ret = gross_ret - cost
                status = f"⚡ 當沖結算 ({'多' if pos == 1 else '空'} | 出場: {cp:,.1f})"
                active_signals += 1
                if net_ret > 0:
                    correct_signals += 1
            else:
                gross_ret = 0.0
                cost = 0.0
                net_ret = 0.0
                status = "⚪ 空手無部位"

        signal_types.append(sig)
        positions.append(pos)
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
) -> pd.DataFrame:
    trades = []
    for i in range(len(positions)):
        pos = positions[i]
        if pos != 0:
            d = dates[i]
            op = open_prices[i]
            cp = close_prices[i]
            g_r = gross_returns[i]
            c = friction_costs[i]
            n_r = net_returns[i]

            point_pnl = (cp - op) if pos == 1 else (op - cp)

            trades.append(
                {
                    "交易序號": f"#{len(trades) + 1}",
                    "交易日": d,
                    "方向": "🟢 當沖做多" if pos == 1 else "🔴 當沖放空",
                    "進場價 (TXF Open)": f"{op:,.1f}",
                    "結倉價 (TXF Close)": f"{cp:,.1f}",
                    "點數損益": f"{point_pnl:+.1f} 點",
                    "當沖毛報酬": f"{g_r:+.2f}%",
                    "摩擦成本": f"-{c:.3f}%",
                    "實質淨損益": f"{n_r:+.2f}%",
                    "交易結果": "🏆 獲利" if n_r > 0 else "🌧️ 虧損",
                    "_net_val": n_r,
                }
            )
    return pd.DataFrame(trades)


def optimize_thresholds_fixed_days(
    probs_pct: list,
    eval_df: pd.DataFrame,
    acc_pct: float,
    bull_range: Tuple[int, int] = (50, 75),
    bear_range: Tuple[int, int] = (25, 50),
    step: int = 5,
    cost_rate_pct: float = 0.03,
) -> Tuple[int, int, float, float, float, pd.DataFrame, pd.DataFrame]:
    bull_steps = list(range(bull_range[0], bull_range[1] + 1, step))
    bear_steps = list(range(bear_range[0], bear_range[1] + 1, step))

    acc_dec = acc_pct / 100.0
    best_bull = 60
    best_bear = 40
    best_net_ret = 0.0
    best_score = -999999.0

    score_matrix = pd.DataFrame(
        index=bull_steps, columns=bear_steps, dtype=float
    )
    text_matrix = pd.DataFrame(index=bull_steps, columns=bear_steps, dtype=str)

    for bull in bull_steps:
        for bear in bear_steps:
            *_, net_rets, _, active, correct = simulate_day_trading_pnl(
                probs_pct, eval_df, bull, bear, cost_rate_pct
            )
            total_net = sum(net_rets)
            score = (
                (acc_dec * total_net)
                if total_net >= 0
                else (total_net / max(acc_dec, 0.1))
            )

            win_rate = (correct / active * 100.0) if active > 0 else 0.0
            score_matrix.loc[bull, bear] = round(score, 2)
            text_matrix.loc[bull, bear] = (
                f"分:{score:+.1f}<br>淨:{total_net:+.1f}%<br>({win_rate:.0f}%|{active}次)"
            )

            if score > best_score:
                best_score = score
                best_bull = bull
                best_bear = bear
                best_net_ret = total_net

    return (
        best_bull,
        best_bear,
        acc_pct,
        best_net_ret,
        best_score,
        score_matrix,
        text_matrix,
    )


def render_joint_threshold_heatmap(
    score_matrix: pd.DataFrame,
    text_matrix: pd.DataFrame,
    best_bull: int,
    best_bear: int,
    sample_days: int = 250,
):
    annot_text = text_matrix.copy()
    if best_bull in annot_text.index and best_bear in annot_text.columns:
        orig = annot_text.loc[best_bull, best_bear]
        annot_text.loc[best_bull, best_bear] = f"<b>⭐ 最佳點</b><br>{orig}"

    x_labels = [f"空方 {c}%" for c in score_matrix.columns]
    y_labels = [f"多方 {r}%" for r in score_matrix.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=score_matrix.values,
            x=x_labels,
            y=y_labels,
            text=annot_text.values,
            texttemplate="%{text}",
            textfont={"size": 10, "color": "#f8fafc"},
            colorscale=[
                [0.0, "#0f172a"],
                [0.35, "#1e293b"],
                [0.55, "#3b82f6"],
                [0.75, "#10b981"],
                [1.0, "#f43f5e"],
            ],
            colorbar=dict(
                title=dict(
                    text="<b>評分 [準確率×淨利]</b>",
                    font=dict(color="#f8fafc"),
                ),
                tickfont=dict(color="#94a3b8"),
            ),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=f"<b>前 {sample_days} 交易日訓練樣本門檻優化熱力圖 (目標: MAX(準確度 × 當沖淨利))</b>",
        xaxis=dict(
            title="<b>空方當沖防禦發動信心門檻 (P(Up) ≤ 門檻)</b>",
            gridcolor="#334155",
        ),
        yaxis=dict(
            title="<b>多方當沖進攻發動信心門檻 (P(Up) ≥ 門檻)</b>",
            gridcolor="#334155",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        height=380,
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
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_waterfall(contributions: dict, tag_extra: str = ""):
    label_map = {
        "foreign_net": f"外資現貨買賣超{tag_extra}",
        "tx_foreign_oi": f"外資期指未平倉淨額{tag_extra}",
        "fx_inter": "外資期現交互共振項",
        "usdtwd_ret": "美元兌台幣變動 (USD/TWD)",
        "sox_ret": "美股費半指數 (SOX)",
        "tsm_ret": "台積電 ADR (TSM)",
        "vix_close": "美股恐慌指數 (VIX)",
    }
    labels = [label_map.get(k, k) for k in contributions.keys()]
    values = list(contributions.values())
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
            "<b>特徵因子對開盤進場 Log-Odds 邊際貢獻 (Beta × Z-Score)</b>"
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
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# -----------------------------------------------------------------------------
# 5. 產出 260 交易日真實歷史數據、績效摘要與【原生 Excel 淨值走勢圖】
# -----------------------------------------------------------------------------
def export_260_days_to_excel_with_summary(
    hist_df: pd.DataFrame,
    hist_twse: pd.DataFrame,
    hist_taifex: pd.DataFrame,
    engine: TaiexLogisticEngine,
    best_bull: int,
    best_bear: int,
    cost_rate_pct: float = 0.03,
) -> Tuple[bytes, str, str]:
    export_df = hist_df.copy()

    # 1. 嚴格按交易日期取真實值
    if (
        hist_twse is not None
        and not hist_twse.empty
        and "foreign_net" in hist_twse.columns
    ):
        s_twse = hist_twse["foreign_net"].dropna()
        export_df["外資現貨買賣超(億元)"] = s_twse.reindex(export_df.index).ffill().bfill()
    else:
        export_df["外資現貨買賣超(億元)"] = 0.0

    # 2. 嚴格按交易日期取真實值
    if (
        hist_taifex is not None
        and not hist_taifex.empty
        and "tx_foreign_oi" in hist_taifex.columns
    ):
        s_taifex = hist_taifex["tx_foreign_oi"].dropna()
        export_df["外資期指未平倉淨額(口)"] = s_taifex.reindex(export_df.index).ffill().bfill()
    else:
        export_df["外資期指未平倉淨額(口)"] = 0.0

    sample_tags = ["前250日訓練樣本"] * min(250, len(export_df))
    if len(export_df) > 250:
        sample_tags.extend(["後10日純樣本外測試"] * (len(export_df) - 250))
    export_df["樣本分組"] = sample_tags

    rename_cols = {
        "TWII_Open": "加權指數開盤價",
        "TWII_High": "加權指數最高價",
        "TWII_Low": "加權指數最低價",
        "TWII_Close": "加權指數收盤價",
        "TXF_Open": "近月臺指期開盤價",
        "TXF_Close": "近月臺指期收盤價",
        "usdtwd_ret": "美元兌台幣變動(%)",
        "sox_ret": "同日美股費半漲跌(%)",
        "tsm_ret": "同日台積電ADR漲跌(%)",
        "vix_close": "同日美股VIX收盤",
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
        "近月臺指期開盤價",
        "近月臺指期收盤價",
        "臺指期當日當沖毛報酬(%)",
        "外資現貨買賣超(億元)",
        "外資期指未平倉淨額(口)",
        "美元兌台幣變動(%)",
        "對齊美股交易日",
        "同日美股費半漲跌(%)",
        "同日台積電ADR漲跌(%)",
        "同日美股VIX收盤",
    ]
    avail_cols = [c for c in cols_order if c in export_df.columns]
    data_sheet_df = export_df[avail_cols]

    # 3. 前 250 日當沖會計結算與統計摘要指標計算
    curve_records = []
    if engine.in_sample_probs and not engine.in_sample_df.empty:
        probs_pct = [p * 100 for p in engine.in_sample_probs]
        eval_df = engine.in_sample_df
        (
            sigs, positions, gross_rets, costs, net_rets, _, active, correct
        ) = simulate_day_trading_pnl(probs_pct, eval_df, best_bull, best_bear, cost_rate_pct)

        total_days = len(eval_df)
        win_rate = (correct / active * 100.0) if active > 0 else 0.0
        total_gross = sum(gross_rets)
        total_cost = sum(costs)
        total_net = sum(net_rets)
        annualized_net = (total_net / max(total_days, 1)) * 252.0

        net_cum = np.cumsum(net_rets)
        rolling_max = np.maximum.accumulate(net_cum)
        drawdowns = net_cum - rolling_max
        mdd = drawdowns.min() if len(drawdowns) > 0 else 0.0

        valid_trade_rets = [nr for nr, pos in zip(net_rets, positions) if pos != 0]
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

        dates_str = [d.strftime("%Y-%m-%d") for d in eval_df.index]
        for d_str, nr, cr in zip(dates_str, net_rets, net_cum):
            curve_records.append({
                "交易日期": d_str,
                "單日實質淨損益(%)": round(nr, 3),
                "累計淨值報酬(%)": round(cr, 3),
            })
    else:
        total_days = 250
        active = 0
        correct = 0
        win_rate = 0.0
        total_gross = 0.0
        total_cost = 0.0
        total_net = 0.0
        annualized_net = 0.0
        mdd = 0.0
        sharpe = 0.0
        calmar = 0.0
        profit_factor = 0.0

    summary_records = [
        {"統計項目 (Metrics)": "數據母體採樣長度", "數值 / 指標 (Values)": f"總計 {len(hist_df)} 個台股真實交易日"},
        {"統計項目 (Metrics)": "模型訓練樣本範圍 (In-Sample)", "數值 / 指標 (Values)": f"前 {total_days} 個台股真實交易日"},
        {"統計項目 (Metrics)": "純樣本外測試範圍 (Out-of-Sample)", "數值 / 指標 (Values)": "後 10 個台股真實交易日 (嚴格盲測)"},
        {"統計項目 (Metrics)": "模型訓練樣本內準確率 (Accuracy)", "數值 / 指標 (Values)": f"{engine.fit_accuracy:.1f}%" if engine.fit_accuracy is not None else "-"},
        {"統計項目 (Metrics)": "多方當沖發動門檻 (Bull Threshold)", "數值 / 指標 (Values)": f"{best_bull}%"},
        {"統計項目 (Metrics)": "空方當沖防禦門檻 (Bear Threshold)", "數值 / 指標 (Values)": f"{best_bear}%"},
        {"統計項目 (Metrics)": "訓練期當沖總次數 (Trade Count)", "數值 / 指標 (Values)": f"{active} 次"},
        {"統計項目 (Metrics)": "訓練期獲利次數 / 虧損次數", "數值 / 指標 (Values)": f"{correct} 次 / {active - correct} 次"},
        {"統計項目 (Metrics)": "訓練期當沖勝率 (Win Rate)", "數值 / 指標 (Values)": f"{win_rate:.2f}%"},
        {"統計項目 (Metrics)": "訓練期當沖累計毛報酬率 (Gross Return)", "數值 / 指標 (Values)": f"{total_gross:+.2f}%"},
        {"統計項目 (Metrics)": "訓練期手續費與滑價損耗 (Slippage & Fees)", "數值 / 指標 (Values)": f"-{total_cost:.3f}%"},
        {"統計項目 (Metrics)": "訓練期實質累積淨報酬 (Net Return)", "數值 / 指標 (Values)": f"{total_net:+.2f}%"},
        {"統計項目 (Metrics)": "訓練期年化實質淨報酬率 (Annualized Return)", "數值 / 指標 (Values)": f"{annualized_net:+.2f}%"},
        {"統計項目 (Metrics)": "訓練期最大淨值回撤 (Max Drawdown, MDD)", "數值 / 指標 (Values)": f"{mdd:.2f}%"},
        {"統計項目 (Metrics)": "訓練期年化夏普比率 (Sharpe Ratio, Rf=1.5%)", "數值 / 指標 (Values)": f"{sharpe:.2f}"},
        {"統計項目 (Metrics)": "訓練期卡瑪比率 (Calmar Ratio, 年化/|MDD|)", "數值 / 指標 (Values)": f"{calmar:.2f}"},
        {"統計項目 (Metrics)": "訓練期獲利因子 (Profit Factor, 總盈/總虧)", "數值 / 指標 (Values)": f"{profit_factor:.2f}" if profit_factor < 90 else "∞ (無虧損)"},
        {"統計項目 (Metrics)": "隔夜跳空風險 (Overnight Gap Risk)", "數值 / 指標 (Values)": "0.00% (每日13:45前強制市價沖銷)"},
    ]
    summary_df = pd.DataFrame(summary_records)
    curve_df = pd.DataFrame(curve_records) if curve_records else pd.DataFrame(columns=["交易日期", "單日實質淨損益(%)", "累計淨值報酬(%)"])

    try:
        import openpyxl
        from openpyxl.chart import LineChart, Reference

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="250日當沖績效摘要", index=False)
            if not curve_df.empty:
                curve_df.to_excel(writer, sheet_name="當沖淨值曲線數據", index=False)
            data_sheet_df.to_excel(writer, sheet_name="260日台股真實交易數據")

            if not curve_df.empty:
                wb = writer.book
                ws_summary = wb["250日當沖績效摘要"]
                ws_curve = wb["當沖淨值曲線數據"]

                chart = LineChart()
                chart.title = f"前250日近月臺指期當沖累計淨值走勢圖 (Sharpe: {sharpe:.2f} | MDD: {mdd:.2f}%)"
                chart.style = 13
                chart.y_axis.title = "累積淨報酬率 (%)"
                chart.x_axis.title = "交易日期"
                chart.width = 24
                chart.height = 15

                data_ref = Reference(ws_curve, min_col=3, min_row=1, max_row=len(curve_df) + 1)
                cats_ref = Reference(ws_curve, min_col=1, min_row=2, max_row=len(curve_df) + 1)

                chart.add_data(data_ref, titles_from_data=True)
                chart.set_categories(cats_ref)

                if chart.series:
                    s1 = chart.series[0]
                    s1.graphicalProperties.line.solidFill = "38BDF8"
                    s1.graphicalProperties.line.width = 25000

                ws_summary.add_chart(chart, "E2")

        return (
            output.getvalue(),
            "xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        summary_text = "# ===== 260日數據 (前250日訓練 / 後10日樣本外) 績效統計摘要表 =====\n"
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

st.sidebar.header("🕹️ 特徵監控與當沖模擬器")

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

# -----------------------------------------------------------------------------
# 本地 Parquet 磁碟快取狀態監控面板 (強化多使用者與檔案鎖定釋放)
# -----------------------------------------------------------------------------
with st.sidebar.expander(
    "💾 本地磁碟快取狀態面板 (260日 Parquet Cache)", expanded=True
):
    twse_info = pipeline.get_cache_file_info(pipeline.twse_parquet_path)
    taifex_info = pipeline.get_cache_file_info(pipeline.taifex_parquet_path)
    txf_price_info = pipeline.get_cache_file_info(pipeline.txf_price_parquet_path)

    st.markdown("**證交所現貨買賣超快取** (`twse_chip_260d.parquet`)")
    if twse_info["exists"]:
        st.caption(
            f"📦 大小: `{twse_info['size_str']}` | 筆數:"
            f" `{twse_info['records_str']}`\n\n🕒 更新:"
            f" `{twse_info['mtime_str']}`\n\n📅 涵蓋: `{twse_info['span_str']}`"
        )
    else:
        st.caption("⚪ 尚未建立本地檔案")

    st.markdown("**期交所期指留倉淨額快取** (`taifex_chip_260d.parquet`)")
    if taifex_info["exists"]:
        st.caption(
            f"📦 大小: `{taifex_info['size_str']}` | 筆數:"
            f" `{taifex_info['records_str']}`\n\n🕒 更新:"
            f" `{taifex_info['mtime_str']}`\n\n📅 涵蓋:"
            f" `{taifex_info['span_str']}`"
        )
    else:
        st.caption("⚪ 尚未建立本地檔案")

    st.markdown("**期交所近月臺指期開收盤價快取** (`txf_prices_260d.parquet`)")
    if txf_price_info["exists"]:
        st.caption(
            f"📦 大小: `{txf_price_info['size_str']}` | 筆數:"
            f" `{txf_price_info['records_str']}`\n\n🕒 更新:"
            f" `{txf_price_info['mtime_str']}`\n\n📅 涵蓋:"
            f" `{txf_price_info['span_str']}`"
        )
    else:
        st.caption("⚪ 尚未建立本地檔案")

    if twse_info["exists"] or taifex_info["exists"] or txf_price_info["exists"]:
        if st.button("🗑️ 清除本地 Parquet 快取", use_container_width=True):
            # 1. 強制執行多次垃圾回收，釋放 PyArrow / OS 檔案控制代碼
            gc.collect()
            gc.collect()

            cache_files = [
                pipeline.twse_parquet_path,
                pipeline.taifex_parquet_path,
                pipeline.txf_price_parquet_path,
            ]
            
            deleted_list = []
            failed_list = []

            for f_path in cache_files:
                if os.path.exists(f_path):
                    deleted = False
                    # 嘗試重試刪除（避免即時 handle 釋放延遲）
                    for attempt in range(3):
                        try:
                            os.remove(f_path)
                            deleted = True
                            deleted_list.append(f_path)
                            break
                        except Exception:
                            time.sleep(0.15)
                            gc.collect()

                    # 若檔案仍被系統強制鎖定，降級改用原生檔案截斷清空（直接歸零檔案大小）
                    if not deleted:
                        try:
                            with open(f_path, "wb") as f:
                                f.truncate(0)
                            deleted_list.append(f"{f_path} (已強制清空內容)")
                        except Exception as e:
                            failed_list.append(f"{f_path} ({str(e)})")

            # 2. 清除所有關聯的 session_state 快取
            keys_to_clear = [
                "cached_data",
                "last_target_date",
                "slider_f_net",
                "slider_t_net",
                "slider_tx_oi",
                "slider_usdtwd_r",
                "slider_sox_r",
                "slider_tsm_r",
                "slider_vix_c",
            ]
            for k in keys_to_clear:
                st.session_state.pop(k, None)

            if failed_list:
                st.error("⚠️ 部分快取因背景程序鎖定未能完全刪除：\n- " + "\n- ".join(failed_list))
            else:
                st.success("✅ 已徹底清除三個本地 Parquet 快取與記憶體資料！下次點選擷取將全面重新抓取。")
            
            time.sleep(0.5)
            st.rerun()

st.sidebar.markdown("---")

model_mode = st.sidebar.radio(
    "權重計算模式",
    ["固定專家權重", "期現水準交互動態滾動配適 (Interaction MLE)"],
    index=2,
    help="第2因子採用外資期指留倉淨額(tx_foreign_oi)，第3因子採用期現貨水準交乘項",
)

chip_transform_mode = st.sidebar.selectbox(
    "🧪 外資籌碼特徵變換 (現貨 & 期貨同步)",
    [
        "原始線性水準項",
        "雙對稱對數 (Signed-Log: sign(x)*ln(1+|x|))",
    ],
    index=1,
    help="預設採用原始線性水準項，亦可切換為雙對稱對數以壓縮厚尾極值",
)

engine = TaiexLogisticEngine(transform_mode=chip_transform_mode)

DATA_WINDOW_DAYS = 260
TRAIN_SAMPLE_DAYS = 250
OOS_TEST_DAYS = 10

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
    f"📌 **當沖交易規則**：以 **{selected_target_date}** 近月臺指期開盤價進場，**當日 13:45 收盤價強制沖銷結倉，不留倉過夜**。"
)


def safe_clamp(val, min_v, max_v, default_v=0.0):
    try:
        f_val = float(val)
        if np.isnan(f_val):
            return float(default_v)
        return float(max(min(f_val, max_v), min_v))
    except Exception:
        return float(default_v)


def reset_slider_states(f_net, t_net, tx_oi, usdtwd_r, sox_r, tsm_r, vix_c):
    st.session_state["slider_f_net"] = safe_clamp(f_net, -1200.0, 1200.0, 0.0)
    st.session_state["slider_t_net"] = safe_clamp(t_net, -300.0, 300.0, 0.0)
    st.session_state["slider_tx_oi"] = safe_clamp(
        tx_oi, -80000.0, 50000.0, -12000.0
    )
    st.session_state["slider_usdtwd_r"] = safe_clamp(usdtwd_r, -2.0, 2.0, 0.0)
    st.session_state["slider_sox_r"] = safe_clamp(sox_r, -15.0, 15.0, 0.0)
    st.session_state["slider_tsm_r"] = safe_clamp(tsm_r, -15.0, 15.0, 0.0)
    st.session_state["slider_vix_c"] = safe_clamp(vix_c, 5.0, 80.0, 18.0)


need_fetch = (
    "cached_data" not in st.session_state
    or fetch_btn
    or st.session_state.get("last_target_date") != selected_target_date
)

if need_fetch:
    with st.spinner(
        f"正在極速擷取 260 交易日真實歷史數據（前250天訓練，後10天OOS樣本外檢驗）..."
    ):
        twse_df, twse_actual_date, twse_msg = (
            pipeline.fetch_twse_institutional_with_fallback(
                selected_target_date
            )
        )
        taifex_df, taifex_actual_date, taifex_msg = (
            pipeline.fetch_taifex_oi_with_fallback(selected_target_date)
        )

        us_df, us_actual_date, us_msg = pipeline.fetch_us_market(
            selected_target_date, days_back=40
        )

        hist_df = pipeline.fetch_historical_training_and_verification_data(
            selected_target_date, exact_train_days=DATA_WINDOW_DAYS
        )

        hist_chip_start = selected_target_date - datetime.timedelta(
            days=int(DATA_WINDOW_DAYS * 1.65) + 60
        )
        hist_chip_end = selected_target_date

        hist_twse_df = pipeline.fetch_twse_historical_range_parquet(
            hist_chip_start, hist_chip_end
        )
        
        target_dates_list = hist_twse_df.index.tolist() if not hist_twse_df.empty else None

        hist_taifex_df = pipeline.fetch_taifex_historical_range_parquet(
            hist_chip_start, hist_chip_end, target_trading_dates=target_dates_list
        )

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

        latest_us = us_df.iloc[-1] if not us_df.empty else None
        usdtwd_r = float(latest_us["usdtwd_ret"]) if latest_us is not None and "usdtwd_ret" in latest_us else 0.0
        sox_r = float(latest_us["sox_ret"]) if latest_us is not None else 0.0
        tsm_r = float(latest_us["tsm_ret"]) if latest_us is not None else 0.0
        vix_c = float(latest_us["vix_close"]) if latest_us is not None else 18.0

        us_macro_event = US_MACRO_EVENTS_MAP.get(us_actual_date, "")

        st.session_state["cached_data"] = {
            "target_date": selected_target_date,
            "base_feature_date": selected_target_date,
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
            "usdtwd_r": usdtwd_r,
            "sox_r": sox_r,
            "tsm_r": tsm_r,
            "vix_c": vix_c,
            "hist_df": hist_df,
            "hist_twse_df": hist_twse_df,
            "hist_taifex_df": hist_taifex_df,
        }
        st.session_state["last_target_date"] = selected_target_date
        reset_slider_states(f_net, t_net, tx_oi, usdtwd_r, sox_r, tsm_r, vix_c)

elif reset_btn:
    c = st.session_state["cached_data"]
    reset_slider_states(
        c["f_net"], c["t_net"], c["tx_oi"], c.get("usdtwd_r", 0.0), c["sox_r"], c["tsm_r"], c["vix_c"]
    )
    st.toast("已還原至真實特徵數值！", icon="🔄")

data = st.session_state["cached_data"]
target_dt = pd.to_datetime(data["target_date"])

# -----------------------------------------------------------------------------
# 核心架構：嚴格以前 250 天訓練模型並尋優門檻，後 10 天做純樣本外 (OOS) 盲測
# -----------------------------------------------------------------------------
cost_rate_setting = 0.03
hist_df_full = data["hist_df"]

if len(hist_df_full) >= DATA_WINDOW_DAYS:
    train_sample_df = hist_df_full.iloc[:TRAIN_SAMPLE_DAYS].copy()
    oos_eval_df = hist_df_full.iloc[TRAIN_SAMPLE_DAYS:].copy()
else:
    train_sample_df = hist_df_full.iloc[:-OOS_TEST_DAYS].copy() if len(hist_df_full) > OOS_TEST_DAYS else hist_df_full.copy()
    oos_eval_df = hist_df_full.tail(min(OOS_TEST_DAYS, len(hist_df_full))).copy()

if (
    model_mode == "期現水準交互動態滾動配適 (Interaction MLE)"
    and not train_sample_df.empty
):
    engine.fit_model_on_sample(
        train_sample=train_sample_df,
        hist_twse=data.get("hist_twse_df"),
        hist_taifex=data.get("hist_taifex_df"),
    )

    if len(engine.in_sample_probs) >= 40:
        in_sample_probs_pct = [p * 100 for p in engine.in_sample_probs]
        (
            joint_best_bull,
            joint_best_bear,
            joint_best_acc,
            joint_best_net_ret,
            joint_best_score,
            joint_score_matrix,
            joint_text_matrix,
        ) = optimize_thresholds_fixed_days(
            probs_pct=in_sample_probs_pct,
            eval_df=engine.in_sample_df,
            acc_pct=engine.fit_accuracy,
            bull_range=(50, 75),
            bear_range=(25, 50),
            step=5,
            cost_rate_pct=cost_rate_setting,
        )
    else:
        joint_best_bull = 60
        joint_best_bear = 40
        joint_best_acc = 50.0
        joint_best_net_ret = 0.0
        joint_best_score = 0.0
        joint_score_matrix = pd.DataFrame()
        joint_text_matrix = pd.DataFrame()
else:
    engine.reset_to_static()
    joint_best_bull = 60
    joint_best_bear = 40
    joint_best_acc = 50.0
    joint_best_net_ret = 0.0
    joint_best_score = 0.0
    joint_score_matrix = pd.DataFrame()
    joint_text_matrix = pd.DataFrame()

if "slider_f_net" not in st.session_state:
    reset_slider_states(
        data["f_net"],
        data["t_net"],
        data["tx_oi"],
        data.get("usdtwd_r", 0.0),
        data["sox_r"],
        data["tsm_r"],
        data["vix_c"],
    )

st.sidebar.caption("💡 可微調各項因子數值進行即時敏感度測試：")

sim_foreign_net = st.sidebar.slider(
    "外資現貨買賣超 (億元)",
    min_value=-1200.0,
    max_value=1200.0,
    value=st.session_state["slider_f_net"],
    step=1.0,
    key="slider_f_net",
)
sim_trust_net = st.sidebar.slider(
    "投信現貨買賣超 (億元) [僅供參考]",
    min_value=-300.0,
    max_value=300.0,
    value=st.session_state["slider_t_net"],
    step=1.0,
    key="slider_t_net",
    help="已從 Logistic 迴歸中排除，僅作為盤面輔助觀察",
)
sim_tx_oi = st.sidebar.slider(
    "外資期指未平倉餘額淨額 OI (口)",
    min_value=-80000.0,
    max_value=50000.0,
    value=st.session_state["slider_tx_oi"],
    step=100.0,
    key="slider_tx_oi",
)
sim_usdtwd_ret = st.sidebar.slider(
    "美元兌台幣 (USD/TWD) 漲跌幅 (%)",
    min_value=-2.0,
    max_value=2.0,
    value=st.session_state.get("slider_usdtwd_r", 0.0),
    step=0.05,
    key="slider_usdtwd_r",
    help="正值代表美元升值／台幣貶值（資金流出偏空）；負值代表台幣升值（熱錢流入偏多）",
)
sim_sox_ret = st.sidebar.slider(
    "費半指數漲跌幅 (%)",
    min_value=-15.0,
    max_value=15.0,
    value=st.session_state["slider_sox_r"],
    step=0.1,
    key="slider_sox_r",
)
sim_tsm_ret = st.sidebar.slider(
    "台積電 ADR 漲跌幅 (%)",
    min_value=-15.0,
    max_value=15.0,
    value=st.session_state["slider_tsm_r"],
    step=0.1,
    key="slider_tsm_r",
)
sim_vix_close = st.sidebar.slider(
    "VIX 恐慌指數",
    min_value=5.0,
    max_value=80.0,
    value=st.session_state["slider_vix_c"],
    step=0.1,
    key="slider_vix_c",
)

trans_sim_f = engine.transform_foreign_net_series(sim_foreign_net)
trans_sim_tx = engine.transform_oi_series(sim_tx_oi)

mu_f = engine.active_scaler.get("foreign_net", {}).get("mu", 0.0)
sig_f = engine.active_scaler.get("foreign_net", {}).get("sigma", 1.0)
mu_tx = engine.active_scaler.get("tx_foreign_oi", {}).get("mu", 0.0)
sig_tx = engine.active_scaler.get("tx_foreign_oi", {}).get("sigma", 1.0)
current_z_f = (trans_sim_f - mu_f) / (sig_f if sig_f > 0 else 1.0)
current_z_tx = (trans_sim_tx - mu_tx) / (sig_tx if sig_tx > 0 else 1.0)
current_fx_inter = current_z_f * current_z_tx

current_features = {
    "foreign_net": sim_foreign_net,
    "tx_foreign_oi": sim_tx_oi,
    "fx_inter": current_fx_inter,
    "usdtwd_ret": sim_usdtwd_ret,
    "sox_ret": sim_sox_ret,
    "tsm_ret": sim_tsm_ret,
    "vix_close": sim_vix_close,
}

prob_up, contributions = engine.predict(current_features)

# -----------------------------------------------------------------------------
# 7. 主視覺顯示 (Dashboard View)
# -----------------------------------------------------------------------------
st.title("🎯 TAIEX 台灣加權指數次日漲跌機率監控儀表 (260日OOS樣本外檢驗版)")

m_desc = (
    f"期現水準交互動態 MLE 配適 (訓練樣本: 前 {len(train_sample_df)} 營業日, 訓練準確率: {engine.fit_accuracy:.1f}% | 樣本外測試: 最近 10 個營業日)"
    if engine.fit_accuracy is not None
    else "固定專家權重"
)

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

st.markdown(
    f"""
> **系統運行狀態與跨市場因果檢驗**  
> • **目標預測交易日 (T)**：`{data['target_date']}` | **模型權重模式**：`{m_desc}`  
> • **交易執行標的**：**臺指近月期貨 (TXF Front-Month)** | **執行策略**：**開盤價進場，收盤價強制沖銷，不留倉過夜**  
> • **架構切分**：**數據採樣 260 交易日 $\\rightarrow$ 前 250 日訓練尋優 (多方門檻: {joint_best_bull}% | 空方門檻: {joint_best_bear}%)；後 10 日為純樣本外 (OOS) 盲測**  
> • **同日嚴格對齊**：**同一列美台交易日嚴格同一天（T 日台股對齊 T 日美股）**  
> • **證券交易所 (TWSE)**：外資現貨買賣超 (擷取日期: `{data['twse_actual_date']}` | `狀態: {data['twse_msg']}`)  
> • **期貨交易所 (TAIFEX)**：臺股期貨外資留倉淨額 (擷取日期: `{data['taifex_actual_date']}` | `狀態: {data['taifex_msg']}`)  
> • **美股行情 (yfinance)**：費半、TSM ADR、VIX、USD/TWD (擷取日期: `{data['us_actual_date']}` | `狀態: {data['us_msg']}`)
"""
)

# -----------------------------------------------------------------------------
# 8. 提供 260 交易日真實歷史數據與「績效統計摘要表＋淨值走勢圖」下載
# -----------------------------------------------------------------------------
if not data["hist_df"].empty:
    file_bytes, file_ext, mime_type = export_260_days_to_excel_with_summary(
        hist_df=data["hist_df"],
        hist_twse=data.get("hist_twse_df"),
        hist_taifex=data.get("hist_taifex_df"),
        engine=engine,
        best_bull=joint_best_bull,
        best_bear=joint_best_bear,
        cost_rate_pct=cost_rate_setting,
    )
    st.download_button(
        label=f"📥 下載 260 交易日台美數據 (含績效摘要與淨值走勢圖) ({file_ext.upper()})",
        data=file_bytes,
        file_name=f"TAIEX_US_260_Days_Performance_{data['target_date']}.{file_ext}",
        mime=mime_type,
        help="下載包含「前250日當沖績效摘要與原生淨值走勢圖」、「淨值曲線數據」與「260日真實交易數據明細」之完整試算表",
        use_container_width=True,
    )

st.markdown("---")

col_gauge, col_waterfall = st.columns([5, 5])

with col_gauge:
    st.plotly_chart(
        render_gauge(prob_up, target_date=data["target_date"]),
        use_container_width=True,
    )
    if prob_up >= (joint_best_bull / 100.0):
        st.markdown(
            "<h3 style='text-align: center;' class='text-bullish'>當沖指令：開盤做多臺指期"
            f" (勝率 {prob_up*100:.1f}% ≥ {joint_best_bull}%)</h3>",
            unsafe_allow_html=True,
        )
    elif prob_up <= (joint_best_bear / 100.0):
        st.markdown(
            "<h3 style='text-align: center;' class='text-bearish'>當沖指令：開盤放空臺指期"
            f" (勝率 {prob_up*100:.1f}% ≤ {joint_best_bear}%)</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h3 style='text-align: center; color: #94a3b8;'>當沖指令：區間觀望空手"
            f" (勝率 {prob_up*100:.1f}% 介於門檻間)</h3>",
            unsafe_allow_html=True,
        )
    st.caption(f"⭐ **目標交易日 ({data['target_date']}) 為前 250 日模型訓練完成後之第 11 筆純樣本外即時預測**。")

with col_waterfall:
    oi_tag = " (Signed-Log)" if "Signed-Log" in chip_transform_mode else ""
    st.plotly_chart(
        render_waterfall(contributions, tag_extra=oi_tag),
        use_container_width=True,
    )

with st.expander(
    f"🔍 檢視當前目標日 (T) 7 大特徵之動態最佳化 Beta 權重與常模 (前 {TRAIN_SAMPLE_DAYS} 交易日訓練樣本)"
):
    w_df = pd.DataFrame([
        {
            "特徵因子": k,
            "最佳化 Beta 權重": round(engine.active_beta.get(k, 0.0), 4),
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

st.subheader("📊 盤前已知核心特徵數值一覽 (台美開休日曆獨立對齊)")
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

with m1:
    f_disp = f"{sim_foreign_net:+.1f} 億"
    if "Signed-Log" in chip_transform_mode:
        f_disp += f" (s-ln={trans_sim_f:.2f})"

    st.metric(
        "外資現貨買賣超",
        f_disp,
        delta=f"Beta: {engine.active_beta.get('foreign_net', 0.0):+.2f}",
        delta_color="normal" if sim_foreign_net >= 0 else "inverse",
    )
    st.caption(f"來源: TWSE ({data['twse_actual_date']})")

with m2:
    oi_status = "淨多單" if sim_tx_oi >= 0 else "淨空單"
    val_disp = f"{int(sim_tx_oi):+,} 口"
    if "Signed-Log" in chip_transform_mode:
        val_disp += f" (s-ln={trans_sim_tx:.2f})"

    st.metric(
        "外資期指留倉淨額 (OI)",
        val_disp,
        delta=(
            f"{oi_status} (Beta:"
            f" {engine.active_beta.get('tx_foreign_oi', 0.0):+.2f})"
        ),
        delta_color="normal" if sim_tx_oi >= 0 else "inverse",
    )
    st.caption(f"來源: TAIFEX ({data['taifex_actual_date']})")

with m3:
    inter_status = (
        "同向共振"
        if current_fx_inter > 0.2
        else ("反向避險" if current_fx_inter < -0.2 else "中性平衡")
    )
    st.metric(
        "期現交互共振項",
        f"{current_fx_inter:+.2f}",
        delta=(
            f"{inter_status} (Beta:"
            f" {engine.active_beta.get('fx_inter', 0.0):+.2f})"
        ),
        delta_color="normal" if current_fx_inter >= 0 else "inverse",
    )
    st.caption("Z(現貨) × Z(期指)")

with m4:
    fx_status = "台幣貶值 (流出)" if sim_usdtwd_ret > 0 else ("台幣升值 (流入)" if sim_usdtwd_ret < 0 else "平盤持平")
    st.metric(
        "USD/TWD 匯率變動",
        f"{sim_usdtwd_ret:+.2f}%",
        delta=f"{fx_status} (Beta: {engine.active_beta.get('usdtwd_ret', 0.0):+.2f})",
        delta_color="inverse" if sim_usdtwd_ret > 0 else "normal",
    )
    st.caption("來源: yfinance (TWD=X)")

with m5:
    ev_tag = (
        f" [{data['us_macro_event']}]" if data.get("us_macro_event") else ""
    )
    st.metric(
        "美股費半 (SOX)",
        f"{sim_sox_ret:+.2f}%",
        delta=f"Beta: {engine.active_beta.get('sox_ret', 0.0):+.2f}",
        delta_color="normal" if sim_sox_ret >= 0 else "inverse",
    )
    st.caption(f"來源: yfinance ({data['us_actual_date']}){ev_tag}")

with m6:
    st.metric(
        "台積電 ADR",
        f"{sim_tsm_ret:+.2f}%",
        delta=f"Beta: {engine.active_beta.get('tsm_ret', 0.0):+.2f}",
        delta_color="normal" if sim_tsm_ret >= 0 else "inverse",
    )
    st.caption(f"來源: yfinance ({data['us_actual_date']})")

with m7:
    st.metric(
        "美股 VIX 指數",
        f"{sim_vix_close:.1f}",
        delta=f"Beta: {engine.active_beta.get('vix_close', 0.0):+.2f}",
        delta_color="inverse" if sim_vix_close >= 20 else "normal",
    )
    st.caption(f"來源: yfinance ({data['us_actual_date']})")

# -----------------------------------------------------------------------------
# 9. 臺指期純當沖 (不留倉) 過往 10 個台股交易日【純樣本外 (OOS) 結算】
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader(
    "📈 臺指期純當沖 (開盤進、收盤平 | 不留倉) 過往 10 個台股交易日純樣本外 (OOS) 結算"
)
st.caption(
    "🛡️ **無偏估計說明**：本段 10 個交易日（第 251 至 260 筆資料）**完全未參與模型的任何訓練與門檻尋優**，純粹用於驗證前 250 天模型的真實樣本外預測能力。"
)

hist_twse = data.get("hist_twse_df", pd.DataFrame())
hist_taifex = data.get("hist_taifex_df", pd.DataFrame())

if not oos_eval_df.empty and len(oos_eval_df) >= 5:
    ten_days_df = oos_eval_df.tail(OOS_TEST_DAYS).copy()
    calc_probs = []

    for i in range(len(ten_days_df)):
        current_date_idx = ten_days_df.index[i]
        curr_tw_row = ten_days_df.loc[current_date_idx]

        f_val = 5.0
        if not hist_twse.empty and "foreign_net" in hist_twse.columns:
            matched_twse = hist_twse[hist_twse.index <= current_date_idx]
            if not matched_twse.empty:
                f_val = float(matched_twse.iloc[-1]["foreign_net"])

        tx_oi_val = -12000.0
        if not hist_taifex.empty and "tx_foreign_oi" in hist_taifex.columns:
            matched_taifex = hist_taifex[hist_taifex.index <= current_date_idx]
            if not matched_taifex.empty:
                tx_oi_val = float(matched_taifex.iloc[-1]["tx_foreign_oi"])

        hist_feat = {
            "foreign_net": f_val,
            "tx_foreign_oi": tx_oi_val,
            "usdtwd_ret": float(curr_tw_row.get("usdtwd_ret", 0.0)),
            "sox_ret": float(curr_tw_row["sox_ret"]),
            "tsm_ret": float(curr_tw_row["tsm_ret"]),
            "vix_close": float(curr_tw_row["vix_close"]),
        }

        p, _ = engine.predict(hist_feat)
        calc_probs.append(round(p, 4))

    ten_days_df["Prob_Up"] = calc_probs
    date_labels = ten_days_df.index.strftime("%Y-%m-%d").tolist()
    probs_pct = [p * 100 for p in ten_days_df["Prob_Up"]]

    txf_open_prices = ten_days_df["TXF_Open"].tolist()
    txf_close_prices = ten_days_df["TXF_Close"].tolist()
    gaps_list = ten_days_df["gap_ret"].tolist()
    filled_list = ten_days_df["gap_filled"].tolist()
    us_dates_list = (
        ten_days_df["us_date"].tolist()
        if "us_date" in ten_days_df
        else []
    )
    us_events_list = (
        ten_days_df["us_event"].tolist()
        if "us_event" in ten_days_df
        else []
    )

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
        "##### 📌 樣本外 10 個台股交易日開盤跳空統計看板 (Gap Statistics)"
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
            delta=f"10 日中回補 {sum(valid_fills)} 天",
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

    ctrl_col0, ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 3, 3, 2.5])
    with ctrl_col0:
        auto_tune_thresholds = st.checkbox(
            "⚡ 自動套用前 250 日最佳當沖門檻",
            value=True,
            help=f"套用前 {TRAIN_SAMPLE_DAYS} 交易日訓練樣本下 MAX(準確度×淨利) 全域最佳解",
        )
    with ctrl_col1:
        if auto_tune_thresholds:
            threshold_bull = joint_best_bull
            st.metric(
                "🎯 多方當沖發動門檻",
                f"{threshold_bull}%",
                delta=f"訓練期評分: {joint_best_score:+.2f}",
            )
        else:
            threshold_bull = st.slider(
                "🎯 多方當沖發動門檻 (%)", 50, 75, joint_best_bull, step=5
            )
    with ctrl_col2:
        if auto_tune_thresholds:
            threshold_bear = joint_best_bear
            st.metric(
                "🛡️ 空方當沖防禦門檻",
                f"{threshold_bear}%",
                delta=f"前{TRAIN_SAMPLE_DAYS}日淨利: {joint_best_net_ret:+.1f}%",
            )
        else:
            threshold_bear = st.slider(
                "🛡️ 空方當沖防禦門檻 (%)", 25, 50, joint_best_bear, step=5
            )
    with ctrl_col3:
        cost_rate_pct = st.number_input(
            "💸 單趟摩擦成本率 (%)",
            min_value=0.00,
            max_value=0.20,
            value=0.03,
            step=0.005,
            format="%.3f",
            help="包含期交稅、手續費與進出滑價（當沖一進一出共計 2 次）",
        )

    with st.expander(
        f"🔥 檢視前 {TRAIN_SAMPLE_DAYS} 交易日訓練樣本門檻評分熱力圖 (目標: MAX(準確度×當沖淨利))",
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
                ),
                use_container_width=True,
            )

    st.caption(
        f"💡 **純樣本外當沖會計基準**：以 **前 250 天訓練得出之固定多空門檻 ({threshold_bull}% / {threshold_bear}%)** 於最近 10 個營業日開盤價進場，**當日 13:45 收盤價強制沖銷，不留倉過夜**。每次當沖一進一出扣除 **{cost_rate_pct*2.0:.3f}%** 摩擦成本。"
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
    )

    trades_df = extract_day_trades(
        date_labels,
        txf_open_prices,
        txf_close_prices,
        positions,
        gross_returns,
        friction_costs,
        net_returns,
    )

    bar_colors = ["#f43f5e" if r >= 0 else "#10b981" for r in gross_returns]
    bar_opacities = [0.85 if pos != 0 else 0.3 for pos in positions]

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
    win_rate = (
        (correct_signals / active_signals * 100.0)
        if active_signals > 0
        else 0.0
    )

    valid_net_returns = [
        nr for nr, pos in zip(net_returns, positions) if pos != 0
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

        annualized_net_return = (total_net_return / 10.0) * 252.0
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
            "10日樣本外勝率",
            f"{win_rate:.1f}%" if active_signals > 0 else "無觸發",
            delta=f"出手 {active_signals} 次 / 獲利 {correct_signals} 次",
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
            "隔夜跳空風險",
            "0.00% (零留倉)",
            delta="每日13:45強制結清",
            delta_color="off",
        )
    with k4:
        st.metric(
            "累計手續與稅費損耗",
            f"-{total_cost_spent:.3f}%",
            delta=f"交易次數: {active_signals} 次",
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
            name="臺指期當日當沖毛損益 (%) [開盤進、收盤平]",
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
        annotation_text=f"多方當沖門檻 ({threshold_bull}%)",
        annotation_position="bottom right",
        annotation_font_color="#f43f5e",
    )
    fig_verify.add_hline(
        y=threshold_bear,
        line_dash="dot",
        line_color="rgba(16, 185, 129, 0.7)",
        annotation_text=f"空方當沖門檻 ({threshold_bear}%)",
        annotation_position="top right",
        annotation_font_color="#10b981",
    )

    fig_verify.update_layout(
        title=(
            "<b>最近 10 個台股交易日【純樣本外預測勝率 vs 臺指期當沖實際損益】疊加比對</b>"
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
            title="當日近月臺指期當沖漲跌幅 (%)",
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
            "<b>最近 10 日純樣本外當沖累積模擬淨值曲線 (OOS Equity Curve) | Sharpe:"
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
        st.markdown("##### 📋 最近 10 日純樣本外逐筆當沖交易明細 (OOS Day-Trades)")
        display_trades = trades_df.drop(
            columns=["_net_val"], errors="ignore"
        )
        st.dataframe(
            display_trades, use_container_width=True, hide_index=True
        )

    st.markdown(
        "##### 📑 最近 10 日純樣本外當沖市價結算流水帳 (含美股同日對齊與重大事件)"
    )
    pos_labels = [
        (
            "🟢 當沖做多 (+1)"
            if p == 1
            else ("🔴 當沖放空 (-1)" if p == -1 else "⚪ 當日空手 (0)")
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

    table_df = pd.DataFrame({
        "樣本類別": ["純樣本外測試 (OOS)"] * len(date_labels),
        "目標交易日 (T)": date_labels,
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
        "當沖訊號": signal_types,
        "當沖部位": pos_labels,
        "會計執行動作": validation_status,
        "當日當沖毛損益": [
            f"{gr:+.2f}%" if not np.isnan(gr) else "尚未結算"
            for gr in gross_returns
        ],
        "當沖成本 (一進一出)": [
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
