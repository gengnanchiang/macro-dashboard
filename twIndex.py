# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:55:00 2026

@author: James Chiang
"""

import datetime
import io
import time
from typing import Dict, Tuple, Optional
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
    page_title="TAIEX 次日漲跌機率監控儀表板",
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
    datetime.date(2026, 10, 26): "光復節補假（休市）",
}


def check_is_taiwan_holiday(d: datetime.date) -> Tuple[bool, str]:
    """檢驗所選日期是否為週末或台灣休假日"""
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
# 1. 資料爬蟲管線（證交所、期交所、yfinance）
# -----------------------------------------------------------------------------
class MarketDataPipeline:

    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

    def get_previous_trading_day(self, target_date: datetime.date) -> datetime.date:
        """尋找嚴格小於 target_date 的前一個台灣市場真實交易日"""
        d = target_date - datetime.timedelta(days=1)
        while d.weekday() >= 5 or d in TW_HOLIDAYS_MAP:
            d -= datetime.timedelta(days=1)
        return d

    def get_previous_weekday(self, target_date: datetime.date) -> datetime.date:
        return self.get_previous_trading_day(target_date)

    def fetch_twse_single_day(
        self, target_date: datetime.date
    ) -> Tuple[pd.DataFrame, bool, str]:
        if target_date.weekday() >= 5 or target_date in TW_HOLIDAYS_MAP:
            return pd.DataFrame(), False, f"休市或非交易日 ({target_date.strftime('%Y%m%d')})"

        date_str = target_date.strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/zh/fund/BFI82U?response=json&dayDate={date_str}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=8)
            data = resp.json()
            if data.get("stat") != "OK" or "data" not in data or len(data["data"]) == 0:
                return pd.DataFrame(), False, f"查無資料或休市 ({date_str})"

            df = pd.DataFrame(data["data"], columns=["item", "buy", "sell", "net"])
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
                    note = "正常取得" if curr == start_date else f"原日前推至有效交易日 ({curr.strftime('%Y-%m-%d')})"
                    return df, curr, note
            curr -= datetime.timedelta(days=1)
        return pd.DataFrame(), start_date, "超過回溯天數，查無有效數據"

    def fetch_twse_historical_range(
        self, start_d: datetime.date, end_d: datetime.date
    ) -> pd.DataFrame:
        records = []
        curr = start_d
        while curr <= end_d:
            if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                df, ok, _ = self.fetch_twse_single_day(curr)
                if ok and not df.empty:
                    records.append(df)
                time.sleep(0.06)
            curr += datetime.timedelta(days=1)
        if records:
            return pd.concat(records).sort_index()
        return pd.DataFrame()

    def fetch_taifex_single_day(
        self, target_date: datetime.date
    ) -> Tuple[pd.DataFrame, bool, str]:
        if target_date.weekday() >= 5 or target_date in TW_HOLIDAYS_MAP:
            return pd.DataFrame(), False, f"休市或非交易日 ({target_date.strftime('%Y/%m/%d')})"

        date_str = target_date.strftime("%Y/%m/%d")
        url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
        payload = {
            "queryStartDate": date_str,
            "queryEndDate": date_str,
            "commodityId": "TXF",
        }
        try:
            resp = requests.post(url, data=payload, headers=self.headers, timeout=8)
            resp.encoding = "big5"
            content = resp.text

            if (
                "查無資料" in content
                or resp.status_code != 200
                or len(content.strip()) == 0
                or "HTML" in content.upper()
            ):
                return pd.DataFrame(), False, f"查無資料或休市 ({date_str})"

            df = pd.read_csv(io.StringIO(content))
            df.columns = [str(c).strip() for c in df.columns]

            name_col = [c for c in df.columns if "商品名稱" in c]
            id_col = [c for c in df.columns if "身份別" in c]

            if not name_col or not id_col:
                return pd.DataFrame(), False, "期交所 CSV 格式解析失敗"

            df = df[df[name_col[0]].astype(str).str.strip() == "臺股期貨"]

            id_map = {
                "外資及陸資": "tx_foreign_oi",
                "外資及陸資(不含外資自營商)": "tx_foreign_oi",
                "投信": "tx_trust_oi",
                "自營商": "tx_dealer_oi",
            }

            res = {"date": pd.to_datetime(target_date)}
            for _, r in df.iterrows():
                ident = str(r[id_col[0]]).strip()
                if ident in id_map:
                    raw_val = str(r.iloc[-2]).replace(",", "").replace(" ", "")
                    try:
                        res[id_map[ident]] = float(raw_val)
                    except ValueError:
                        pass

            if "tx_foreign_oi" in res:
                return pd.DataFrame([res]).set_index("date"), True, "正常取得"
            else:
                return pd.DataFrame(), False, f"未能在臺股期貨中找到外資留倉數據 ({date_str})"
        except Exception as e:
            return pd.DataFrame(), False, f"擷取異常: {str(e)}"

    def fetch_taifex_oi_with_fallback(
        self, start_date: datetime.date, max_lookback_days: int = 10
    ) -> Tuple[pd.DataFrame, datetime.date, str]:
        """從指定日期（含）開始向前回溯，嚴格不擷取未來日期"""
        curr = start_date
        while (start_date - curr).days <= max_lookback_days:
            if curr.weekday() < 5 and curr not in TW_HOLIDAYS_MAP:
                df, ok, msg = self.fetch_taifex_single_day(curr)
                if ok and not df.empty:
                    note = "正常取得" if curr == start_date else f"原日前推至有效交易日 ({curr.strftime('%Y-%m-%d')})"
                    return df, curr, note
            curr -= datetime.timedelta(days=1)
        return pd.DataFrame(), start_date, "超過回溯天數，查無有效數據"

    def fetch_taifex_historical_range(
        self, start_d: datetime.date, end_d: datetime.date
    ) -> pd.DataFrame:
        url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
        payload = {
            "queryStartDate": start_d.strftime("%Y/%m/%d"),
            "queryEndDate": end_d.strftime("%Y/%m/%d"),
            "commodityId": "TXF",
        }
        try:
            resp = requests.post(url, data=payload, headers=self.headers, timeout=12)
            resp.encoding = "big5"
            content = resp.text
            if "查無資料" in content or resp.status_code != 200:
                return pd.DataFrame()

            df = pd.read_csv(io.StringIO(content))
            df.columns = [str(c).strip() for c in df.columns]

            date_col = [c for c in df.columns if "日期" in c][0]
            name_col = [c for c in df.columns if "商品名稱" in c][0]
            id_col = [c for c in df.columns if "身份別" in c][0]

            df = df[df[name_col].astype(str).str.strip() == "臺股期貨"]
            df = df[df[id_col].astype(str).str.strip().isin(["外資及陸資", "外資及陸資(不含外資自營商)"])]

            rows = []
            for _, r in df.iterrows():
                d_str = str(r[date_col]).strip()
                dt = pd.to_datetime(d_str)
                # 嚴格過濾避免任何大於 end_d 的日期
                if dt.date() > end_d:
                    continue
                raw_val = str(r.iloc[-2]).replace(",", "").replace(" ", "")
                try:
                    oi_val = float(raw_val)
                    rows.append({"date": dt, "tx_foreign_oi": oi_val})
                except ValueError:
                    continue

            if rows:
                return pd.DataFrame(rows).drop_duplicates(subset=["date"]).set_index("date").sort_index()
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def fetch_us_market(
        self, target_date: datetime.date, days_back: int = 15
    ) -> Tuple[pd.DataFrame, str]:
        tickers = {
            "^SOX": "sox_ret",
            "TSM": "tsm_ret",
            "^GSPC": "sp500_ret",
            "^VIX": "vix_close",
        }
        try:
            start_d = (target_date - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
            end_d = (target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            raw = yf.download(
                list(tickers.keys()),
                start=start_d,
                end=end_d,
                progress=False,
                auto_adjust=False,
            )
            price = raw["Close"] if "Close" in raw else raw["Adj Close"]

            features = pd.DataFrame(index=price.index)
            for tk, name in tickers.items():
                if name == "vix_close":
                    features[name] = price[tk]
                else:
                    features[name] = price[tk].pct_change() * 100.0

            features = features.dropna()
            features.index = pd.to_datetime(features.index).tz_localize(None)
            # 確保美股只保留到 target_date 之前或當日
            features = features[features.index.date <= target_date]
            return features, "正常取得"
        except Exception as e:
            return pd.DataFrame(), f"擷取異常: {str(e)}"

    def fetch_historical_training_and_verification_data(
        self, target_date: datetime.date, max_train_days: int = 750
    ) -> pd.DataFrame:
        try:
            calendar_days = int(max_train_days * 1.65) + 60
            start_d = (target_date - datetime.timedelta(days=calendar_days)).strftime("%Y-%m-%d")
            end_d = (target_date + datetime.timedelta(days=5)).strftime("%Y-%m-%d")

            tickers = {
                "^TWII": "twii",
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

            if isinstance(raw.columns, pd.MultiIndex):
                twii_open = raw["Open"]["^TWII"]
                twii_high = raw["High"]["^TWII"]
                twii_low = raw["Low"]["^TWII"]
                twii_close = raw["Close"]["^TWII"]
                sox_close = raw["Close"]["^SOX"]
                tsm_close = raw["Close"]["TSM"]
                vix_close = raw["Close"]["^VIX"]
            else:
                twii_open = raw["Open"]
                twii_high = raw["High"]
                twii_low = raw["Low"]
                twii_close = raw["Close"]
                sox_close = raw["Close"]
                tsm_close = raw["Close"]
                vix_close = raw["Close"]

            df = pd.DataFrame(index=raw.index)
            df["TWII_Open"] = twii_open
            df["TWII_High"] = twii_high
            df["TWII_Low"] = twii_low
            df["TWII_Close"] = twii_close
            df["sox_ret"] = sox_close.pct_change() * 100.0
            df["tsm_ret"] = tsm_close.pct_change() * 100.0
            df["vix_close"] = vix_close

            df["prev_twii_close"] = df["TWII_Close"].shift(1)
            df["gap_ret"] = ((df["TWII_Open"] - df["prev_twii_close"]) / df["prev_twii_close"]) * 100.0

            gap_up_filled = (df["gap_ret"] > 0) & (df["TWII_Low"] <= df["prev_twii_close"])
            gap_down_filled = (df["gap_ret"] < 0) & (df["TWII_High"] >= df["prev_twii_close"])
            df["gap_filled"] = gap_up_filled | gap_down_filled

            df["TWII_Open_to_Close"] = (
                (df["TWII_Close"] - df["TWII_Open"]) / df["TWII_Open"]
            ) * 100.0

            df["TWII_Close_to_Close"] = df["TWII_Close"].pct_change() * 100.0

            df = df.dropna(subset=["TWII_Open", "TWII_Close", "gap_ret", "sox_ret", "tsm_ret", "vix_close"])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception:
            return pd.DataFrame()


# -----------------------------------------------------------------------------
# 2. Logistic Regression 機率推論引擎 (導入 fx_inter 外資期現交互作用項)
# -----------------------------------------------------------------------------
class TaiexLogisticEngine:

    def __init__(self):
        self.ALL_FEATURES = [
            "foreign_net",
            "tx_foreign_oi",
            "fx_inter",
            "sox_ret",
            "tsm_ret",
            "vix_close",
        ]

        self.default_scaler = {
            "foreign_net": {"mu": 5.0, "sigma": 150.0},
            "tx_foreign_oi": {"mu": -12000.0, "sigma": 10000.0},
            "fx_inter": {"mu": 0.15, "sigma": 1.20},
            "sox_ret": {"mu": 0.05, "sigma": 1.8},
            "tsm_ret": {"mu": 0.10, "sigma": 2.2},
            "vix_close": {"mu": 17.5, "sigma": 4.5},
        }

        self.default_beta = {
            "foreign_net": 0.42,
            "tx_foreign_oi": 0.48,
            "fx_inter": 0.25,
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

    def fit_rolling_model(
        self,
        hist_df: pd.DataFrame,
        cutoff_dt: pd.Timestamp,
        train_days: int = 250,
        hist_twse: Optional[pd.DataFrame] = None,
        hist_taifex: Optional[pd.DataFrame] = None,
    ) -> bool:
        valid_df = hist_df[hist_df.index < cutoff_dt].copy()
        if len(valid_df) < 40:
            return False

        train_sample = valid_df.tail(train_days).copy()
        self.fit_sample_size = len(train_sample)

        train_sample["y"] = (train_sample["TWII_Open_to_Close"] > 0).astype(int)

        features_lagged = ["sox_ret", "tsm_ret", "vix_close"]
        X_raw = pd.DataFrame(index=train_sample.index)
        for col in features_lagged:
            X_raw[col] = train_sample[col].shift(1)

        f_list, tx_list = [], []
        for dt_idx in train_sample.index:
            loc = hist_df.index.get_loc(dt_idx) - 1
            if loc >= 0:
                p_dt = hist_df.index[loc]
                f_val = 5.0
                if hist_twse is not None and not hist_twse.empty:
                    m_twse = hist_twse[hist_twse.index <= p_dt]
                    if not m_twse.empty:
                        f_val = float(m_twse.iloc[-1].get("foreign_net", 5.0))

                tx_val = -12000.0
                if hist_taifex is not None and not hist_taifex.empty:
                    m_taifex = hist_taifex[hist_taifex.index <= p_dt]
                    if not m_taifex.empty:
                        tx_val = float(m_taifex.iloc[-1].get("tx_foreign_oi", -12000.0))

                f_list.append(f_val)
                tx_list.append(tx_val)
            else:
                f_list.append(5.0)
                tx_list.append(-12000.0)

        X_raw["foreign_net"] = f_list
        X_raw["tx_foreign_oi"] = tx_list

        X_raw = X_raw.dropna()
        y = train_sample.loc[X_raw.index, "y"]

        new_scaler = self.default_scaler.copy()
        for col in ["foreign_net", "tx_foreign_oi", "sox_ret", "tsm_ret", "vix_close"]:
            mu = float(X_raw[col].mean())
            sigma = float(X_raw[col].std()) if X_raw[col].std() > 0 else 1.0
            new_scaler[col] = {"mu": mu, "sigma": sigma}

        z_f = (X_raw["foreign_net"] - new_scaler["foreign_net"]["mu"]) / new_scaler["foreign_net"]["sigma"]
        z_tx = (X_raw["tx_foreign_oi"] - new_scaler["tx_foreign_oi"]["mu"]) / new_scaler["tx_foreign_oi"]["sigma"]
        X_raw["fx_inter"] = z_f * z_tx

        inter_mu = float(X_raw["fx_inter"].mean())
        inter_sigma = float(X_raw["fx_inter"].std()) if X_raw["fx_inter"].std() > 0 else 1.0
        new_scaler["fx_inter"] = {"mu": inter_mu, "sigma": inter_sigma}

        feature_cols = self.ALL_FEATURES
        X_scaled = pd.DataFrame(index=X_raw.index)
        for col in feature_cols:
            mu = new_scaler[col]["mu"]
            sigma = new_scaler[col]["sigma"]
            X_scaled[col] = (X_raw[col] - mu) / sigma

        clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200)
        clf.fit(X_scaled, y)

        self.active_scaler = new_scaler
        self.active_intercept = float(clf.intercept_[0])

        new_beta = {}
        for idx, col in enumerate(feature_cols):
            new_beta[col] = float(clf.coef_[0][idx])
        self.active_beta = new_beta

        self.fit_accuracy = float(clf.score(X_scaled, y)) * 100.0
        return True

    def find_best_train_days(
        self,
        hist_df: pd.DataFrame,
        cutoff_dt: pd.Timestamp,
        min_days: int = 60,
        max_days: int = 750,
        step: int = 10,
        hist_twse: Optional[pd.DataFrame] = None,
        hist_taifex: Optional[pd.DataFrame] = None,
    ) -> Tuple[int, float]:
        valid_df = hist_df[hist_df.index < cutoff_dt].copy()
        available_days = len(valid_df)
        upper_limit = min(max_days, available_days)

        best_days = 250
        best_acc = -1.0

        for d in range(min_days, upper_limit + 1, step):
            temp_eng = TaiexLogisticEngine()
            ok = temp_eng.fit_rolling_model(
                hist_df,
                cutoff_dt,
                train_days=d,
                hist_twse=hist_twse,
                hist_taifex=hist_taifex,
            )
            if ok and temp_eng.fit_accuracy is not None:
                if temp_eng.fit_accuracy >= best_acc:
                    best_acc = temp_eng.fit_accuracy
                    best_days = d

        return best_days, (best_acc if best_acc >= 0 else 50.0)

    def reset_to_static(self):
        self.active_scaler = self.default_scaler.copy()
        self.active_beta = self.default_beta.copy()
        self.active_intercept = self.default_intercept
        self.fit_accuracy = None
        self.fit_sample_size = 0

    def predict(self, raw_features: Dict[str, float]) -> Tuple[float, dict]:
        feats = raw_features.copy()
        if "fx_inter" not in feats:
            f_val = feats.get("foreign_net", 0.0)
            tx_val = feats.get("tx_foreign_oi", -12000.0)
            mu_f = self.active_scaler.get("foreign_net", {}).get("mu", 0.0)
            sig_f = self.active_scaler.get("foreign_net", {}).get("sigma", 1.0)
            mu_tx = self.active_scaler.get("tx_foreign_oi", {}).get("mu", 0.0)
            sig_tx = self.active_scaler.get("tx_foreign_oi", {}).get("sigma", 1.0)
            z_f = (f_val - mu_f) / (sig_f if sig_f > 0 else 1.0)
            z_tx = (tx_val - mu_tx) / (sig_tx if sig_tx > 0 else 1.0)
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

    def predict_out_of_sample_step(
        self,
        hist_df: pd.DataFrame,
        t_day: pd.Timestamp,
        features_dict: Dict[str, float],
        train_days: int = 250,
        use_dynamic: bool = True,
        hist_twse: Optional[pd.DataFrame] = None,
        hist_taifex: Optional[pd.DataFrame] = None,
    ) -> float:
        if not use_dynamic:
            p, _ = self.predict(features_dict)
            return p

        temp_engine = TaiexLogisticEngine()
        fitted = temp_engine.fit_rolling_model(
            hist_df,
            t_day,
            train_days=train_days,
            hist_twse=hist_twse,
            hist_taifex=hist_taifex,
        )
        if not fitted:
            temp_engine.reset_to_static()

        p, _ = temp_engine.predict(features_dict)
        return p


# -----------------------------------------------------------------------------
# 3. 雙變數信心門檻最佳化與熱力圖渲染函式
# -----------------------------------------------------------------------------
def optimize_thresholds_with_matrix(
    probs_pct: list,
    returns_list: list,
    bull_range: Tuple[int, int] = (50, 75),
    bear_range: Tuple[int, int] = (25, 50),
    step: int = 5,
) -> Tuple[int, int, float, int, pd.DataFrame, pd.DataFrame]:
    bull_steps = list(range(bull_range[0], bull_range[1] + 1, step))
    bear_steps = list(range(bear_range[0], bear_range[1] + 1, step))

    win_rate_matrix = pd.DataFrame(index=bull_steps, columns=bear_steps, dtype=float)
    text_matrix = pd.DataFrame(index=bull_steps, columns=bear_steps, dtype=str)

    best_bull = 60
    best_bear = 40
    best_win_rate = -1.0
    best_active = 0

    for bull in bull_steps:
        for bear in bear_steps:
            active = 0
            correct = 0
            for p, r in zip(probs_pct, returns_list):
                if np.isnan(r):
                    continue
                if p >= bull:
                    active += 1
                    if r > 0:
                        correct += 1
                elif p <= bear:
                    active += 1
                    if r < 0:
                        correct += 1

            if active > 0:
                win_rate = (correct / active) * 100.0
                win_rate_matrix.loc[bull, bear] = round(win_rate, 1)
                text_matrix.loc[bull, bear] = f"{win_rate:.0f}%<br>({correct}/{active}天)"
                if (win_rate > best_win_rate) or (win_rate == best_win_rate and active > best_active):
                    best_win_rate = win_rate
                    best_bull = bull
                    best_bear = bear
                    best_active = active
            else:
                win_rate_matrix.loc[bull, bear] = np.nan
                text_matrix.loc[bull, bear] = "無信號"

    return best_bull, best_bear, (best_win_rate if best_win_rate >= 0 else 0.0), best_active, win_rate_matrix, text_matrix


def render_threshold_heatmap(
    win_matrix: pd.DataFrame,
    text_matrix: pd.DataFrame,
    best_bull: int,
    best_bear: int,
):
    annot_text = text_matrix.copy()
    if best_bull in annot_text.index and best_bear in annot_text.columns:
        orig = annot_text.loc[best_bull, best_bear]
        annot_text.loc[best_bull, best_bear] = f"<b>⭐ {orig}</b><br><b>[最佳點]</b>"

    x_labels = [f"空方 {c}%" for c in win_matrix.columns]
    y_labels = [f"多方 {r}%" for r in win_matrix.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=win_matrix.values,
            x=x_labels,
            y=y_labels,
            text=annot_text.values,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#f8fafc"},
            colorscale=[
                [0.0, "#0f172a"],
                [0.4, "#1e293b"],
                [0.55, "#3b82f6"],
                [0.75, "#10b981"],
                [1.0, "#f43f5e"],
            ],
            colorbar=dict(
                title=dict(text="<b>勝率 (%)</b>", font=dict(color="#f8fafc")),
                ticksuffix="%",
                tickfont=dict(color="#94a3b8"),
            ),
            hoverongaps=False,
            zmin=30,
            zmax=100,
        )
    )

    fig.update_layout(
        title="<b>多空發動門檻雙變數網格搜尋熱力圖 (開盤進場基準 | 步長 5%)</b>",
        xaxis=dict(title="<b>空方防禦發動信心門檻 (P(Up) ≤ 門檻)</b>", gridcolor="#334155"),
        yaxis=dict(title="<b>多方進攻發動信心門檻 (P(Up) ≥ 門檻)</b>", gridcolor="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# -----------------------------------------------------------------------------
# 4. 跳空幅度 vs. 盤中漲跌散布圖與 OLS 趨勢線
# -----------------------------------------------------------------------------
def render_gap_scatter(
    dates: list,
    gaps: list,
    intraday_rets: list,
    filled_status: list,
):
    clean_data = [
        (d, g, r, f)
        for d, g, r, f in zip(dates, gaps, intraday_rets, filled_status)
        if not np.isnan(g) and not np.isnan(r)
    ]

    if len(clean_data) < 3:
        return go.Figure()

    d_list = [item[0] for item in clean_data]
    g_arr = np.array([item[1] for item in clean_data])
    r_arr = np.array([item[2] for item in clean_data])
    f_list = [item[3] for item in clean_data]

    slope, intercept = np.polyfit(g_arr, r_arr, 1)
    correlation = np.corrcoef(g_arr, r_arr)[0, 1]
    r_squared = correlation**2

    x_line = np.linspace(min(g_arr) - 0.2, max(g_arr) + 0.2, 50)
    y_line = slope * x_line + intercept

    point_colors = ["#f43f5e" if r >= 0 else "#10b981" for r in r_arr]
    hover_texts = [
        f"<b>交易日:</b> {d}<br>"
        f"<b>開盤跳空 (Gap):</b> {g:+.2f}%<br>"
        f"<b>盤中漲跌 (O to C):</b> {r:+.2f}%<br>"
        f"<b>缺口回補:</b> {'✅ 完全回補' if fl else '❌ 未回補'}"
        for d, g, r, fl in zip(d_list, g_arr, r_arr, f_list)
    ]

    fig = go.Figure()

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

    fig.add_hline(y=0, line_width=1, line_color="#64748b", line_dash="solid")
    fig.add_vline(x=0, line_width=1, line_color="#64748b", line_dash="solid")

    fig.update_layout(
        title=f"<b>跳空幅度 vs. 盤中開盤至收盤漲跌散布圖 (趨勢特性: {trend_desc})</b>",
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


# -----------------------------------------------------------------------------
# 5. 儀表板繪圖元件 (Plotly)
# -----------------------------------------------------------------------------
def render_gauge(prob_up: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_up * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "<b>加權指數開盤進場上漲勝率 P(Up)</b>",
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


def render_waterfall(contributions: dict):
    label_map = {
        "foreign_net": "外資現貨買賣超",
        "tx_foreign_oi": "外資期指淨留倉",
        "fx_inter": "外資期現共振交互項",
        "sox_ret": "前日費半指數 (SOX)",
        "tsm_ret": "前日台積電 ADR (TSM)",
        "vix_close": "前日恐慌指數 (VIX)",
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
        title="<b>特徵因子對開盤進場 Log-Odds 邊際貢獻 (Beta × Z-Score)</b>",
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
# 6. Streamlit 介面佈局與資料載入
# -----------------------------------------------------------------------------
pipeline = MarketDataPipeline()
engine = TaiexLogisticEngine()

st.sidebar.header("🕹️ 特徵監控與模擬器")

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

model_mode = st.sidebar.radio(
    "權重計算模式",
    ["固定專家權重", "期現交互全動態滾動配適 (Interaction MLE)"],
    index=1,
    help="選擇是否將外資現貨、外資期指、外資期現交互共振項與美股等納入動態 MLE 最佳化",
)

auto_tune_train_days = False
if model_mode == "期現交互全動態滾動配適 (Interaction MLE)":
    auto_tune_train_days = st.sidebar.checkbox(
        "⚡ 自動搜尋最佳訓練天數 (步長 10 天)",
        value=True,
        help="自動網格搜尋 60 ~ 750 營業日，鎖定使含期現交互作用模型準確率最高的天數",
    )
    if not auto_tune_train_days:
        train_days_setting = st.sidebar.slider(
            "📅 動態訓練天數 (營業日)",
            min_value=60,
            max_value=750,
            value=st.session_state.get("optimized_train_days", 250),
            step=10,
            help="手動設定以評估日前多少個營業日的歷史數據進行 Logistic MLE 配適",
        )
    else:
        train_days_setting = st.session_state.get("optimized_train_days", 250)
        st.sidebar.caption(f"🎯 當前鎖定之最佳天數：**{train_days_setting} 營業日**")
else:
    train_days_setting = 250

selected_target_date = st.sidebar.date_input(
    "選擇目標預測交易日 (T)", datetime.date.today()
)

is_holiday, holiday_reason = check_is_taiwan_holiday(selected_target_date)
if is_holiday:
    st.sidebar.error(f"⚠️ 所選日期為 **{holiday_reason}**，台灣股市未開盤！")
    st.error(f"🛑 **目標交易日 ({selected_target_date}) 為【{holiday_reason}】**。非交易日市場休市，無需計算盤前預測機率，請於左側選單選擇一般交易日。")
    st.stop()

# 嚴格取得目標日前一個開盤交易日（例如 2026/10/27 前一日自動避開 10/26 補假、10/24-25 週末、10/23 颱風假，直接為 10/22）
base_prev_date = pipeline.get_previous_trading_day(selected_target_date)

btn_col1, btn_col2 = st.sidebar.columns([3, 2])
with btn_col1:
    fetch_btn = st.button("🚀 擷取前日數據", use_container_width=True)
with btn_col2:
    reset_btn = st.button("↺ 還原特徵", use_container_width=True)

st.sidebar.caption(
    f"📌 **執行邏輯說明**：目標預測日為 **{selected_target_date}**。模型利用盤前已知（**{base_prev_date}**）外資期現貨、期現共振交互項及美股指標，動態求解開盤進場勝率。"
)


def safe_clamp(val, min_v, max_v, default_v=0.0):
    try:
        f_val = float(val)
        if np.isnan(f_val):
            return float(default_v)
        return float(max(min(f_val, max_v), min_v))
    except Exception:
        return float(default_v)


def reset_slider_states(f_net, t_net, tx_oi, sox_r, tsm_r, vix_c):
    st.session_state["slider_f_net"] = safe_clamp(f_net, -1200.0, 1200.0, 0.0)
    st.session_state["slider_t_net"] = safe_clamp(t_net, -300.0, 300.0, 0.0)
    st.session_state["slider_tx_oi"] = safe_clamp(tx_oi, -80000.0, 50000.0, -12000.0)
    st.session_state["slider_sox_r"] = safe_clamp(sox_r, -15.0, 15.0, 0.0)
    st.session_state["slider_tsm_r"] = safe_clamp(tsm_r, -15.0, 15.0, 0.0)
    st.session_state["slider_vix_c"] = safe_clamp(vix_c, 5.0, 80.0, 18.0)


need_fetch = (
    "cached_data" not in st.session_state
    or fetch_btn
    or st.session_state.get("last_target_date") != selected_target_date
)

if need_fetch:
    with st.spinner(f"正在擷取有效交易日數據 (鎖定基準前日: {base_prev_date})、歷史真實籌碼與樣本..."):
        # 嚴格以 base_prev_date 為上限向前回溯，絕不抓取 >= selected_target_date 的盤後資料
        twse_df, twse_actual_date, twse_msg = pipeline.fetch_twse_institutional_with_fallback(base_prev_date)
        taifex_df, taifex_actual_date, taifex_msg = pipeline.fetch_taifex_oi_with_fallback(base_prev_date)
        us_df, us_msg = pipeline.fetch_us_market(base_prev_date, days_back=15)
        hist_df = pipeline.fetch_historical_training_and_verification_data(
            selected_target_date, max_train_days=750
        )

        hist_chip_start = base_prev_date - datetime.timedelta(days=35)
        hist_chip_end = base_prev_date
        hist_twse_df = pipeline.fetch_twse_historical_range(hist_chip_start, hist_chip_end)
        hist_taifex_df = pipeline.fetch_taifex_historical_range(hist_chip_start, hist_chip_end)

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
        sox_r = float(latest_us["sox_ret"]) if latest_us is not None else 0.0
        tsm_r = float(latest_us["tsm_ret"]) if latest_us is not None else 0.0
        vix_c = float(latest_us["vix_close"]) if latest_us is not None else 18.0

        us_actual_date = latest_us.name.date() if latest_us is not None else base_prev_date

        st.session_state["cached_data"] = {
            "target_date": selected_target_date,
            "base_feature_date": base_prev_date,
            "twse_actual_date": twse_actual_date,
            "taifex_actual_date": taifex_actual_date,
            "us_actual_date": us_actual_date,
            "twse_msg": twse_msg,
            "taifex_msg": taifex_msg,
            "us_msg": us_msg,
            "f_net": f_net,
            "t_net": t_net,
            "tx_oi": tx_oi,
            "sox_r": sox_r,
            "tsm_r": tsm_r,
            "vix_c": vix_c,
            "hist_df": hist_df,
            "hist_twse_df": hist_twse_df,
            "hist_taifex_df": hist_taifex_df,
        }
        st.session_state["last_target_date"] = selected_target_date
        reset_slider_states(f_net, t_net, tx_oi, sox_r, tsm_r, vix_c)

elif reset_btn:
    c = st.session_state["cached_data"]
    reset_slider_states(
        c["f_net"], c["t_net"], c["tx_oi"], c["sox_r"], c["tsm_r"], c["vix_c"]
    )
    st.toast("已還原至前日真實特徵數值！", icon="🔄")

data = st.session_state["cached_data"]
target_dt = pd.to_datetime(data["target_date"])

# -----------------------------------------------------------------------------
# 期現交互全特徵動態 MLE 最佳化排程
# -----------------------------------------------------------------------------
if model_mode == "期現交互全動態滾動配適 (Interaction MLE)" and not data["hist_df"].empty:
    if auto_tune_train_days:
        best_days, best_acc = engine.find_best_train_days(
            data["hist_df"],
            target_dt,
            min_days=60,
            max_days=750,
            step=10,
            hist_twse=data.get("hist_twse_df"),
            hist_taifex=data.get("hist_taifex_df"),
        )
        st.session_state["optimized_train_days"] = best_days
        train_days_setting = best_days

    engine.fit_rolling_model(
        data["hist_df"],
        target_dt,
        train_days=train_days_setting,
        hist_twse=data.get("hist_twse_df"),
        hist_taifex=data.get("hist_taifex_df"),
    )
else:
    engine.reset_to_static()

if "slider_f_net" not in st.session_state:
    reset_slider_states(
        data["f_net"],
        data["t_net"],
        data["tx_oi"],
        data["sox_r"],
        data["tsm_r"],
        data["vix_c"],
    )

st.sidebar.caption("💡 可微調各項因子數值進行即時敏感度測試：")

sim_foreign_net = st.sidebar.slider(
    "前日外資現貨買賣超 (億元)",
    min_value=-1200.0,
    max_value=1200.0,
    value=st.session_state["slider_f_net"],
    step=1.0,
    key="slider_f_net",
)
sim_trust_net = st.sidebar.slider(
    "前日投信現貨買賣超 (億元) [僅供參考]",
    min_value=-300.0,
    max_value=300.0,
    value=st.session_state["slider_t_net"],
    step=1.0,
    key="slider_t_net",
    help="已從 Logistic 迴歸中排除，僅作為盤面輔助觀察",
)
sim_tx_oi = st.sidebar.slider(
    "前日外資期貨淨留倉 (口)",
    min_value=-80000.0,
    max_value=50000.0,
    value=st.session_state["slider_tx_oi"],
    step=500.0,
    key="slider_tx_oi",
)
sim_sox_ret = st.sidebar.slider(
    "前日費半指數漲跌幅 (%)",
    min_value=-15.0,
    max_value=15.0,
    value=st.session_state["slider_sox_r"],
    step=0.1,
    key="slider_sox_r",
)
sim_tsm_ret = st.sidebar.slider(
    "前日台積電 ADR 漲跌幅 (%)",
    min_value=-15.0,
    max_value=15.0,
    value=st.session_state["slider_tsm_r"],
    step=0.1,
    key="slider_tsm_r",
)
sim_vix_close = st.sidebar.slider(
    "前日 VIX 恐慌指數",
    min_value=5.0,
    max_value=80.0,
    value=st.session_state["slider_vix_c"],
    step=0.1,
    key="slider_vix_c",
)

mu_f = engine.active_scaler.get("foreign_net", {}).get("mu", 0.0)
sig_f = engine.active_scaler.get("foreign_net", {}).get("sigma", 1.0)
mu_tx = engine.active_scaler.get("tx_foreign_oi", {}).get("mu", 0.0)
sig_tx = engine.active_scaler.get("tx_foreign_oi", {}).get("sigma", 1.0)
current_z_f = (sim_foreign_net - mu_f) / (sig_f if sig_f > 0 else 1.0)
current_z_tx = (sim_tx_oi - mu_tx) / (sig_tx if sig_tx > 0 else 1.0)
current_fx_inter = current_z_f * current_z_tx

current_features = {
    "foreign_net": sim_foreign_net,
    "tx_foreign_oi": sim_tx_oi,
    "fx_inter": current_fx_inter,
    "sox_ret": sim_sox_ret,
    "tsm_ret": sim_tsm_ret,
    "vix_close": sim_vix_close,
}

prob_up, contributions = engine.predict(current_features)

# -----------------------------------------------------------------------------
# 7. 主視覺顯示 (Dashboard View)
# -----------------------------------------------------------------------------
st.title("🎯 TAIEX 台灣加權指數開盤進場漲跌機率監控儀表")

opt_tag = " (⚡期現交互自動尋優)" if auto_tune_train_days else ""
m_desc = (
    f"期現交互動態 MLE 配適 (視窗: {train_days_setting} 營業日{opt_tag}, 實際樣本: {engine.fit_sample_size} 天, 訓練準確率: {engine.fit_accuracy:.1f}%)"
    if engine.fit_accuracy is not None
    else "固定專家權重"
)

fallback_warnings = []
if data["twse_actual_date"] != data["base_feature_date"]:
    fallback_warnings.append(f"TWSE 三大法人數據向前校正至 `{data['twse_actual_date']}`（原基準日 {data['base_feature_date']} 未開盤/無資料）")
if data["taifex_actual_date"] != data["base_feature_date"]:
    fallback_warnings.append(f"TAIFEX 期貨留倉數據向前校正至 `{data['taifex_actual_date']}`（原基準日 {data['base_feature_date']} 未開盤/無資料）")

if fallback_warnings:
    st.info("ℹ️ **特徵日期因休市/颱風假自動向前校正**：\n- " + "\n- ".join(fallback_warnings))

st.markdown(
    f"""
> **系統運行狀態與模型參數因果檢驗**  
> • **目標預測交易日 (T)**：`{data['target_date']}` | **模型權重模式**：`{m_desc}`  
> • **執行基準**：**開盤價 (Open) 進場 ➔ 收盤價 (Close) 結算**（已加入外資期現貨非線性交互作用項）  
> • **證券交易所 (TWSE)**：外資現貨買賣超 (擷取日期: `{data['twse_actual_date']}` | `狀態: {data['twse_msg']}`)  
> • **期貨交易所 (TAIFEX)**：臺股期貨外資留倉 (擷取日期: `{data['taifex_actual_date']}` | `狀態: {data['taifex_msg']}`)  
> • **美股行情 (yfinance)**：費半、TSM ADR、VIX (擷取日期: `{data['us_actual_date']}` | `狀態: {data['us_msg']}`)
"""
)

st.markdown("---")

col_gauge, col_waterfall = st.columns([5, 5])

with col_gauge:
    st.plotly_chart(render_gauge(prob_up), use_container_width=True)
    if prob_up >= 0.55:
        st.markdown(
            f"<h3 style='text-align: center;' class='text-bullish'>預測信號：強烈看多 (開盤進場上漲勝率 {prob_up*100:.1f}%)</h3>",
            unsafe_allow_html=True,
        )
    elif prob_up <= 0.45:
        st.markdown(
            f"<h3 style='text-align: center;' class='text-bearish'>預測信號：偏空防禦 (開盤進場上漲勝率 {prob_up*100:.1f}%)</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h3 style='text-align: center; color: #94a3b8;'>預測信號：區間震盪整理 (開盤進場上漲勝率 {prob_up*100:.1f}%)</h3>",
            unsafe_allow_html=True,
        )

with col_waterfall:
    st.plotly_chart(render_waterfall(contributions), use_container_width=True)

with st.expander("🔍 檢視當前目標日 (T) 6 大特徵之動態最佳化 Beta 權重與常模 (Beta & Scaler)"):
    w_df = pd.DataFrame(
        [
            {
                "特徵因子": k,
                "最佳化 Beta 權重": round(engine.active_beta.get(k, 0.0), 4),
                "常模均值 (μ)": round(engine.active_scaler.get(k, {}).get("mu", 0.0), 3),
                "常模標準差 (σ)": round(engine.active_scaler.get(k, {}).get("sigma", 1.0), 3),
            }
            for k in engine.ALL_FEATURES
        ]
    )
    st.dataframe(w_df, hide_index=True, use_container_width=True)

st.subheader("📊 盤前已知核心特徵數值一覽 (含外資期現共振項)")
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.metric(
        "外資現貨買賣超",
        f"{sim_foreign_net:+.1f} 億",
        delta=f"Beta: {engine.active_beta.get('foreign_net', 0.0):+.2f}",
        delta_color="normal" if sim_foreign_net >= 0 else "inverse",
    )
    st.caption(f"來源: TWSE ({data['twse_actual_date']})")

with m2:
    oi_status = "淨空單" if sim_tx_oi < 0 else "淨多單"
    st.metric(
        f"外資期貨 ({oi_status})",
        f"{abs(int(sim_tx_oi)):,} 口",
        delta=f"Beta: {engine.active_beta.get('tx_foreign_oi', 0.0):+.2f}",
        delta_color="normal" if sim_tx_oi >= -10000 else "inverse",
    )
    st.caption(f"來源: TAIFEX ({data['taifex_actual_date']})")

with m3:
    inter_status = "同向共振" if current_fx_inter > 0.2 else ("反向對沖" if current_fx_inter < -0.2 else "中性平衡")
    st.metric(
        "期現交互共振值",
        f"{current_fx_inter:+.2f}",
        delta=f"{inter_status} (Beta: {engine.active_beta.get('fx_inter', 0.0):+.2f})",
        delta_color="normal" if current_fx_inter >= 0 else "inverse",
    )
    st.caption("Z(現貨) × Z(期貨)")

with m4:
    st.metric(
        "美股費半 (SOX)",
        f"{sim_sox_ret:+.2f}%",
        delta=f"Beta: {engine.active_beta.get('sox_ret', 0.0):+.2f}",
        delta_color="normal" if sim_sox_ret >= 0 else "inverse",
    )
    st.caption(f"來源: yfinance ({data['us_actual_date']})")

with m5:
    st.metric(
        "台積電 ADR",
        f"{sim_tsm_ret:+.2f}%",
        delta=f"Beta: {engine.active_beta.get('tsm_ret', 0.0):+.2f}",
        delta_color="normal" if sim_tsm_ret >= 0 else "inverse",
    )
    st.caption(f"來源: yfinance ({data['us_actual_date']})")

with m6:
    st.metric(
        "美股 VIX 指數",
        f"{sim_vix_close:.1f}",
        delta=f"Beta: {engine.active_beta.get('vix_close', 0.0):+.2f}",
        delta_color="inverse" if sim_vix_close >= 20 else "normal",
    )
    st.caption(f"來源: yfinance ({data['us_actual_date']})")

# -----------------------------------------------------------------------------
# 8. 過往 10 日純樣本外 (Interaction OOS) 疊加核驗與策略損益
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 過往 10 日純樣本外 (Interaction OOS | 開盤進場 + 期現交互共振) 疊加核驗")

hist_df = data["hist_df"]
hist_twse = data.get("hist_twse_df", pd.DataFrame())
hist_taifex = data.get("hist_taifex_df", pd.DataFrame())

if not hist_df.empty:
    valid_hist = hist_df[hist_df.index <= target_dt]

    if len(valid_hist) >= 10:
        ten_days_df = valid_hist.tail(10).copy()

        calc_probs = []
        is_dynamic = (model_mode == "期現交互全動態滾動配適 (Interaction MLE)")

        for i in range(len(ten_days_df)):
            current_date_idx = ten_days_df.index[i]
            if current_date_idx == target_dt:
                p, _ = engine.predict(current_features)
            else:
                prev_row_loc = hist_df.index.get_loc(current_date_idx) - 1
                if prev_row_loc >= 0:
                    prev_r = hist_df.iloc[prev_row_loc]
                    prev_dt = hist_df.index[prev_row_loc]

                    f_val = 5.0
                    if not hist_twse.empty:
                        matched_twse = hist_twse[hist_twse.index <= prev_dt]
                        if not matched_twse.empty:
                            last_twse_row = matched_twse.iloc[-1]
                            f_val = float(last_twse_row.get("foreign_net", 5.0))

                    tx_val = -12000.0
                    if not hist_taifex.empty:
                        matched_taifex = hist_taifex[hist_taifex.index <= prev_dt]
                        if not matched_taifex.empty:
                            last_taifex_row = matched_taifex.iloc[-1]
                            tx_val = float(last_taifex_row.get("tx_foreign_oi", -12000.0))

                    hist_feat = {
                        "foreign_net": f_val,
                        "tx_foreign_oi": tx_val,
                        "sox_ret": float(prev_r["sox_ret"]),
                        "tsm_ret": float(prev_r["tsm_ret"]),
                        "vix_close": float(prev_r["vix_close"]),
                    }

                    p = engine.predict_out_of_sample_step(
                        hist_df=hist_df,
                        t_day=current_date_idx,
                        features_dict=hist_feat,
                        train_days=train_days_setting,
                        use_dynamic=is_dynamic,
                        hist_twse=hist_twse,
                        hist_taifex=hist_taifex,
                    )
                else:
                    p = 0.5
            calc_probs.append(round(p, 4))

        ten_days_df["Prob_Up"] = calc_probs
        date_labels = ten_days_df.index.strftime("%Y-%m-%d").tolist()
        probs_pct = [p * 100 for p in ten_days_df["Prob_Up"]]

        returns_list = ten_days_df["TWII_Open_to_Close"].tolist()
        close_to_close_list = ten_days_df["TWII_Close_to_Close"].tolist()
        open_prices = ten_days_df["TWII_Open"].tolist()
        close_prices = ten_days_df["TWII_Close"].tolist()
        gaps_list = ten_days_df["gap_ret"].tolist()
        filled_list = ten_days_df["gap_filled"].tolist()

        valid_gaps = [g for g in gaps_list if not np.isnan(g)]
        valid_fills = [f for f in filled_list if not np.isnan(f)]

        avg_gap = np.mean(valid_gaps) if valid_gaps else 0.0
        max_gap_up = max([g for g in valid_gaps if g > 0], default=0.0)
        max_gap_down = min([g for g in valid_gaps if g < 0], default=0.0)
        gap_fill_rate = (sum(valid_fills) / len(valid_fills) * 100.0) if valid_fills else 0.0

        cont_count = 0
        valid_pairs = 0
        for g, r in zip(gaps_list, returns_list):
            if not np.isnan(g) and not np.isnan(r) and abs(g) >= 0.05:
                valid_pairs += 1
                if (g > 0 and r > 0) or (g < 0 and r < 0):
                    cont_count += 1
        cont_rate = (cont_count / valid_pairs * 100.0) if valid_pairs > 0 else 0.0

        st.markdown("##### 📌 過往 10 日開盤跳空統計特徵看板 (Gap Statistics)")
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

        with st.expander("📊 檢視跳空幅度 vs. 盤中漲跌散布圖與 OLS 迴歸線 (Gap Scatter Plot)", expanded=False):
            st.plotly_chart(
                render_gap_scatter(date_labels, gaps_list, returns_list, filled_list),
                use_container_width=True,
            )

        st.markdown("---")

        best_bull_val, best_bear_val, opt_win_rate, opt_signals, win_matrix, text_matrix = optimize_thresholds_with_matrix(
            probs_pct, returns_list, bull_range=(50, 75), bear_range=(25, 50), step=5
        )

        ctrl_col0, ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.5, 3, 3, 2.5])
        with ctrl_col0:
            auto_tune_thresholds = st.checkbox(
                "⚡ 自動鎖定最佳勝率門檻 (變動 5%)",
                value=True,
                help="以 5% 為步長進行雙變數網格搜尋，自動找出使開盤進場勝率達到最高之門檻組合",
            )
        with ctrl_col1:
            if auto_tune_thresholds:
                threshold_bull = best_bull_val
                st.metric("🎯 多方發動信心門檻", f"{threshold_bull}%", delta="自動尋優鎖定")
            else:
                threshold_bull = st.slider("🎯 多方發動信心門檻 (%)", 50, 75, best_bull_val, step=5)
        with ctrl_col2:
            if auto_tune_thresholds:
                threshold_bear = best_bear_val
                st.metric("🛡️ 空方防禦信心門檻", f"{threshold_bear}%", delta="自動尋優鎖定")
            else:
                threshold_bear = st.slider("🛡️ 空方防禦信心門檻 (%)", 25, 50, best_bear_val, step=5)
        with ctrl_col3:
            cost_rate_pct = st.number_input(
                "💸 單趟摩擦成本率 (%)",
                min_value=0.00,
                max_value=0.20,
                value=0.03,
                step=0.005,
                format="%.3f",
                help="包含期交稅、手續費與進出滑價折算率（預設 0.03%）",
            )

        with st.expander("🔥 檢視多空信心門檻雙變數網格搜尋熱力圖 (Grid Search Heatmap)", expanded=False):
            st.plotly_chart(
                render_threshold_heatmap(win_matrix, text_matrix, best_bull_val, best_bear_val),
                use_container_width=True,
            )

        st.caption(
            f"💡 **真實執行因果檢定**：以預測當日 **開盤價進場、收盤價結算**（納入外資期現共振交互項）。當勝率 $\\ge {threshold_bull}\\%$ 做多 (+1)，$\\le {threshold_bear}\\%$ 放空 (-1)；其餘**中性觀望 (0 部位)**。每次部位變動扣除 **{cost_rate_pct:.3f}%** 摩擦成本。"
        )

        signal_types = []
        positions = []
        gross_returns = []
        net_returns = []
        friction_costs = []
        validation_status = []
        bar_colors = []
        bar_opacities = []

        active_signals = 0
        correct_signals = 0
        prev_pos = 0

        for p, r in zip(probs_pct, returns_list):
            is_pending = np.isnan(r)

            if p >= threshold_bull:
                sig = "📈 多方訊號"
                pos = 1
                gross_ret = r if not is_pending else 0.0

                if is_pending:
                    status = "⏳ 待結算"
                elif r > 0:
                    status = "✅ 吻合 (開盤進場看多獲利)"
                    correct_signals += 1
                else:
                    status = "❌ 不吻合 (開盤進場看多虧損)"

                if not is_pending:
                    active_signals += 1
                bar_colors.append(
                    "#f43f5e" if (not is_pending and r >= 0) else "#10b981"
                )
                bar_opacities.append(0.85)

            elif p <= threshold_bear:
                sig = "📉 空方訊號"
                pos = -1
                gross_ret = -r if not is_pending else 0.0

                if is_pending:
                    status = "⏳ 待結算"
                elif r < 0:
                    status = "✅ 吻合 (開盤進場放空獲利)"
                    correct_signals += 1
                else:
                    status = "❌ 不吻合 (開盤進場放空虧損)"

                if not is_pending:
                    active_signals += 1
                bar_colors.append(
                    "#f43f5e" if (not is_pending and r >= 0) else "#10b981"
                )
                bar_opacities.append(0.85)

            else:
                sig = "⚪ 觀望區間"
                pos = 0
                gross_ret = 0.0
                status = "濾除不計 (中性觀望)"
                bar_colors.append("rgba(148, 163, 184, 0.4)")
                bar_opacities.append(0.3)

            trade_turnover = abs(pos - prev_pos)
            cost = trade_turnover * cost_rate_pct if not is_pending else 0.0
            net_ret = gross_ret - cost if not is_pending else 0.0

            signal_types.append(sig)
            positions.append(pos)
            gross_returns.append(gross_ret)
            friction_costs.append(cost)
            net_returns.append(net_ret)
            validation_status.append(status)

            prev_pos = pos

        gross_cum = [0.0]
        net_cum = [0.0]
        bnh_cum = [0.0]

        for g_r, n_r, b_r in zip(gross_returns, net_returns, returns_list):
            if np.isnan(b_r):
                gross_cum.append(gross_cum[-1])
                net_cum.append(net_cum[-1])
                bnh_cum.append(bnh_cum[-1])
            else:
                gross_cum.append(gross_cum[-1] + g_r)
                net_cum.append(net_cum[-1] + n_r)
                bnh_cum.append(bnh_cum[-1] + b_r)

        gross_equity = gross_cum[1:]
        net_equity = net_cum[1:]
        bnh_equity = bnh_cum[1:]

        net_series = pd.Series(net_equity)
        rolling_max = np.maximum.accumulate(net_series)
        drawdowns = net_series - rolling_max
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0.0

        total_net_return = net_equity[-1]
        total_gross_return = gross_equity[-1]
        total_cost_spent = sum(friction_costs)
        total_bnh_return = bnh_equity[-1]
        win_rate = (
            (correct_signals / active_signals * 100.0)
            if active_signals > 0
            else 0.0
        )

        valid_net_returns = [nr for nr, r in zip(net_returns, returns_list) if not np.isnan(r)]
        sample_n = len(valid_net_returns)
        rf_daily = (1.5 / 100.0) / 252.0

        if sample_n > 1:
            ret_series = pd.Series(valid_net_returns) / 100.0
            ret_std = ret_series.std()
            excess_mean = (ret_series - rf_daily).mean()
            sharpe_ratio = (excess_mean / ret_std * np.sqrt(252)) if ret_std > 0 else 0.0

            annualized_net_return = (total_net_return / sample_n) * 252.0
            calmar_ratio = (annualized_net_return / abs(max_drawdown)) if abs(max_drawdown) > 0 else (annualized_net_return if annualized_net_return > 0 else 0.0)

            gains = sum([r for r in valid_net_returns if r > 0])
            losses = abs(sum([r for r in valid_net_returns if r < 0]))
            profit_factor = (gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)
        else:
            sharpe_ratio = 0.0
            calmar_ratio = 0.0
            profit_factor = 0.0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(
                "開盤進場高信心勝率",
                f"{win_rate:.1f}%" if active_signals > 0 else "無觸發",
                delta=f"有效 {active_signals} 天 / 命中 {correct_signals} 天",
                delta_color="normal" if win_rate >= 60 else "inverse",
            )
        with k2:
            st.metric(
                "策略累積淨報酬 (扣成本)",
                f"{total_net_return:+.2f}%",
                delta=f"{total_net_return - total_bnh_return:+.2f}% 超額 (vs 開盤買入持有)",
                delta_color="normal" if total_net_return >= 0 else "inverse",
            )
        with k3:
            st.metric(
                "同期開盤買入持有",
                f"{total_bnh_return:+.2f}%",
                delta=f"{total_bnh_return:+.2f}%",
                delta_color="normal" if total_bnh_return >= 0 else "inverse",
            )
        with k4:
            st.metric(
                "累計摩擦損耗",
                f"{total_cost_spent:.3f}%",
                delta=f"毛報酬: {total_gross_return:+.2f}%",
                delta_color="off",
            )

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric(
                "最大淨值回撤 (MDD)",
                f"{max_drawdown:.2f}%",
                delta="安全防禦評級" if abs(max_drawdown) < 2.0 else "需關注回撤",
                delta_color="inverse",
            )
        with r2:
            st.metric(
                "年化夏普比率 (Sharpe)",
                f"{sharpe_ratio:.2f}",
                delta="報酬波動比 (Rf=1.5%)",
                delta_color="normal" if sharpe_ratio >= 1.0 else "off",
            )
        with r3:
            st.metric(
                "卡瑪比率 (Calmar)",
                f"{calmar_ratio:.2f}",
                delta="年化淨報酬 / |MDD|",
                delta_color="normal" if calmar_ratio >= 2.0 else "off",
            )
        with r4:
            pf_str = f"{profit_factor:.2f}" if profit_factor < 90 else "∞ (無虧損)"
            st.metric(
                "策略獲利因子 (Profit Factor)",
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
                name="前日樣本外預測勝率 P(Up)",
                line=dict(color="#f59e0b", width=3),
                text=[f"{p:.1f}%" for p in probs_pct],
                textposition="top center",
                yaxis="y1",
            )
        )

        fig_verify.add_trace(
            go.Bar(
                x=date_labels,
                y=returns_list,
                name="當日實際【開盤至收盤】漲跌幅 (%)",
                marker=dict(color=bar_colors, opacity=bar_opacities),
                text=[
                    f"{r:+.2f}%" if not np.isnan(r) else "尚未結算"
                    for r in returns_list
                ],
                textposition="auto",
                yaxis="y2",
            )
        )

        fig_verify.add_hline(
            y=threshold_bull,
            line_dash="dot",
            line_color="rgba(244, 63, 94, 0.7)",
            annotation_text=f"多方門檻 ({threshold_bull}%)",
            annotation_position="bottom right",
            annotation_font_color="#f43f5e",
        )
        fig_verify.add_hline(
            y=threshold_bear,
            line_dash="dot",
            line_color="rgba(16, 185, 129, 0.7)",
            annotation_text=f"空方門檻 ({threshold_bear}%)",
            annotation_position="top right",
            annotation_font_color="#10b981",
        )

        fig_verify.update_layout(
            title="<b>過往 10 日純樣本外勝率 vs 實際【開盤至收盤】執行漲跌幅疊加比對 (期現交互全動態最佳化)</b>",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            xaxis=dict(title="目標預測交易日 (T)", gridcolor="#334155"),
            yaxis=dict(
                title="前日純樣本外勝率 P(Up)",
                range=[0, 105],
                gridcolor="#334155",
                ticksuffix="%",
            ),
            yaxis2=dict(
                title="開盤進場實際漲跌幅 (%)",
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
                name="開盤進場策略淨報酬 (扣除摩擦成本)",
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
                name="策略毛報酬 (未扣摩擦成本)",
                line=dict(color="#818cf8", width=1.8, dash="dot"),
            )
        )

        fig_equity.add_trace(
            go.Scatter(
                x=date_labels,
                y=bnh_equity,
                mode="lines+markers",
                name="開盤買入持有基準 (Open Buy & Hold)",
                line=dict(color="#94a3b8", width=2, dash="dash"),
            )
        )

        fig_equity.add_hline(
            y=0, line_width=1, line_color="#64748b", line_dash="solid"
        )

        fig_equity.update_layout(
            title=f"<b>開盤進場累積模擬損益曲線 (Net Equity Curve) | Sharpe: {sharpe_ratio:.2f} | Calmar: {calmar_ratio:.2f} | MDD: {max_drawdown:.2f}%</b>",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            xaxis=dict(title="目標交易日 (T)", gridcolor="#334155"),
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

        pos_labels = [
            "🟢 做多 (+1)" if p == 1 else ("🔴 放空 (-1)" if p == -1 else "⚪ 空手 (0)")
            for p in positions
        ]
        table_df = pd.DataFrame(
            {
                "目標交易日 (T)": date_labels,
                "開盤進場價": [f"{op:,.1f}" if not np.isnan(op) else "-" for op in open_prices],
                "收盤結算價": [f"{cp:,.1f}" if not np.isnan(cp) else "-" for cp in close_prices],
                "開盤跳空 (Gap)": [f"{gp:+.2f}%" if not np.isnan(gp) else "-" for gp in gaps_list],
                "缺口盤中回補": ["✅ 已回補" if fl else "❌ 未回補" for fl in filled_list],
                "開盤至收盤漲跌": [
                    f"{r:+.2f}%" if not np.isnan(r) else "尚未結算"
                    for r in returns_list
                ],
                "傳統收盤漲跌(參考)": [
                    f"{c2c:+.2f}%" if not np.isnan(c2c) else "尚未結算"
                    for c2c in close_to_close_list
                ],
                "當日樣本外勝率": [f"{p:.1f}%" for p in probs_pct],
                "訊號類型": signal_types,
                "策略部位": pos_labels,
                "單日毛損益": [
                    f"{gr:+.2f}%" if not np.isnan(r) else "尚未結算"
                    for gr, r in zip(gross_returns, returns_list)
                ],
                "摩擦成本": [
                    f"-{c:.3f}%" if c > 0 else "0.000%"
                    for c in friction_costs
                ],
                "單日淨損益": [
                    f"{nr:+.2f}%" if not np.isnan(r) else "尚未結算"
                    for nr, r in zip(net_returns, returns_list)
                ],
                "執行核驗狀態": validation_status,
            }
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.warning("歷史交易日資料不足 10 天，無法繪製完整比對圖。")
else:
    st.warning("加權指數歷史數據擷取中或暫時無法取得。")