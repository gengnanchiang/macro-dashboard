# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 21:58:46 2026

@author: James Chiang
"""

import datetime
import io
import time
from typing import Dict, Tuple
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
# 0. 多使用者權限與身分驗證層 (Multi-user Authentication Gate)
# -----------------------------------------------------------------------------
def check_password() -> bool:
    """檢查多個使用者的帳號密碼，未登入時以 st.stop() 阻斷後續資料載入"""
    if st.session_state.get("authenticated", False):
        return True

    # 1. 預設的多組帳號密碼（帳號: 密碼）
    default_users = {
        "admin": "taiex888",
        "analyst01": "quant2026",
        "manager": "fund999",
        "guest": "read123",
    }

    # 2. 優先讀取 st.secrets 中的多使用者設定（若雲端有配置）
    if "credentials" in st.secrets and "users" in st.secrets["credentials"]:
        valid_users = dict(st.secrets["credentials"]["users"])
    else:
        valid_users = default_users

    # 3. 渲染登入介面
    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        st.markdown("### 🔒 TAIEX 量化監控終端使用者登入")
        st.caption("請輸入授權之帳號與密碼以解鎖即時模型與數據權限。")
        with st.form("login_form"):
            input_user = st.text_input("使用者帳號 (Username)").strip()
            input_pwd = st.text_input("存取密碼 (Password)", type="password").strip()
            submit = st.form_submit_button("登入系統", use_container_width=True)

            if submit:
                # 比對帳號是否存在，且密碼是否正確
                if input_user in valid_users and valid_users[input_user] == input_pwd:
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = input_user  # 記錄當前登入者
                    st.success(f"身分驗證成功，歡迎 {input_user}！正在載入...")
                    st.rerun()
                else:
                    st.error("❌ 帳號不存在或密碼錯誤，請重新確認！")

    return False


# 若未通過登入驗證，立即終止執行後續程式碼
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

    def get_previous_trading_date(
        self, target_date: datetime.date
    ) -> datetime.date:
        """往前推算前一個營業日（排除週末）作為已知特徵日 (T-1)"""
        d = target_date - datetime.timedelta(days=1)
        while d.weekday() >= 5:  # 5: 週六, 6: 週日
            d -= datetime.timedelta(days=1)
        return d

    def fetch_twse_institutional(
        self, target_date: datetime.date
    ) -> Tuple[pd.DataFrame, str]:
        """爬取證交所三大法人買賣超 (單位: 億元)"""
        date_str = target_date.strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/zh/fund/BFI82U?response=json&dayDate={date_str}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=8)
            data = resp.json()
            if data.get("stat") != "OK" or "data" not in data:
                return pd.DataFrame(), f"查無資料或尚未公告 ({date_str})"

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
                "投信": "trust_net",
                "自營商(合計)": "dealer_net",
            }
            df["key"] = df["item"].map(mapping)
            feat_df = df.dropna(subset=["key"])[["key", "net"]]
            feat_dict = dict(zip(feat_df["key"], feat_df["net"]))
            feat_dict["date"] = pd.to_datetime(target_date)
            return pd.DataFrame([feat_dict]).set_index("date"), "正常取得"
        except Exception as e:
            return pd.DataFrame(), f"擷取異常: {str(e)}"

    def fetch_taifex_oi(
        self, target_date: datetime.date
    ) -> Tuple[pd.DataFrame, str]:
        """爬取期交所三大法人台指期未平倉多空淨額 (TXF)"""
        date_str = target_date.strftime("%Y/%m/%d")
        url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
        payload = {
            "queryStartDate": date_str,
            "queryEndDate": date_str,
            "commodityId": "TXF",
        }
        try:
            resp = requests.post(
                url, data=payload, headers=self.headers, timeout=8
            )
            resp.encoding = "big5"
            content = resp.text

            if (
                "查無資料" in content
                or resp.status_code != 200
                or len(content.strip()) == 0
            ):
                return pd.DataFrame(), f"查無資料或尚未公告 ({date_str})"

            df = pd.read_csv(io.StringIO(content))
            df.columns = [str(c).strip() for c in df.columns]

            name_col = [c for c in df.columns if "商品名稱" in c]
            id_col = [c for c in df.columns if "身份別" in c]

            if not name_col or not id_col:
                return pd.DataFrame(), "期交所 CSV 格式解析失敗"

            df = df[df[name_col[0]].astype(str).str.strip() == "臺股期貨"]

            id_map = {
                "外資及陸資": "tx_foreign_oi",
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
                return pd.DataFrame([res]).set_index("date"), "正常取得"
            else:
                return (
                    pd.DataFrame(),
                    f"未能在臺股期貨中找到外資留倉數據 ({date_str})",
                )
        except Exception as e:
            return pd.DataFrame(), f"擷取異常: {str(e)}"

    def fetch_us_market(
        self, target_date: datetime.date, days_back: int = 15
    ) -> Tuple[pd.DataFrame, str]:
        """爬取美股指標：費半、TSM ADR、S&P500、VIX"""
        tickers = {
            "^SOX": "sox_ret",
            "TSM": "tsm_ret",
            "^GSPC": "sp500_ret",
            "^VIX": "vix_close",
        }
        try:
            start_d = (
                target_date - datetime.timedelta(days=days_back)
            ).strftime("%Y-%m-%d")
            end_d = (target_date + datetime.timedelta(days=2)).strftime(
                "%Y-%m-%d"
            )
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
            return features, "正常取得"
        except Exception as e:
            return pd.DataFrame(), f"擷取異常: {str(e)}"

    def fetch_historical_training_and_verification_data(
        self, target_date: datetime.date, train_days: int = 250
    ) -> pd.DataFrame:
        """擷取過去約 250 個營業日數據，用於動態 Logistic 迴歸配適與回測比對"""
        try:
            start_d = (
                target_date - datetime.timedelta(days=int(train_days * 1.6))
            ).strftime("%Y-%m-%d")
            end_d = (target_date + datetime.timedelta(days=5)).strftime(
                "%Y-%m-%d"
            )

            tickers = {
                "^TWII": "twii_close",
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
            price = raw["Close"] if "Close" in raw else raw["Adj Close"]

            df = pd.DataFrame(index=price.index)
            df["TWII"] = price["^TWII"]
            df["sox_ret"] = price["^SOX"].pct_change() * 100.0
            df["tsm_ret"] = price["TSM"].pct_change() * 100.0
            df["vix_close"] = price["^VIX"]

            df["TWII_Return"] = df["TWII"].pct_change() * 100.0

            df = df.dropna(subset=["TWII", "sox_ret", "tsm_ret", "vix_close"])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception:
            return pd.DataFrame()


# -----------------------------------------------------------------------------
# 2. Logistic Regression 機率推論引擎 (支援靜態先驗與動態 250 日 MLE 配適)
# -----------------------------------------------------------------------------
class TaiexLogisticEngine:

    def __init__(self):
        self.default_scaler = {
            "foreign_net": {"mu": 5.0, "sigma": 150.0},
            "trust_net": {"mu": 10.0, "sigma": 30.0},
            "tx_foreign_oi": {"mu": -12000.0, "sigma": 10000.0},
            "sox_ret": {"mu": 0.05, "sigma": 1.8},
            "tsm_ret": {"mu": 0.10, "sigma": 2.2},
            "vix_close": {"mu": 17.5, "sigma": 4.5},
        }

        self.default_beta = {
            "foreign_net": 0.45,
            "trust_net": 0.28,
            "tx_foreign_oi": 0.52,
            "sox_ret": 0.65,
            "tsm_ret": 0.58,
            "vix_close": -0.32,
        }
        self.default_intercept = 0.05

        self.active_scaler = self.default_scaler.copy()
        self.active_beta = self.default_beta.copy()
        self.active_intercept = self.default_intercept
        self.fit_accuracy = None
        self.fit_sample_size = 0

    def fit_rolling_model(
        self, hist_df: pd.DataFrame, target_dt: pd.Timestamp
    ):
        """以評估日前 250 個交易日真實行情，動態配適 Logistic 權重"""
        valid_df = hist_df[hist_df.index < target_dt].copy()
        if len(valid_df) < 60:
            return False

        train_sample = valid_df.tail(250).copy()
        self.fit_sample_size = len(train_sample)

        train_sample["y"] = (train_sample["TWII_Return"] > 0).astype(int)

        features = ["sox_ret", "tsm_ret", "vix_close"]
        X_raw = train_sample[features].shift(1).dropna()
        y = train_sample.loc[X_raw.index, "y"]

        new_scaler = self.default_scaler.copy()
        X_scaled = pd.DataFrame(index=X_raw.index)
        for col in features:
            mu = float(X_raw[col].mean())
            sigma = float(X_raw[col].std()) if X_raw[col].std() > 0 else 1.0
            new_scaler[col] = {"mu": mu, "sigma": sigma}
            X_scaled[col] = (X_raw[col] - mu) / sigma

        clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200)
        clf.fit(X_scaled, y)

        self.active_scaler = new_scaler
        self.active_intercept = float(clf.intercept_[0])

        new_beta = self.default_beta.copy()
        for idx, col in enumerate(features):
            new_beta[col] = float(clf.coef_[0][idx])
        self.active_beta = new_beta

        self.fit_accuracy = float(clf.score(X_scaled, y)) * 100.0
        return True

    def reset_to_static(self):
        """還原為預設專家固定係數"""
        self.active_scaler = self.default_scaler.copy()
        self.active_beta = self.default_beta.copy()
        self.active_intercept = self.default_intercept
        self.fit_accuracy = None
        self.fit_sample_size = 0

    def predict(self, raw_features: Dict[str, float]) -> Tuple[float, dict]:
        """計算勝率機率 P(Up) 與各特徵 Log-Odds 貢獻度"""
        z = self.active_intercept
        contributions = {}

        for k, weight in self.active_beta.items():
            val = raw_features.get(k, 0.0)
            mu = self.active_scaler[k]["mu"]
            sigma = self.active_scaler[k]["sigma"]
            z_score = (val - mu) / sigma
            contrib = weight * z_score
            contributions[k] = contrib
            z += contrib

        prob_up = 1.0 / (1.0 + np.exp(-z))
        return prob_up, contributions


# -----------------------------------------------------------------------------
# 3. 儀表板繪圖元件 (Plotly)
# -----------------------------------------------------------------------------
def render_gauge(prob_up: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_up * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "<b>加權指數上漲勝率預測 P(Up)</b>",
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
        "sox_ret": "前日費半指數 (SOX)",
        "tsm_ret": "前日台積電 ADR (TSM)",
        "tx_foreign_oi": "前日外資期貨淨留倉 (TXF)",
        "foreign_net": "前日外資現貨買賣超",
        "trust_net": "前日投信現貨買賣超",
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
        title="<b>前日因子對今日 Log-Odds 邊際貢獻 (Beta × Z-Score)</b>",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="邊際貢獻 (大於 0 偏多，小於 0 偏空)",
            gridcolor="#334155",
            zerolinecolor="#64748b",
        ),
        yaxis=dict(autorange="reversed"),
        font={"color": "#f8fafc"},
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# -----------------------------------------------------------------------------
# 4. Streamlit 介面佈局與資料載入
# -----------------------------------------------------------------------------
pipeline = MarketDataPipeline()
engine = TaiexLogisticEngine()

st.sidebar.header("🕹️ 特徵監控與模擬器")

# 在側邊欄顯示目前登入的使用者名稱與登出按鈕
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
    ["固定專家權重", "過去 250 日動態滾動配適 (MLE)"],
    index=1,
    help="選擇是否以評估日前 250 個營業日數據自動擬合出最新 Beta 係數與常模",
)

selected_target_date = st.sidebar.date_input(
    "選擇目標預測交易日 (T)", datetime.date.today()
)

prev_feature_date = pipeline.get_previous_trading_date(selected_target_date)

btn_col1, btn_col2 = st.sidebar.columns([3, 2])
with btn_col1:
    fetch_btn = st.button("🚀 擷取前日數據", use_container_width=True)
with btn_col2:
    reset_btn = st.button("↺ 還原特徵", use_container_width=True)

st.sidebar.caption(
    f"📌 **邏輯說明**：目標預測日為 **{selected_target_date}**，特徵因子均擷取前一營業日（**{prev_feature_date}**）之結算數據。"
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
    st.session_state["slider_tx_oi"] = safe_clamp(
        tx_oi, -80000.0, 50000.0, -12000.0
    )
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
        f"正在連線擷取前一交易日 ({prev_feature_date}) 籌碼與過去 250 日訓練數據..."
    ):
        twse_df, twse_msg = pipeline.fetch_twse_institutional(prev_feature_date)
        taifex_df, taifex_msg = pipeline.fetch_taifex_oi(prev_feature_date)
        us_df, us_msg = pipeline.fetch_us_market(
            prev_feature_date, days_back=15
        )
        hist_df = pipeline.fetch_historical_training_and_verification_data(
            selected_target_date, train_days=250
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
        sox_r = float(latest_us["sox_ret"]) if latest_us is not None else 0.0
        tsm_r = float(latest_us["tsm_ret"]) if latest_us is not None else 0.0
        vix_c = float(latest_us["vix_close"]) if latest_us is not None else 18.0

        st.session_state["cached_data"] = {
            "target_date": selected_target_date,
            "feature_date": prev_feature_date,
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
if (
    model_mode == "過去 250 日動態滾動配適 (MLE)"
    and not data["hist_df"].empty
):
    engine.fit_rolling_model(data["hist_df"], target_dt)
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

st.sidebar.caption("💡 可手動微調前日特徵參數進行敏感度測試：")

sim_foreign_net = st.sidebar.slider(
    "前日外資現貨買賣超 (億元)",
    min_value=-1200.0,
    max_value=1200.0,
    value=st.session_state["slider_f_net"],
    step=1.0,
    key="slider_f_net",
)
sim_trust_net = st.sidebar.slider(
    "前日投信現貨買賣超 (億元)",
    min_value=-300.0,
    max_value=300.0,
    value=st.session_state["slider_t_net"],
    step=1.0,
    key="slider_t_net",
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

current_features = {
    "foreign_net": sim_foreign_net,
    "trust_net": sim_trust_net,
    "tx_foreign_oi": sim_tx_oi,
    "sox_ret": sim_sox_ret,
    "tsm_ret": sim_tsm_ret,
    "vix_close": sim_vix_close,
}

prob_up, contributions = engine.predict(current_features)

# -----------------------------------------------------------------------------
# 5. 主視覺顯示 (Dashboard View)
# -----------------------------------------------------------------------------
st.title("🎯 TAIEX 台灣加權指數次日漲跌機率監控儀表")

fetch_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
m_desc = (
    f"動態 250 日 MLE 配適 (樣本數: {engine.fit_sample_size} 天, 訓練準確率: {engine.fit_accuracy:.1f}%)"
    if engine.fit_accuracy is not None
    else "固定專家權重"
)

st.markdown(
    f"""
> **系統運行狀態與模型參數因果檢驗**  
> • **目標預測交易日 (T)**：`{data['target_date']}` | **模型權重模式**：`{m_desc}`  
> • **特徵基準日 (T-1)**：`{data['feature_date']}`（以盤前已公布之真實數據進行前瞻預測）  
> • **證券交易所 (TWSE)**：三大法人買賣超 (`狀態: {data['twse_msg']}`)  
> • **期貨交易所 (TAIFEX)**：臺股期貨外資留倉 (`狀態: {data['taifex_msg']}`)  
> • **美股行情 (yfinance)**：費半、TSM ADR、VIX (`狀態: {data['us_msg']}`)
"""
)

st.markdown("---")

col_gauge, col_waterfall = st.columns([5, 5])

with col_gauge:
    st.plotly_chart(render_gauge(prob_up), use_container_width=True)
    if prob_up >= 0.55:
        st.markdown(
            f"<h3 style='text-align: center;' class='text-bullish'>預測信號：強烈看多 (上漲勝率 {prob_up*100:.1f}%)</h3>",
            unsafe_allow_html=True,
        )
    elif prob_up <= 0.45:
        st.markdown(
            f"<h3 style='text-align: center;' class='text-bearish'>預測信號：偏空防禦 (上漲勝率 {prob_up*100:.1f}%)</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h3 style='text-align: center; color: #94a3b8;'>預測信號：區間震盪整理 (上漲勝率 {prob_up*100:.1f}%)</h3>",
            unsafe_allow_html=True,
        )

with col_waterfall:
    st.plotly_chart(render_waterfall(contributions), use_container_width=True)

with st.expander("🔍 檢視當前 Logistic 迴歸係數與常模 (Beta & Scaler)"):
    w_df = pd.DataFrame(
        [
            {
                "特徵因子": k,
                "Beta 迴歸權重": round(engine.active_beta[k], 4),
                "常模均值 (μ)": round(engine.active_scaler[k]["mu"], 3),
                "常模標準差 (σ)": round(
                    engine.active_scaler[k]["sigma"], 3
                ),
            }
            for k in engine.active_beta.keys()
        ]
    )
    st.dataframe(w_df, hide_index=True, use_container_width=True)

st.subheader("📊 盤前已知前日核心特徵數值一覽 (T-1 數據)")
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.metric(
        "前日外資現貨買賣超",
        f"{sim_foreign_net:+.1f} 億",
        delta=f"{sim_foreign_net:+.1f} 億",
        delta_color="normal" if sim_foreign_net >= 0 else "inverse",
    )
    st.caption("來源: TWSE (T-1)")

with m2:
    st.metric(
        "前日投信現貨買賣超",
        f"{sim_trust_net:+.1f} 億",
        delta=f"{sim_trust_net:+.1f} 億",
        delta_color="normal" if sim_trust_net >= 0 else "inverse",
    )
    st.caption("來源: TWSE (T-1)")

with m3:
    oi_status = "淨空單" if sim_tx_oi < 0 else "淨多單"
    st.metric(
        f"前日外資期貨 ({oi_status})",
        f"{abs(int(sim_tx_oi)):,} 口",
        delta=f"{int(sim_tx_oi):+,} 口",
        delta_color="normal" if sim_tx_oi >= -10000 else "inverse",
    )
    st.caption("來源: TAIFEX (T-1)")

with m4:
    st.metric(
        "前日費半 (SOX)",
        f"{sim_sox_ret:+.2f}%",
        delta=f"{sim_sox_ret:+.2f}%",
        delta_color="normal" if sim_sox_ret >= 0 else "inverse",
    )
    st.caption("來源: yfinance (T-1)")

with m5:
    st.metric(
        "前日台積電 ADR",
        f"{sim_tsm_ret:+.2f}%",
        delta=f"{sim_tsm_ret:+.2f}%",
        delta_color="normal" if sim_tsm_ret >= 0 else "inverse",
    )
    st.caption("來源: yfinance (T-1)")

with m6:
    st.metric(
        "前日 VIX 恐慌指數",
        f"{sim_vix_close:.1f}",
        delta=f"{sim_vix_close - 18.0:+.1f}",
        delta_color="inverse" if sim_vix_close >= 20 else "normal",
    )
    st.caption("來源: yfinance (T-1)")

# -----------------------------------------------------------------------------
# 6. 過往 10 日真實預測勝率、疊加核驗與策略累積模擬損益 (以 T-1 預測 T)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 過往 10 日預測勝率、疊加核驗與策略累積模擬損益 (Equity Curve)")

hist_df = data["hist_df"]

if not hist_df.empty:
    valid_hist = hist_df[hist_df.index <= target_dt]

    if len(valid_hist) >= 10:
        ten_days_df = valid_hist.tail(10).copy()

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 3, 3])
        with ctrl_col1:
            threshold_bull = st.slider(
                "🎯 多方發動信心門檻 (%)", 50, 75, 60, step=1
            )
        with ctrl_col2:
            threshold_bear = st.slider(
                "🛡️ 空方防禦信心門檻 (%)", 25, 50, 40, step=1
            )
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

        st.caption(
            f"💡 **交易回測因果關係**：以評估日 $T-1$ 的數據預測 $T$ 日之走勢。勝率 $\\ge {threshold_bull}\\%$ 做多 (+1)，$\\le {threshold_bear}\\%$ 放空 (-1)；其餘**中性觀望 (0 部位)**。每次部位變動扣除 **{cost_rate_pct:.3f}%** 摩擦成本。"
        )

        calc_probs = []
        for i in range(len(ten_days_df)):
            current_date_idx = ten_days_df.index[i]
            if current_date_idx == target_dt:
                p, _ = engine.predict(current_features)
            else:
                prev_row_loc = hist_df.index.get_loc(current_date_idx) - 1
                if prev_row_loc >= 0:
                    prev_r = hist_df.iloc[prev_row_loc]
                    hist_feat = {
                        "foreign_net": 5.0,
                        "trust_net": 10.0,
                        "tx_foreign_oi": -12000.0,
                        "sox_ret": float(prev_r["sox_ret"]),
                        "tsm_ret": float(prev_r["tsm_ret"]),
                        "vix_close": float(prev_r["vix_close"]),
                    }
                    p, _ = engine.predict(hist_feat)
                else:
                    p = 0.5
            calc_probs.append(round(p, 4))

        ten_days_df["Prob_Up"] = calc_probs
        date_labels = ten_days_df.index.strftime("%Y-%m-%d").tolist()
        probs_pct = [p * 100 for p in ten_days_df["Prob_Up"]]
        returns_list = ten_days_df["TWII_Return"].tolist()

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
                    status = "✅ 吻合 (看多上漲)"
                    correct_signals += 1
                else:
                    status = "❌ 不吻合 (看多下跌)"

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
                    status = "✅ 吻合 (看空下跌)"
                    correct_signals += 1
                else:
                    status = "❌ 不吻合 (看空上漲)"

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
                status = "濾除不計 (中性震盪)"
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

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric(
                "高信心預測勝率",
                f"{win_rate:.1f}%" if active_signals > 0 else "無觸發",
                delta=f"有效 {active_signals} 天 / 命中 {correct_signals} 天",
                delta_color="normal" if win_rate >= 60 else "inverse",
            )
        with k2:
            st.metric(
                "策略累積淨報酬 (扣成本)",
                f"{total_net_return:+.2f}%",
                delta=f"{total_net_return - total_bnh_return:+.2f}% 超額",
                delta_color="normal" if total_net_return >= 0 else "inverse",
            )
        with k3:
            st.metric(
                "同期大盤買入持有",
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
        with k5:
            st.metric(
                "最大淨值回撤 (MDD)",
                f"{max_drawdown:.2f}%",
                delta_color="inverse",
            )

        fig_verify = go.Figure()

        fig_verify.add_trace(
            go.Scatter(
                x=date_labels,
                y=probs_pct,
                mode="lines+markers+text",
                name="前日預測今日上漲勝率 P(Up)",
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
                name="加權指數當日實際收盤漲跌幅 (%)",
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
            title="<b>過往 10 日模型勝率 (T-1) 與加權指數當日實際漲跌幅 (T) 疊加比對</b>",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            xaxis=dict(title="目標預測交易日 (T)", gridcolor="#334155"),
            yaxis=dict(
                title="前日預測勝率 P(Up)",
                range=[0, 105],
                gridcolor="#334155",
                ticksuffix="%",
            ),
            yaxis2=dict(
                title="當日實際收盤漲跌幅 (%)",
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
                name="策略累積淨報酬 (扣除摩擦成本)",
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
                name="大盤買入持有基準 (Buy & Hold)",
                line=dict(color="#94a3b8", width=2, dash="dash"),
            )
        )

        fig_equity.add_hline(
            y=0, line_width=1, line_color="#64748b", line_dash="solid"
        )

        fig_equity.update_layout(
            title="<b>累積模擬損益曲線 (Net Equity Curve) vs 摩擦損耗與大盤基準</b>",
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
                "前日勝率預測": [f"{p:.1f}%" for p in probs_pct],
                "訊號類型": signal_types,
                "策略部位": pos_labels,
                "當日指數實際漲跌": [
                    f"{r:+.2f}%" if not np.isnan(r) else "尚未結算"
                    for r in returns_list
                ],
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
                "信心核驗狀態": validation_status,
            }
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.warning("歷史交易日資料不足 10 天，無法繪製完整比對圖。")
else:
    st.warning("加權指數歷史數據擷取中或暫時無法取得。")