import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="MDD 계산기 대시보드",
    page_icon="📉",
    layout="wide",
)

st.title("📉 MDD 계산기 대시보드")
st.caption(
    "최대낙폭(MDD) 분석, 누적수익률, 낙폭 사건 로그, 벤치마크 비교, 백테스트 요약표까지 한 번에 보는 Streamlit 버전입니다."
)

TRADING_DAYS = 252


# -----------------------------
# 데이터 구조
# -----------------------------
@dataclass
class PriceBundle:
    series: pd.Series
    display_name: str
    symbol_used: str
    source: str
    currency: str


# -----------------------------
# 유틸 함수
# -----------------------------
def normalize_input_ticker(raw: str) -> str:
    ticker = raw.strip().upper().replace(" ", "")
    alias_map = {
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
        "USDKRW": "KRW=X",
        "USD:KRW": "KRW=X",
        "USD/KRW": "KRW=X",
        "GCW00": "GC=F",
        "XAUUSD": "GC=F",
        "GOLD": "GC=F",
    }
    return alias_map.get(ticker, ticker)



def is_krx_numeric_ticker(ticker: str) -> bool:
    return bool(re.fullmatch(r"A?\d{6}", ticker))



def parse_optional_date(text: str) -> Optional[pd.Timestamp]:
    text = text.strip()
    if not text:
        return None
    try:
        return pd.Timestamp(text).normalize()
    except Exception:
        raise ValueError(f"날짜 형식이 올바르지 않습니다: {text} (예: 2020-01-01)")



def format_pct(v: Optional[float], digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "-"
    return f"{v:.{digits}%}"



def format_num(v: Optional[float], digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "-"
    return f"{v:,.{digits}f}"



def bucket_label(threshold: float) -> str:
    if threshold == 0:
        return "0% (ATH)"
    return f"{int(threshold * 100)}%"


@st.cache_data(show_spinner=False)
def get_krx_listing() -> pd.DataFrame:
    if fdr is None:
        return pd.DataFrame()
    try:
        df = fdr.StockListing("KRX")
        if "Code" in df.columns:
            df["Code"] = df["Code"].astype(str).str.zfill(6)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_with_fdr(symbol: str) -> PriceBundle:
    if fdr is None:
        raise RuntimeError("FinanceDataReader가 설치되어 있지 않습니다.")

    clean_symbol = symbol.replace("A", "") if re.fullmatch(r"A\d{6}", symbol) else symbol
    df = fdr.DataReader(clean_symbol)
    if df is None or df.empty:
        raise ValueError(f"FinanceDataReader에서 데이터를 찾지 못했습니다: {clean_symbol}")

    close_col = "Close" if "Close" in df.columns else df.columns[0]
    series = df[close_col].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.dropna().sort_index().astype(float)

    listing = get_krx_listing()
    display_name = clean_symbol
    if not listing.empty and "Name" in listing.columns:
        matched = listing.loc[listing["Code"] == clean_symbol, "Name"]
        if not matched.empty:
            display_name = f"{matched.iloc[0]} ({clean_symbol})"

    return PriceBundle(
        series=series,
        display_name=display_name,
        symbol_used=clean_symbol,
        source="FinanceDataReader",
        currency="KRW",
    )


@st.cache_data(show_spinner=False)
def fetch_with_yfinance(symbol: str) -> PriceBundle:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="max", auto_adjust=True, actions=False)
    if hist is None or hist.empty:
        raise ValueError(f"Yahoo Finance에서 데이터를 찾지 못했습니다: {symbol}")
    if "Close" not in hist.columns:
        raise ValueError("Yahoo Finance 응답에 Close 컬럼이 없습니다.")

    series = hist["Close"].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.dropna().sort_index().astype(float)

    display_name = symbol
    currency = "USD"
    try:
        info = ticker.get_info()
        display_name = info.get("shortName") or info.get("longName") or symbol
        currency = info.get("currency") or currency
    except Exception:
        if symbol.endswith(".KS") or symbol.endswith(".KQ"):
            currency = "KRW"

    return PriceBundle(
        series=series,
        display_name=f"{display_name} ({symbol})",
        symbol_used=symbol,
        source="Yahoo Finance",
        currency=currency,
    )


@st.cache_data(show_spinner=False)
def fetch_price_data(user_ticker: str) -> PriceBundle:
    symbol = normalize_input_ticker(user_ticker)
    errors = []

    if is_krx_numeric_ticker(symbol):
        for candidate in [symbol.replace("A", "")]:
            try:
                return fetch_with_fdr(candidate)
            except Exception as e:
                errors.append(f"FDR({candidate}): {e}")

        for suffix in [".KS", ".KQ"]:
            try:
                return fetch_with_yfinance(symbol.replace("A", "") + suffix)
            except Exception as e:
                errors.append(f"Yahoo({symbol}{suffix}): {e}")

    for candidate in [symbol]:
        try:
            return fetch_with_yfinance(candidate)
        except Exception as e:
            errors.append(f"Yahoo({candidate}): {e}")

    if re.fullmatch(r"\d{6}", symbol):
        for suffix in [".KS", ".KQ"]:
            try:
                return fetch_with_yfinance(symbol + suffix)
            except Exception as e:
                errors.append(f"Yahoo({symbol}{suffix}): {e}")

    raise RuntimeError("\n".join(errors))


@st.cache_data(show_spinner=False)
def fetch_usdkrw_series() -> pd.Series:
    errors = []

    if fdr is not None:
        try:
            fx = fdr.DataReader("USD/KRW")
            if fx is not None and not fx.empty:
                close_col = "Close" if "Close" in fx.columns else fx.columns[0]
                s = fx[close_col].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                s = s.dropna().sort_index().astype(float)
                if not s.empty:
                    return s
        except Exception as e:
            errors.append(f"FDR USD/KRW: {e}")

    try:
        return fetch_with_yfinance("KRW=X").series
    except Exception as e:
        errors.append(f"Yahoo KRW=X: {e}")

    raise RuntimeError(" | ".join(errors))


# -----------------------------
# 계산 함수
# -----------------------------
def restrict_date_range(
    s: pd.Series,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Tuple[pd.Series, pd.Timestamp, pd.Timestamp, List[str]]:
    notes: List[str] = []
    s = s.dropna().sort_index()
    min_date = s.index.min().normalize()
    max_date = s.index.max().normalize()

    actual_start = min_date if start is None else max(start.normalize(), min_date)
    actual_end = max_date if end is None else min(end.normalize(), max_date)

    if start is not None and start.normalize() < min_date:
        notes.append(f"시작일이 데이터 시작일보다 빨라서 {min_date.date()}로 자동 조정했습니다.")
    if end is not None and end.normalize() > max_date:
        notes.append(f"종료일이 데이터 종료일보다 늦어서 {max_date.date()}로 자동 조정했습니다.")
    if actual_start > actual_end:
        raise ValueError("시작일이 종료일보다 늦습니다.")

    filtered = s.loc[(s.index.normalize() >= actual_start) & (s.index.normalize() <= actual_end)]
    if filtered.empty:
        raise ValueError("선택한 기간에 조회 가능한 데이터가 없습니다.")

    return filtered, actual_start, actual_end, notes



def apply_period_preset(series: pd.Series, preset: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[str]]:
    if preset == "전체":
        return None, None, None

    end = series.index.max().normalize()
    years_map = {
        "최근 20년": 20,
        "최근 10년": 10,
        "최근 5년": 5,
        "최근 3년": 3,
    }
    years = years_map.get(preset)
    if years is None:
        return None, None, None

    start_candidate = end - pd.DateOffset(years=years)
    note = f"기간 프리셋 '{preset}'을 적용했습니다. 종료일 기준으로 최근 {years}년 구간을 사용합니다."
    return start_candidate, end, note



def days_to_next_ath(dd: pd.Series) -> pd.Series:
    dd_values = dd.to_numpy()
    n = len(dd_values)
    next_zero_idx = np.full(n, np.nan)

    last_zero = np.nan
    for i in range(n - 1, -1, -1):
        if np.isclose(dd_values[i], 0.0):
            last_zero = i
        next_zero_idx[i] = last_zero

    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(next_zero_idx[i]):
            out[i] = np.nan
        else:
            out[i] = float(next_zero_idx[i] - i)
    return pd.Series(out, index=dd.index, name="DaysToNextATH")



def build_price_frame(price: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Close": price.dropna().astype(float)})
    df["High"] = df["Close"].cummax()
    df["Drawdown"] = df["Close"] / df["High"] - 1.0
    df["CumReturn"] = df["Close"] / df["Close"].iloc[0] - 1.0
    df["Indexed100"] = df["Close"] / df["Close"].iloc[0] * 100.0
    df["DailyReturn"] = df["Close"].pct_change()
    df["DaysToNextATH"] = days_to_next_ath(df["Drawdown"])
    return df



def summarize_stats(df: pd.DataFrame) -> Dict[str, object]:
    current_price = float(df["Close"].iloc[-1])
    ath = float(df["High"].max())
    current_dd = float(df["Drawdown"].iloc[-1])
    mdd = float(df["Drawdown"].min())
    mdd_date = pd.Timestamp(df["Drawdown"].idxmin())
    avg_dd = float(df["Drawdown"].mean())
    total_days = int(len(df))
    required_return_to_ath = ath / current_price - 1.0 if current_price > 0 else np.nan
    current_bucket = get_current_bucket_label(current_dd)
    deeper_or_equal_ratio = float((df["Drawdown"] <= current_dd).mean())
    shallower_or_equal_ratio = float((df["Drawdown"] >= current_dd).mean())

    return {
        "current_price": current_price,
        "ath": ath,
        "current_dd": current_dd,
        "mdd": mdd,
        "mdd_date": mdd_date,
        "avg_dd": avg_dd,
        "total_days": total_days,
        "required_return_to_ath": required_return_to_ath,
        "current_bucket": current_bucket,
        "deeper_or_equal_ratio": deeper_or_equal_ratio,
        "shallower_or_equal_ratio": shallower_or_equal_ratio,
    }



def get_current_bucket_label(current_dd: float) -> str:
    if np.isclose(current_dd, 0.0):
        return "0% (ATH)"
    lower = np.floor(current_dd * 20) / 20  # 5% 구간 단위
    upper = lower + 0.05
    return f"{upper:.0%} ~ {lower:.0%}"



def build_thresholds(step_pct: int = 5, max_pct: int = 100) -> List[float]:
    thresholds = [0.0]
    for p in range(step_pct, max_pct + step_pct, step_pct):
        thresholds.append(-p / 100)
    return thresholds



def compute_segment_recovery_stats(dd: pd.Series, thresholds: List[float]):
    values = dd.to_numpy(dtype=float)
    n = len(values)

    def assign_segment(x: float) -> int:
        if np.isclose(x, 0.0):
            return 0
        for i in range(1, len(thresholds)):
            upper = thresholds[i - 1]
            lower = thresholds[i]
            if lower <= x < upper:
                return i
        return len(thresholds) - 1

    segments = np.array([assign_segment(x) for x in values], dtype=int)
    entry_counts = [0] * len(thresholds)
    recovered_counts = [0] * len(thresholds)
    recovery_days_store: Dict[int, List[int]] = {i: [] for i in range(len(thresholds))}

    next_zero = np.full(n, -1, dtype=int)
    next_idx = -1
    for i in range(n - 1, -1, -1):
        if np.isclose(values[i], 0.0):
            next_idx = i
        next_zero[i] = next_idx

    prev_seg = segments[0]
    for i in range(1, n):
        cur_seg = segments[i]
        if cur_seg > prev_seg and cur_seg > 0:
            entry_counts[cur_seg] += 1
            zero_idx = next_zero[i]
            if zero_idx != -1 and zero_idx > i:
                recovered_counts[cur_seg] += 1
                recovery_days_store[cur_seg].append(int(zero_idx - i))
        prev_seg = cur_seg

    avg_recovery_days = []
    for i in range(len(thresholds)):
        avg_days = np.nan if len(recovery_days_store[i]) == 0 else float(np.mean(recovery_days_store[i]))
        avg_recovery_days.append(avg_days)

    return entry_counts, recovered_counts, avg_recovery_days



def build_bucket_table(df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    dd = df["Drawdown"].copy()
    total_days = len(dd)

    cumulative_ratios = []
    condition_days_list = []
    segment_weights = []
    prev_cum = 0.0
    for t in thresholds:
        condition_days = int((dd >= t).sum())
        cum_ratio = condition_days / total_days if total_days > 0 else np.nan
        seg_weight = cum_ratio - prev_cum
        cumulative_ratios.append(cum_ratio)
        condition_days_list.append(condition_days)
        segment_weights.append(seg_weight)
        prev_cum = cum_ratio

    entry_counts, recovered_counts, avg_recovery_days = compute_segment_recovery_stats(dd, thresholds)

    out = pd.DataFrame(
        {
            "MDD": thresholds,
            "누적 비율": cumulative_ratios,
            "조건일": condition_days_list,
            "개장일": total_days,
            "구간 비중": segment_weights,
            "진입사건": entry_counts,
            "이후 회복": recovered_counts,
            "평균 회복일수(구간 진입→전고점 회복)": avg_recovery_days,
        }
    )
    return out



def convert_usd_to_krw(price_usd: pd.Series, usdkrw: pd.Series) -> pd.Series:
    fx = usdkrw.copy().sort_index().astype(float)
    fx_aligned = fx.reindex(price_usd.index.union(fx.index)).sort_index().ffill().reindex(price_usd.index)
    if fx_aligned.isna().all():
        raise ValueError("환율 데이터를 가격 데이터 날짜에 맞춰 정렬할 수 없습니다.")
    return price_usd * fx_aligned



def build_metrics_table(price_df: pd.DataFrame, label: str) -> pd.DataFrame:
    r = price_df["DailyReturn"].dropna().astype(float)
    if r.empty:
        raise ValueError("수익률 계산에 필요한 데이터가 부족합니다.")

    calendar_days = max((price_df.index[-1] - price_df.index[0]).days, 1)
    years = calendar_days / 365.25
    cagr = float((price_df["Close"].iloc[-1] / price_df["Close"].iloc[0]) ** (1 / years) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else np.nan
    std = r.std(ddof=1)
    sharpe = float((r.mean() / std) * np.sqrt(TRADING_DAYS)) if pd.notna(std) and not np.isclose(std, 0) else np.nan

    downside = r[r < 0]
    downside_std = downside.std(ddof=1)
    sortino = float((r.mean() / downside_std) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 and pd.notna(downside_std) and not np.isclose(downside_std, 0) else np.nan

    dd = price_df["Drawdown"].astype(float)
    ulcer_index = float(np.sqrt(np.mean(np.square(dd.clip(upper=0)))))
    upi = float(cagr / ulcer_index) if pd.notna(ulcer_index) and not np.isclose(ulcer_index, 0) else np.nan
    mdd = float(dd.min())
    uwp = float((dd < 0).mean())

    out = pd.DataFrame(
        {
            "지표": ["CAGR", "MDD", "연변동성", "샤프지수", "소르티노지수", "UPI", "UWP(underwaterperiod)"],
            label: [cagr, mdd, vol, sharpe, sortino, upi, uwp],
        }
    )
    return out



def merge_metric_tables(left: pd.DataFrame, right: Optional[pd.DataFrame]) -> pd.DataFrame:
    if right is None:
        return left
    return left.merge(right, on="지표", how="outer")



def style_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    pct_metrics = {"CAGR", "MDD", "연변동성", "UWP(underwaterperiod)"}
    for col in view.columns[1:]:
        view[col] = [format_pct(v) if metric in pct_metrics else format_num(v) for metric, v in zip(view["지표"], view[col])]
    return view



def detect_drawdown_events(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    dd = df["Drawdown"]
    close = df["Close"]
    high = df["High"]

    events = []
    in_event = False
    entry_idx = None
    trough_idx = None
    trough_dd = None

    for i in range(len(df)):
        cur_dd = float(dd.iloc[i])
        idx = dd.index[i]

        if (not in_event) and (cur_dd <= threshold):
            in_event = True
            entry_idx = idx
            trough_idx = idx
            trough_dd = cur_dd

        if in_event:
            if cur_dd < trough_dd:
                trough_dd = cur_dd
                trough_idx = idx

            if np.isclose(cur_dd, 0.0):
                recovery_idx = idx
                entry_price = float(close.loc[entry_idx])
                trough_price = float(close.loc[trough_idx])
                recovery_price = float(close.loc[recovery_idx])
                ath_at_entry = float(high.loc[entry_idx])
                events.append(
                    {
                        "진입일": entry_idx.date(),
                        "진입 낙폭": float(dd.loc[entry_idx]),
                        "진입가": entry_price,
                        "당시 전고점": ath_at_entry,
                        "저점일": trough_idx.date(),
                        "최저 낙폭": trough_dd,
                        "저점가": trough_price,
                        "회복일": recovery_idx.date(),
                        "회복가": recovery_price,
                        "회복 거래일수": int(df.index.get_loc(recovery_idx) - df.index.get_loc(entry_idx)),
                        "회복 달력일수": int((recovery_idx - entry_idx).days),
                    }
                )
                in_event = False
                entry_idx = None
                trough_idx = None
                trough_dd = None

    if in_event and entry_idx is not None:
        entry_price = float(close.loc[entry_idx])
        trough_price = float(close.loc[trough_idx])
        ath_at_entry = float(high.loc[entry_idx])
        events.append(
            {
                "진입일": entry_idx.date(),
                "진입 낙폭": float(dd.loc[entry_idx]),
                "진입가": entry_price,
                "당시 전고점": ath_at_entry,
                "저점일": trough_idx.date(),
                "최저 낙폭": trough_dd,
                "저점가": trough_price,
                "회복일": "미회복",
                "회복가": np.nan,
                "회복 거래일수": np.nan,
                "회복 달력일수": np.nan,
            }
        )

    return pd.DataFrame(events)



def style_event_log(df: pd.DataFrame, currency: str) -> pd.DataFrame:
    if df.empty:
        return df
    view = df.copy()
    for col in ["진입 낙폭", "최저 낙폭"]:
        view[col] = view[col].map(lambda x: format_pct(x))
    for col in ["진입가", "당시 전고점", "저점가", "회복가"]:
        view[col] = view[col].map(lambda x: "-" if pd.isna(x) else format_price(float(x), currency))
    for col in ["회복 거래일수", "회복 달력일수"]:
        view[col] = view[col].map(lambda x: "-" if pd.isna(x) else f"{int(x):,}")
    return view



def align_for_compare(main_df: pd.DataFrame, bench_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_index = main_df.index.intersection(bench_df.index)
    if len(common_index) < 2:
        raise ValueError("비교 대상과 겹치는 날짜가 너무 적어서 비교할 수 없습니다.")
    return main_df.loc[common_index].copy(), bench_df.loc[common_index].copy()


# -----------------------------
# 시각화 함수
# -----------------------------
def format_price(v: float, currency: str) -> str:
    if currency == "KRW":
        return f"₩{v:,.0f}"
    if currency == "USD":
        return f"${v:,.2f}"
    return f"{v:,.2f} {currency}"



def drawdown_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Drawdown"] * 100,
            mode="lines",
            name="Drawdown",
            line=dict(width=1.5),
            fill="tozeroy",
        )
    )
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="고점대비 하락률 (%)",
        xaxis_title="날짜",
        hovermode="x unified",
    )
    fig.update_yaxes(ticksuffix="%")
    return fig



def price_and_high_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="현재가", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["High"], mode="lines", name="과거 최고가", line=dict(width=1.5, dash="dash")))
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="날짜",
        yaxis_title="가격",
        hovermode="x unified",
    )
    return fig



def indexed_return_chart(main_df: pd.DataFrame, main_label: str, benchmark_df: Optional[pd.DataFrame] = None, benchmark_label: Optional[str] = None, title: str = "누적 수익률 차트 (시작점 100)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df["Indexed100"], mode="lines", name=main_label, line=dict(width=2.2)))
    if benchmark_df is not None and benchmark_label is not None:
        fig.add_trace(go.Scatter(x=benchmark_df.index, y=benchmark_df["Indexed100"], mode="lines", name=benchmark_label, line=dict(width=1.8, dash="dash")))
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="날짜",
        yaxis_title="지수화 수익률 (시작=100)",
        hovermode="x unified",
    )
    return fig



def bucket_bar_chart(bucket_df: pd.DataFrame, title: str):
    chart_df = bucket_df.copy()
    chart_df["라벨"] = chart_df["MDD"].apply(bucket_label)
    fig = px.bar(chart_df, x="라벨", y="구간 비중", title=title)
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20), xaxis_title="MDD 구간", yaxis_title="비중")
    fig.update_yaxes(tickformat=".1%")
    return fig



def styled_bucket_table(bucket_df: pd.DataFrame) -> pd.DataFrame:
    view = bucket_df.copy()
    view["MDD"] = view["MDD"].map(lambda x: f"{x:.0%}")
    for col in ["누적 비율", "구간 비중"]:
        view[col] = view[col].map(lambda x: "-" if pd.isna(x) else f"{x:.1%}")
    for col in ["조건일", "개장일", "진입사건", "이후 회복"]:
        view[col] = view[col].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
    avg_col = "평균 회복일수(구간 진입→전고점 회복)"
    view[avg_col] = view[avg_col].map(lambda x: "-" if pd.isna(x) else f"{x:,.1f}")
    return view



def render_help_guide():
    with st.expander("📘 이 대시보드는 이렇게 해석하세요", expanded=False):
        st.markdown(
            """
### 1) 위쪽 핵심 지표
- **현재가**: 마지막 거래일 종가입니다.
- **과거 최고가**: 조회 기간 내에서 가장 높았던 가격입니다.
- **고점대비 하락률**: 지금 가격이 과거 최고가 대비 얼마나 내려와 있는지 보여줍니다.
- **MDD**: 조회 기간 중 가장 깊었던 낙폭입니다.
- **평균 고점대비 하락률**: 전체 기간 동안 평균적으로 얼마나 고점 아래에 있었는지 보는 보조지표입니다.
- **전고점 회복 필요 상승률**: 지금 가격에서 다시 최고가를 회복하려면 몇 % 올라야 하는지 보여줍니다.
  - 예: 현재 `-20%`면 회복에 필요한 상승률은 `+25%`입니다.

### 2) 현재 위치 요약 카드
- **현재 구간**: 지금 낙폭이 어느 구간에 속하는지 바로 보여줍니다.
- **현재보다 더 깊었던 날 비중**: 과거 전체 거래일 중 지금보다 더 깊게 빠졌던 날이 얼마나 있었는지 보여줍니다.
  - 낮을수록 현재 낙폭이 역사적으로 깊은 편이라는 뜻입니다.
- **현재보다 얕았던 날 비중**: 반대로 지금보다 덜 빠져 있거나 ATH에 가까웠던 날의 비중입니다.

### 3) 기간 프리셋
- **전체 / 최근 20년 / 최근 10년 / 최근 5년 / 최근 3년 / 직접 입력**을 선택할 수 있습니다.
- 같은 종목도 기간에 따라 MDD와 회복 속도가 크게 달라질 수 있어서, 여러 구간을 번갈아 보는 것이 좋습니다.

### 4) 누적 수익률 차트 (시작점 100)
- 시작일을 100으로 맞춘 차트입니다.
- 가격 수준이 달라도, 같은 기간에 **누가 더 잘 버텼고 더 많이 올랐는지** 직관적으로 보기 좋습니다.
- 벤치마크를 넣으면 둘을 같은 출발선에서 비교합니다.

### 5) 벤치마크 비교
- 기준 종목과 비교 대상을 같은 기간의 공통 날짜로 맞춘 뒤 성과를 비교합니다.
- MDD만 볼 때보다 **수익률 / 변동성 / 샤프 / 소르티노 / UPI**까지 같이 봐야 해석이 균형 잡힙니다.

### 6) MDD 구간별 통계표
- **누적 비율**: 해당 기준 이내에서 있었던 비중입니다.
- **조건일**: 그 누적 비율을 날짜 수로 환산한 값입니다.
- **구간 비중**: 바로 위 구간과 현재 구간 사이에 실제로 머문 비중입니다.
- **진입사건**: 더 깊은 하락 구간으로 처음 들어간 횟수입니다.
- **이후 회복**: 그 사건 중 나중에 전고점을 회복한 횟수입니다.
- **평균 회복일수(구간 진입→전고점 회복)**: 해당 구간에 처음 들어간 날부터 다시 전고점에 도달할 때까지 걸린 평균 거래일수입니다.
  - **저점에서 회복까지**가 아니라 **구간 진입 시점에서 회복까지**입니다.

### 7) 낙폭 사건 로그
- 특정 기준(예: -10%, -20%, -30%) 아래로 내려간 사건들을 하나씩 보여줍니다.
- **진입일 / 저점일 / 회복일 / 회복 거래일수**를 함께 보면서 과거 하락장이 얼마나 깊고 오래 갔는지 사례처럼 볼 수 있습니다.
- 아직 전고점을 회복하지 못한 최근 사건은 **미회복**으로 표시됩니다.

### 8) 백테스트 결과표
- **CAGR**: 연복리수익률
- **연변동성**: 일간 수익률 기준 연환산 변동성
- **샤프지수**: 변동성 대비 평균 수익률
- **소르티노지수**: 하락 변동성만 반영한 위험조정수익률
- **UPI**: 낙폭의 깊이와 지속기간을 반영한 Ulcer Performance Index 기반 지표
- **MDD**: 최대낙폭
- **UWP(underwaterperiod)**: 전체 거래일 중 전고점 아래(DD < 0)에 있었던 날짜 비중입니다

### 9) 해석할 때 주의할 점
- 이 대시보드는 **미래 예측기**가 아니라 **과거 패턴 요약기**입니다.
- MDD가 작다고 앞으로도 항상 안전하다는 뜻은 아닙니다.
- 가능하면 **전체 / 최근 10년 / 최근 5년**을 번갈아 보면서 해석하는 것이 좋습니다.
            """
        )



def render_context_summary(stats: Dict[str, object]):
    st.markdown("#### 현재 위치 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("현재 구간", stats["current_bucket"])
    c2.metric("현재보다 더 깊었던 날 비중", format_pct(stats["deeper_or_equal_ratio"]))
    c3.metric("현재보다 얕았던 날 비중", format_pct(stats["shallower_or_equal_ratio"]))
    st.caption(
        "해석 팁: '현재보다 더 깊었던 날 비중'이 낮을수록, 지금 낙폭이 과거 기준으로는 상대적으로 깊은 편입니다."
    )



def render_price_threshold_cards(stats: Dict[str, object], currency: str):
    st.markdown("#### 기준 가격")
    cols = st.columns(3)
    for col, th in zip(cols, [-0.10, -0.20, -0.30]):
        col.metric(f"기준 {int(abs(th)*100)}%", format_price(stats["ath"] * (1 + th), currency))



def render_main_block(
    df: pd.DataFrame,
    stats: Dict[str, object],
    bundle: PriceBundle,
    thresholds: List[float],
    benchmark_bundle: Optional[PriceBundle],
    benchmark_df: Optional[pd.DataFrame],
    event_threshold: float,
    section_prefix: str,
):
    bucket_df = build_bucket_table(df, thresholds)
    metric_df = build_metrics_table(df, bundle.display_name)

    aligned_main = None
    aligned_bench = None
    merged_metric_df = metric_df
    if benchmark_bundle is not None and benchmark_df is not None:
        try:
            aligned_main, aligned_bench = align_for_compare(df, benchmark_df)
            bench_metric_df = build_metrics_table(aligned_bench, benchmark_bundle.display_name)
            main_metric_aligned = build_metrics_table(aligned_main, bundle.display_name)
            merged_metric_df = merge_metric_tables(main_metric_aligned, bench_metric_df)
        except Exception as e:
            st.warning(f"벤치마크 비교를 건너뛰었습니다: {e}")
            aligned_main, aligned_bench = None, None

    mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
    mc1.metric("현재가", format_price(stats["current_price"], bundle.currency))
    mc2.metric("과거 최고가", format_price(stats["ath"], bundle.currency))
    mc3.metric("고점대비 하락률", format_pct(stats["current_dd"]))
    mc4.metric("MDD", format_pct(stats["mdd"]))
    mc5.metric("MAX 하락일", str(stats["mdd_date"].date()))
    mc6.metric("평균 고점대비 하락률", format_pct(stats["avg_dd"]))
    mc7.metric("전고점 회복 필요 상승률", format_pct(stats["required_return_to_ath"]))

    render_context_summary(stats)
    render_price_threshold_cards(stats, bundle.currency)

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(drawdown_chart(df, f"{section_prefix}MDD 차트"), use_container_width=True)
    with right:
        st.plotly_chart(bucket_bar_chart(bucket_df, f"{section_prefix}MDD 구간 비중"), use_container_width=True)

    st.plotly_chart(price_and_high_chart(df, f"{section_prefix}현재가 vs 과거 최고가"), use_container_width=True)

    if aligned_main is not None and aligned_bench is not None and benchmark_bundle is not None:
        st.plotly_chart(
            indexed_return_chart(
                aligned_main,
                bundle.display_name,
                aligned_bench,
                benchmark_bundle.display_name,
                title=f"{section_prefix}누적 수익률 차트 (시작점 100, 벤치마크 포함)",
            ),
            use_container_width=True,
        )
    else:
        st.plotly_chart(
            indexed_return_chart(df, bundle.display_name, title=f"{section_prefix}누적 수익률 차트 (시작점 100)"),
            use_container_width=True,
        )

    st.markdown("#### 백테스트 결과표")
    st.dataframe(style_metric_table(merged_metric_df), use_container_width=True, hide_index=True)
    st.caption("UWP(underwaterperiod)는 전체 거래일 중 전고점 아래(DD < 0)에 있었던 날짜 비중으로 계산합니다. 값이 낮을수록 더 자주 신고가 부근에 머물렀다는 뜻입니다.")

    st.markdown("#### MDD 구간별 통계")
    st.dataframe(styled_bucket_table(bucket_df), use_container_width=True, hide_index=True)

    st.markdown("#### 낙폭 사건 로그")
    event_log = detect_drawdown_events(df, threshold=event_threshold)
    if event_log.empty:
        st.info(f"선택한 기간에는 {event_threshold:.0%} 이하로 내려간 사건이 없습니다.")
    else:
        st.dataframe(style_event_log(event_log, bundle.currency), use_container_width=True, hide_index=True)
        st.caption(
            f"사건 로그는 낙폭이 처음 {event_threshold:.0%} 이하로 진입한 시점부터, 다시 전고점을 회복할 때까지를 한 사건으로 묶어 보여줍니다."
        )

    with st.expander("원시 데이터 보기"):
        raw_view = df.copy().reset_index().rename(columns={"index": "Date"})
        raw_view["Date"] = raw_view["Date"].dt.date.astype(str)
        st.dataframe(raw_view, use_container_width=True, hide_index=True)
        csv = raw_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            f"{section_prefix}CSV 다운로드",
            csv,
            file_name=f"{section_prefix.replace(' ', '_').lower()}mdd_raw_data.csv",
            mime="text/csv",
        )


# -----------------------------
# 화면 렌더링
# -----------------------------
with st.sidebar:
    st.header("입력")
    ticker_input = st.text_input(
        "종목 티커 / 코드",
        value="SPY",
        help="예: SPY, QQQ, BTCUSD, GCW00, 005930, 069500, 005930.KS",
    )
    period_mode = st.selectbox("기간 선택", ["전체", "최근 20년", "최근 10년", "최근 5년", "최근 3년", "직접 입력"], index=0)
    start_text = st.text_input("시작일 (직접 입력용)", value="", placeholder="YYYY-MM-DD")
    end_text = st.text_input("종료일 (직접 입력용)", value="", placeholder="YYYY-MM-DD")
    threshold_step = st.selectbox("MDD 구간 간격", [1, 2, 5, 10], index=2)
    max_bucket = st.selectbox("최대 하락률 구간", [50, 60, 70, 80, 90, 100], index=5)
    compare_enabled = st.checkbox("벤치마크 비교 사용", value=False)
    benchmark_input = st.text_input("벤치마크 티커 / 코드", value="SPY", disabled=not compare_enabled)
    event_threshold_pct = st.selectbox("낙폭 사건 로그 기준", [-5, -10, -15, -20, -30, -40, -50], index=1)
    run = st.button("조회", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(
        "- 기간 프리셋을 선택하면 시작일/종료일을 자동 계산합니다.\n"
        "- '직접 입력'일 때만 날짜 입력값을 사용합니다.\n"
        "- 한국 6자리 코드는 FDR 우선, 실패 시 Yahoo .KS/.KQ 순으로 조회합니다.\n"
        "- 해외자산이 USD 기준이면 원화 환산 탭을 같이 보여줍니다.\n"
        "- 벤치마크 비교는 누적수익률 차트와 백테스트 결과표에 함께 반영됩니다.\n- UWP는 전체 거래일 중 전고점 아래에 있었던 날짜 비중입니다."
    )


render_help_guide()

if run:
    try:
        with st.spinner("가격 데이터를 불러오는 중입니다..."):
            bundle = fetch_price_data(ticker_input)

            preset_note = None
            if period_mode == "직접 입력":
                start_dt = parse_optional_date(start_text)
                end_dt = parse_optional_date(end_text)
            else:
                start_dt, end_dt, preset_note = apply_period_preset(bundle.series, period_mode)

            restricted_price, actual_start, actual_end, notes = restrict_date_range(bundle.series, start_dt, end_dt)
            base_df = build_price_frame(restricted_price)
            base_stats = summarize_stats(base_df)
            thresholds = build_thresholds(step_pct=threshold_step, max_pct=max_bucket)

            benchmark_bundle = None
            benchmark_df = None
            if compare_enabled and benchmark_input.strip() and normalize_input_ticker(benchmark_input) != normalize_input_ticker(ticker_input):
                benchmark_bundle = fetch_price_data(benchmark_input)
                bench_series, _, _, _ = restrict_date_range(benchmark_bundle.series, actual_start, actual_end)
                benchmark_df = build_price_frame(bench_series)

        st.success("조회가 완료되었습니다.")

        info1, info2, info3, info4 = st.columns([1.8, 1.1, 1.1, 1.6])
        info1.metric("종목", bundle.display_name)
        info2.metric("소스", bundle.source)
        info3.metric("기준 통화", bundle.currency)
        info4.metric("실제 적용 기간", f"{actual_start.date()} ~ {actual_end.date()}")

        if preset_note:
            st.info(preset_note)
        if notes:
            for note in notes:
                st.info(note)

        st.caption(
            f"사용된 심볼: `{bundle.symbol_used}` · 전체 가능 기간: {bundle.series.index.min().date()} ~ {bundle.series.index.max().date()}"
        )
        if benchmark_bundle is not None:
            st.caption(
                f"벤치마크: `{benchmark_bundle.symbol_used}` · {benchmark_bundle.display_name} · 전체 가능 기간: {benchmark_bundle.series.index.min().date()} ~ {benchmark_bundle.series.index.max().date()}"
            )

        tabs = ["원래 통화"]
        show_krw = bundle.currency.upper() == "USD"
        if show_krw:
            tabs.append("원화 환산")
        tab_objs = st.tabs(tabs)

        event_threshold = event_threshold_pct / 100.0

        with tab_objs[0]:
            render_main_block(
                df=base_df,
                stats=base_stats,
                bundle=bundle,
                thresholds=thresholds,
                benchmark_bundle=benchmark_bundle,
                benchmark_df=benchmark_df,
                event_threshold=event_threshold,
                section_prefix="",
            )

        if show_krw:
            with tab_objs[1]:
                with st.spinner("USD/KRW 환율을 불러오는 중입니다..."):
                    usdkrw_full = fetch_usdkrw_series()

                fx_start = usdkrw_full.index.min().normalize()
                fx_end = usdkrw_full.index.max().normalize()
                overlap_start = max(actual_start, fx_start)
                overlap_end = min(actual_end, fx_end)

                if overlap_start > overlap_end:
                    st.error("선택한 기간과 환율 데이터 기간이 겹치지 않아 원화 환산을 계산할 수 없습니다.")
                else:
                    if overlap_start > actual_start:
                        st.warning(f"원화 환산은 환율 데이터가 있는 기간만 계산했습니다: {overlap_start.date()} ~ {overlap_end.date()}")

                    price_overlap = restricted_price.loc[
                        (restricted_price.index.normalize() >= overlap_start)
                        & (restricted_price.index.normalize() <= overlap_end)
                    ]
                    usdkrw_overlap = usdkrw_full.loc[
                        (usdkrw_full.index.normalize() >= overlap_start)
                        & (usdkrw_full.index.normalize() <= overlap_end)
                    ]
                    price_krw = convert_usd_to_krw(price_overlap, usdkrw_overlap)
                    krw_df = build_price_frame(price_krw)
                    krw_stats = summarize_stats(krw_df)

                    bench_bundle_krw = None
                    bench_df_krw = None
                    if benchmark_bundle is not None and benchmark_df is not None:
                        if benchmark_bundle.currency.upper() == "USD":
                            benchmark_overlap = benchmark_bundle.series.loc[
                                (benchmark_bundle.series.index.normalize() >= overlap_start)
                                & (benchmark_bundle.series.index.normalize() <= overlap_end)
                            ]
                            if not benchmark_overlap.empty:
                                benchmark_price_krw = convert_usd_to_krw(benchmark_overlap, usdkrw_overlap)
                                bench_df_krw = build_price_frame(benchmark_price_krw)
                                bench_bundle_krw = PriceBundle(
                                    series=benchmark_price_krw,
                                    display_name=f"{benchmark_bundle.display_name} [KRW 환산]",
                                    symbol_used=benchmark_bundle.symbol_used,
                                    source=benchmark_bundle.source,
                                    currency="KRW",
                                )
                        elif benchmark_bundle.currency.upper() == "KRW":
                            benchmark_overlap = benchmark_bundle.series.loc[
                                (benchmark_bundle.series.index.normalize() >= overlap_start)
                                & (benchmark_bundle.series.index.normalize() <= overlap_end)
                            ]
                            if not benchmark_overlap.empty:
                                bench_df_krw = build_price_frame(benchmark_overlap)
                                bench_bundle_krw = benchmark_bundle

                    krw_bundle = PriceBundle(
                        series=price_krw,
                        display_name=f"{bundle.display_name} [KRW 환산]",
                        symbol_used=bundle.symbol_used,
                        source=bundle.source,
                        currency="KRW",
                    )

                    render_main_block(
                        df=krw_df,
                        stats=krw_stats,
                        bundle=krw_bundle,
                        thresholds=thresholds,
                        benchmark_bundle=bench_bundle_krw,
                        benchmark_df=bench_df_krw,
                        event_threshold=event_threshold,
                        section_prefix="원화 기준 ",
                    )

                    with st.expander("환율 및 원화 환산 데이터 보기"):
                        fx_aligned = usdkrw_overlap.reindex(price_overlap.index.union(usdkrw_overlap.index)).sort_index().ffill().reindex(price_overlap.index)
                        krw_view = pd.DataFrame(
                            {
                                "Date": price_overlap.index.date.astype(str),
                                "Close(USD)": price_overlap.values,
                                "USD/KRW": fx_aligned.values,
                                "Close(KRW)": price_krw.values,
                            }
                        )
                        st.dataframe(krw_view, use_container_width=True, hide_index=True)
                        csv2 = krw_view.to_csv(index=False).encode("utf-8-sig")
                        st.download_button("원화 환산 CSV 다운로드", csv2, file_name="mdd_krw_data.csv", mime="text/csv")

        st.markdown("---")
        st.caption(
            "참고: 서로 다른 데이터 소스는 휴장일, 수정주가 반영 방식, 환율 제공 시작일이 조금씩 다를 수 있습니다. "
            "그래서 시트와 숫자가 100% 동일하지는 않을 수 있지만, 구조와 계산 방식은 최대한 비슷하게 맞췄습니다."
        )

    except Exception as e:
        st.error(str(e))
else:
    st.info("왼쪽에서 종목 티커/코드를 입력하고 **조회**를 눌러주세요.")
