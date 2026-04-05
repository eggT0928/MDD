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
except Exception:  # pragma: no cover
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
    "구글 스프레드시트 대신 웹에서 바로 조회할 수 있도록 만든 Streamlit 버전입니다. "
    "종목 티커/코드를 넣으면 최대기간 자동 조회, MDD 차트, 구간별 누적비율, 회복 통계까지 한 번에 보여줍니다."
)


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
    """사용자 입력을 데이터 소스가 이해하기 쉬운 형태로 정리한다."""
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

    # 1) 한국 6자리 숫자 티커는 FDR 우선
    if is_krx_numeric_ticker(symbol):
        for candidate in [symbol.replace("A", "")]:
            try:
                return fetch_with_fdr(candidate)
            except Exception as e:
                errors.append(f"FDR({candidate}): {e}")

        # FDR 실패 시 야후 .KS / .KQ 폴백
        for suffix in [".KS", ".KQ"]:
            try:
                return fetch_with_yfinance(symbol.replace("A", "") + suffix)
            except Exception as e:
                errors.append(f"Yahoo({symbol}{suffix}): {e}")

    # 2) 그 외는 Yahoo 우선
    for candidate in [symbol]:
        try:
            return fetch_with_yfinance(candidate)
        except Exception as e:
            errors.append(f"Yahoo({candidate}): {e}")

    # 3) 혹시 KRX 입력인데 .KS/.KQ 없이 넣은 경우 한 번 더 시도
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

    for candidate in ["KRW=X"]:
        try:
            bundle = fetch_with_yfinance(candidate)
            return bundle.series
        except Exception as e:
            errors.append(f"Yahoo {candidate}: {e}")

    raise RuntimeError(" | ".join(errors))


# -----------------------------
# 계산 함수
# -----------------------------
def restrict_date_range(
    s: pd.Series,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Tuple[pd.Series, pd.Timestamp, pd.Timestamp, List[str]]:
    """사용자가 날짜를 비우면 전체 가능 기간을 자동 사용한다."""
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



def build_price_frame(price: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Close": price.dropna().astype(float)})
    df["High"] = df["Close"].cummax()
    df["Drawdown"] = df["Close"] / df["High"] - 1.0
    df["CumReturn"] = df["Close"] / df["Close"].iloc[0] - 1.0
    df["DaysToNextATH"] = days_to_next_ath(df["Drawdown"])
    return df



def days_to_next_ath(dd: pd.Series) -> pd.Series:
    """각 시점에서 다음 전고점(DD=0) 회복까지 남은 거래일 수를 계산한다."""
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



def summarize_stats(df: pd.DataFrame) -> Dict[str, object]:
    current_price = float(df["Close"].iloc[-1])
    ath = float(df["High"].max())
    current_dd = float(df["Drawdown"].iloc[-1])
    mdd = float(df["Drawdown"].min())
    mdd_date = pd.Timestamp(df["Drawdown"].idxmin())
    avg_dd = float(df["Drawdown"].mean())
    total_days = int(len(df))

    return {
        "current_price": current_price,
        "ath": ath,
        "current_dd": current_dd,
        "mdd": mdd,
        "mdd_date": mdd_date,
        "avg_dd": avg_dd,
        "total_days": total_days,
    }



def build_thresholds(step_pct: int = 5, max_pct: int = 100) -> List[float]:
    thresholds = [0.0]
    for p in range(step_pct, max_pct + step_pct, step_pct):
        thresholds.append(-p / 100)
    return thresholds



def bucket_label(threshold: float) -> str:
    if threshold == 0:
        return "0% (ATH)"
    return f"{int(threshold * 100)}%"



def build_bucket_table(df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    dd = df["Drawdown"].copy()
    total_days = len(dd)

    # 누적 비율: 해당 기준 이상(예: -10% 이내)인 날짜의 비중
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

    # 구간 진입/회복 통계: 각 구간으로 "처음 들어간" 사건 기준
    entry_counts, recovered_counts, recovery_probs, avg_recovery_days = compute_segment_recovery_stats(dd, thresholds)

    out = pd.DataFrame(
        {
            "MDD": thresholds,
            "누적 비율": cumulative_ratios,
            "조건일": condition_days_list,
            "개장일": total_days,
            "구간 비중": segment_weights,
            "진입사건": entry_counts,
            "이후 회복": recovered_counts,
            "회복확률": recovery_probs,
            "평균 거래일수": avg_recovery_days,
        }
    )
    return out



def compute_segment_recovery_stats(dd: pd.Series, thresholds: List[float]):
    """
    구간별 진입 사건을 계산한다.

    예시
    - -5% 행: 0% ~ -5% 구간으로 처음 내려온 사건 수
    - -10% 행: -5% ~ -10% 구간으로 더 깊게 내려온 사건 수

    이후 회복은 해당 진입 시점 이후 DD가 0으로 회복한 경우를 센다.
    평균 거래일수는 회복한 케이스만 평균낸다.
    """
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

    # 이후 첫 DD=0 복귀 시점을 빠르게 찾기 위한 배열
    next_zero = np.full(n, -1, dtype=int)
    next_idx = -1
    for i in range(n - 1, -1, -1):
        if np.isclose(values[i], 0.0):
            next_idx = i
        next_zero[i] = next_idx

    prev_seg = segments[0]
    for i in range(1, n):
        cur_seg = segments[i]

        # 더 깊은 구간으로 이동할 때만 해당 구간 진입으로 본다.
        if cur_seg > prev_seg and cur_seg > 0:
            entry_counts[cur_seg] += 1
            zero_idx = next_zero[i]
            if zero_idx != -1 and zero_idx > i:
                recovered_counts[cur_seg] += 1
                recovery_days_store[cur_seg].append(int(zero_idx - i))

        prev_seg = cur_seg

    recovery_probs = []
    avg_recovery_days = []
    for i in range(len(thresholds)):
        entry = entry_counts[i]
        recovered = recovered_counts[i]
        prob = np.nan if entry == 0 else recovered / entry
        avg_days = np.nan if len(recovery_days_store[i]) == 0 else float(np.mean(recovery_days_store[i]))
        recovery_probs.append(prob)
        avg_recovery_days.append(avg_days)

    return entry_counts, recovered_counts, recovery_probs, avg_recovery_days



def convert_usd_to_krw(price_usd: pd.Series, usdkrw: pd.Series) -> pd.Series:
    fx = usdkrw.copy().sort_index().astype(float)
    fx_aligned = fx.reindex(price_usd.index.union(fx.index)).sort_index().ffill().reindex(price_usd.index)
    if fx_aligned.isna().all():
        raise ValueError("환율 데이터를 가격 데이터 날짜에 맞춰 정렬할 수 없습니다.")
    return price_usd * fx_aligned


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
    for col in ["누적 비율", "구간 비중", "회복확률"]:
        view[col] = view[col].map(lambda x: "-" if pd.isna(x) else f"{x:.1%}")
    for col in ["조건일", "개장일", "진입사건", "이후 회복"]:
        view[col] = view[col].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
    view["평균 거래일수"] = view["평균 거래일수"].map(lambda x: "-" if pd.isna(x) else f"{x:,.1f}")
    return view


# -----------------------------
# 화면 렌더링
# -----------------------------
with st.sidebar:
    st.header("입력")
    ticker_input = st.text_input("종목 티커 / 코드", value="SPY", help="예: SPY, QQQ, BTCUSD, GCW00, 005930, 069500, 005930.KS")
    start_text = st.text_input("시작일 (선택)", value="", placeholder="YYYY-MM-DD")
    end_text = st.text_input("종료일 (선택)", value="", placeholder="YYYY-MM-DD")
    threshold_step = st.selectbox("MDD 구간 간격", [1, 2, 5, 10], index=2)
    max_bucket = st.selectbox("최대 하락률 구간", [50, 60, 70, 80, 90, 100], index=5)
    run = st.button("조회", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(
        "- 시작일/종료일을 비워두면 해당 종목의 조회 가능한 최대 기간을 자동 사용합니다.\n"
        "- 한국 6자리 코드(예: 005930, 069500)는 FinanceDataReader 우선, 실패 시 Yahoo .KS/.KQ 순으로 조회합니다.\n"
        "- 해외자산이 USD 기준이면 원화 환산 탭을 같이 보여줍니다."
    )


if run:
    try:
        start_dt = parse_optional_date(start_text)
        end_dt = parse_optional_date(end_text)

        with st.spinner("가격 데이터를 불러오는 중입니다..."):
            bundle = fetch_price_data(ticker_input)
            restricted_price, actual_start, actual_end, notes = restrict_date_range(bundle.series, start_dt, end_dt)
            base_df = build_price_frame(restricted_price)
            base_stats = summarize_stats(base_df)
            thresholds = build_thresholds(step_pct=threshold_step, max_pct=max_bucket)
            base_bucket = build_bucket_table(base_df, thresholds)

        st.success("조회가 완료되었습니다.")

        info1, info2, info3, info4 = st.columns([1.6, 1.2, 1.2, 1.5])
        info1.metric("종목", bundle.display_name)
        info2.metric("소스", bundle.source)
        info3.metric("기준 통화", bundle.currency)
        info4.metric("실제 적용 기간", f"{actual_start.date()} ~ {actual_end.date()}")

        if notes:
            for note in notes:
                st.info(note)

        st.caption(
            f"사용된 심볼: `{bundle.symbol_used}` · 전체 가능 기간: {bundle.series.index.min().date()} ~ {bundle.series.index.max().date()}"
        )

        tabs = ["원래 통화"]
        show_krw = bundle.currency.upper() == "USD"
        if show_krw:
            tabs.append("원화 환산")
        tab_objs = st.tabs(tabs)

        # -----------------------------
        # 1) 원래 통화 탭
        # -----------------------------
        with tab_objs[0]:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("현재가", format_price(base_stats["current_price"], bundle.currency))
            c2.metric("과거 최고가", format_price(base_stats["ath"], bundle.currency))
            c3.metric("고점대비 하락률", f"{base_stats['current_dd']:.2%}")
            c4.metric("MDD", f"{base_stats['mdd']:.2%}")
            c5.metric("MAX 하락일", str(base_stats["mdd_date"].date()))
            c6.metric("평균 고점대비 하락률", f"{base_stats['avg_dd']:.2%}")

            st.markdown("#### 기준 가격")
            k1, k2, k3 = st.columns(3)
            for col, th in zip([k1, k2, k3], [-0.10, -0.20, -0.30]):
                col.metric(f"기준 {int(abs(th)*100)}%", format_price(base_stats["ath"] * (1 + th), bundle.currency))

            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(drawdown_chart(base_df, "MDD 차트"), use_container_width=True)
            with right:
                st.plotly_chart(bucket_bar_chart(base_bucket, "MDD 구간 비중"), use_container_width=True)

            st.plotly_chart(price_and_high_chart(base_df, "현재가 vs 과거 최고가"), use_container_width=True)

            st.markdown("#### MDD 구간별 통계")
            st.dataframe(styled_bucket_table(base_bucket), use_container_width=True, hide_index=True)

            with st.expander("원시 데이터 보기"):
                raw_view = base_df.copy().reset_index().rename(columns={"index": "Date"})
                raw_view["Date"] = raw_view["Date"].dt.date.astype(str)
                st.dataframe(raw_view, use_container_width=True, hide_index=True)
                csv = raw_view.to_csv(index=False).encode("utf-8-sig")
                st.download_button("CSV 다운로드", csv, file_name="mdd_raw_data.csv", mime="text/csv")

        # -----------------------------
        # 2) 원화 환산 탭
        # -----------------------------
        if show_krw:
            with tab_objs[1]:
                with st.spinner("USD/KRW 환율을 불러오는 중입니다..."):
                    usdkrw_full = fetch_usdkrw_series()

                fx_start = usdkrw_full.index.min().normalize()
                fx_end = usdkrw_full.index.max().normalize()

                # 원화 환산은 자산과 환율이 겹치는 기간만 사용
                overlap_start = max(actual_start, fx_start)
                overlap_end = min(actual_end, fx_end)
                if overlap_start > overlap_end:
                    st.error("선택한 기간과 환율 데이터 기간이 겹치지 않아 원화 환산을 계산할 수 없습니다.")
                else:
                    if overlap_start > actual_start:
                        st.warning(
                            f"원화 환산은 환율 데이터가 있는 기간만 계산했습니다: {overlap_start.date()} ~ {overlap_end.date()}"
                        )

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
                    krw_bucket = build_bucket_table(krw_df, thresholds)

                    kc1, kc2, kc3, kc4, kc5, kc6 = st.columns(6)
                    kc1.metric("현재가", format_price(krw_stats["current_price"], "KRW"))
                    kc2.metric("과거 최고가", format_price(krw_stats["ath"], "KRW"))
                    kc3.metric("고점대비 하락률", f"{krw_stats['current_dd']:.2%}")
                    kc4.metric("MDD", f"{krw_stats['mdd']:.2%}")
                    kc5.metric("MAX 하락일", str(krw_stats["mdd_date"].date()))
                    kc6.metric("평균 고점대비 하락률", f"{krw_stats['avg_dd']:.2%}")

                    st.markdown("#### 기준 가격")
                    kk1, kk2, kk3 = st.columns(3)
                    for col, th in zip([kk1, kk2, kk3], [-0.10, -0.20, -0.30]):
                        col.metric(f"기준 {int(abs(th)*100)}%", format_price(krw_stats["ath"] * (1 + th), "KRW"))

                    left2, right2 = st.columns([2, 1])
                    with left2:
                        st.plotly_chart(drawdown_chart(krw_df, "원화 기준 MDD 차트"), use_container_width=True)
                    with right2:
                        st.plotly_chart(bucket_bar_chart(krw_bucket, "원화 기준 MDD 구간 비중"), use_container_width=True)

                    st.plotly_chart(price_and_high_chart(krw_df, "원화 기준 현재가 vs 과거 최고가"), use_container_width=True)

                    st.markdown("#### 원화 기준 MDD 구간별 통계")
                    st.dataframe(styled_bucket_table(krw_bucket), use_container_width=True, hide_index=True)

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
