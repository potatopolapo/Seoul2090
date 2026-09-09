import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="centered")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
BASE_YEAR_LIMIT = 2025      # 기준 기간: 이 해까지만 사용
MIN_OBS_DAYS = 300          # 연간 최소 관측일수


@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data
def build_yearly(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("연도").agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count"),
    ).reset_index()

    # 기준 기간(2025년까지)과 관측일수(300일 이상) 조건 적용
    filtered = grouped[
        (grouped["연도"] <= BASE_YEAR_LIMIT) & (grouped["관측일수"] >= MIN_OBS_DAYS)
    ].sort_values("연도").reset_index(drop=True)

    return filtered


st.title("🌡️ 서울 기온 예측기")
st.caption("서울 연평균기온 데이터를 이용한 선형 회귀 기반 기온 예측 앱")

with st.spinner("데이터를 불러오는 중입니다..."):
    raw_df = load_data(DATA_URL)
    yearly_df = build_yearly(raw_df)

if len(yearly_df) < 2:
    st.error("회귀 분석에 필요한 데이터가 충분하지 않습니다.")
    st.stop()

years = yearly_df["연도"].values.astype(float)
temps = yearly_df["평균기온"].values.astype(float)

# 상관계수
corr = float(np.corrcoef(years, temps)[0, 1])

# 선형 회귀 (최소제곱법)
slope, intercept = np.polyfit(years, temps, 1)


def predict(year: float) -> float:
    return slope * year + intercept


n_years = len(yearly_df)
start_year = int(yearly_df["연도"].min())
end_year = int(yearly_df["연도"].max())

# 100년당 기온 상승률 (전체 기간)
slope_per_100 = slope * 100

# 최근 20년만 이용한 회귀
RECENT_WINDOW = 20
recent_df = yearly_df[yearly_df["연도"] >= end_year - (RECENT_WINDOW - 1)]
has_recent = len(recent_df) >= 2

if has_recent:
    recent_years = recent_df["연도"].values.astype(float)
    recent_temps = recent_df["평균기온"].values.astype(float)
    slope_recent, intercept_recent = np.polyfit(recent_years, recent_temps, 1)
    slope_recent_per_100 = slope_recent * 100
    recent_start = int(recent_df["연도"].min())
    recent_end = int(recent_df["연도"].max())
    recent_n = len(recent_df)

st.markdown(
    f"**회귀 직선에 사용된 연도 수:** {n_years}개&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"**시작 연도:** {start_year}년&nbsp;&nbsp;|&nbsp;&nbsp;**끝 연도:** {end_year}년"
)
st.markdown(f"**상관계수 (r):** {corr:.4f}")

# ---- 100년당 상승률 비교 ----
st.subheader("100년당 기온 상승률 비교")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 15px; border-radius:10px; background-color:#f0f2f6;">
            <div style="font-size:16px; color:gray;">전체 기간 ({start_year}~{end_year}, {n_years}개 연도)</div>
            <div style="font-size:44px; font-weight:bold; color:#1f77b4;">{slope_per_100:+.2f} °C</div>
            <div style="font-size:14px; color:gray;">/ 100년</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    if has_recent:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 15px; border-radius:10px; background-color:#fdf0f0;">
                <div style="font-size:16px; color:gray;">최근 {RECENT_WINDOW}년 ({recent_start}~{recent_end}, {recent_n}개 연도)</div>
                <div style="font-size:44px; font-weight:bold; color:#d62728;">{slope_recent_per_100:+.2f} °C</div>
                <div style="font-size:14px; color:gray;">/ 100년</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(f"최근 {RECENT_WINDOW}년 구간의 데이터가 부족합니다.")

# ---- 산점도 + 회귀 직선 ----
line_x = np.linspace(min(years.min(), 1900), max(years.max(), 2100), 200)
line_y = predict(line_x)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=years, y=temps,
    mode="markers",
    name="연평균기온",
    marker=dict(color="#1f77b4", size=8),
))
fig.add_trace(go.Scatter(
    x=line_x, y=line_y,
    mode="lines",
    name="회귀 직선 (전체 기간)",
    line=dict(color="#1f77b4", width=2, dash="dash"),
))
if has_recent:
    recent_line_y = slope_recent * line_x + intercept_recent
    fig.add_trace(go.Scatter(
        x=line_x, y=recent_line_y,
        mode="lines",
        name=f"회귀 직선 (최근 {RECENT_WINDOW}년)",
        line=dict(color="#d62728", width=2, dash="dot"),
    ))
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- 슬라이더로 특정 연도 예측 ----
st.subheader("연도별 예상 기온")
selected_year = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2025, step=1)
predicted_temp = predict(selected_year)

st.markdown(
    f"""
    <div style="text-align:center; padding: 30px 0;">
        <div style="font-size:22px; color:gray;">{selected_year}년 예상 평균기온</div>
        <div style="font-size:64px; font-weight:bold; color:#d62728;">{predicted_temp:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("사용된 연도별 데이터 보기"):
    st.dataframe(
        yearly_df.rename(columns={"관측일수": "관측일수(일)"}),
        use_container_width=True,
    )
