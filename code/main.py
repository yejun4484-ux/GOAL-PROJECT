
import io
from urllib.parse import unquote

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. 기본 설정
# =========================================================
st.set_page_config(
    page_title="축구 전반전 득점과 최종 득점의 관계",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ 전반전 득점으로 최종 득점을 예측할 수 있을까?")
st.caption(
    "대한민국 남자 축구 국가대표팀의 경기 데이터를 활용한 선형회귀 탐구"
)

DATA_URL = (
    "https://raw.githubusercontent.com/yejun4484-ux/GOAL-PROJECT/"
    "refs/heads/main/data/"
    "%25EA%25BF%2588%25EB%2591%2590%25EB%25A0%2588_"
    "%25EB%258D%25B0%25EC%259D%25B4%25ED%2584%25B0_"
    "%25EC%25B6%2595%25EA%25B5%25AC.csv"
)

X_COL = "한국_전반득점"
Y_COL = "한국_최종득점"


# =========================================================
# 2. 데이터 불러오기
# =========================================================
@st.cache_data(ttl=300)
def load_data():
    candidates = [DATA_URL, unquote(DATA_URL)]
    errors = []

    for url in candidates:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()

            for encoding in ("utf-8-sig", "utf-8", "cp949"):
                try:
                    df = pd.read_csv(
                        io.BytesIO(response.content),
                        encoding=encoding,
                    )
                    df.columns = [str(c).strip() for c in df.columns]
                    return df, url, None

                except (
                    UnicodeDecodeError,
                    pd.errors.ParserError,
                    ValueError,
                ) as exc:
                    errors.append(f"{encoding}: {exc}")

        except Exception as exc:
            errors.append(f"{url}: {exc}")

    return None, None, "\n".join(errors)


df_raw, loaded_url, load_error = load_data()

if df_raw is None:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("GitHub 저장소의 CSV 주소와 공개 여부를 확인해 주세요.")
    with st.expander("오류 내용"):
        st.code(load_error or "알 수 없는 오류")
    st.stop()


# =========================================================
# 3. 데이터 전처리
# =========================================================
required_columns = [X_COL, Y_COL]
missing_columns = [
    col for col in required_columns
    if col not in df_raw.columns
]

if missing_columns:
    st.error(f"필요한 열이 데이터에 없습니다: {missing_columns}")
    st.write("현재 CSV의 열 이름:")
    st.write(df_raw.columns.tolist())
    st.stop()

df = df_raw.copy()

# 숫자형으로 변환
for col in [
    "연도",
    "한국_전반득점",
    "상대_전반득점",
    "한국_최종득점",
    "상대_최종득점",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 날짜형으로 변환
if "날짜" in df.columns:
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

# 선형회귀에 필요한 두 열이 모두 있는 경기만 사용
analysis_df = df.dropna(subset=[X_COL, Y_COL]).copy()

# 득점은 정수형으로 표시
analysis_df[X_COL] = analysis_df[X_COL].astype(int)
analysis_df[Y_COL] = analysis_df[Y_COL].astype(int)

if len(analysis_df) < 5:
    st.error("선형회귀를 수행하기 위한 데이터가 충분하지 않습니다.")
    st.stop()


# =========================================================
# 4. 선형회귀 모델 학습
# =========================================================
X = analysis_df[[X_COL]]
y = analysis_df[Y_COL]

# 학습 데이터와 평가 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

# 예측
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# 평가 지표
r2 = r2_score(y_test, test_pred)
mae = mean_absolute_error(y_test, test_pred)
rmse = mean_squared_error(y_test, test_pred) ** 0.5

slope = float(model.coef_[0])
intercept = float(model.intercept_)

# 전체 데이터 예측값
analysis_df["예측_최종득점"] = model.predict(X)
analysis_df["예측_오차"] = (
    analysis_df[Y_COL] - analysis_df["예측_최종득점"]
)


# =========================================================
# 5. 사이드바
# =========================================================
with st.sidebar:
    st.title("⚽ 탐구 메뉴")
    st.markdown(
        """
        1. 탐구 주제
        2. 데이터 탐색
        3. 데이터 시각화
        4. 선형회귀 모델
        5. 모델 평가
        6. 결과 해석
        """
    )

    st.divider()

    st.write("**분석 데이터**")
    st.write(f"- 전체 경기: {len(df_raw)}개")
    st.write(f"- 분석에 사용한 경기: {len(analysis_df)}개")
    st.write(f"- 학습 데이터: {len(X_train)}개")
    st.write(f"- 평가 데이터: {len(X_test)}개")

    st.divider()

    if st.button("데이터 새로고침", use_container_width=True):
        load_data.clear()
        st.rerun()


# =========================================================
# 6. 메뉴 구성
# =========================================================
tabs = st.tabs([
    "① 탐구 주제",
    "② 데이터 탐색",
    "③ 데이터 시각화",
    "④ 선형회귀 모델",
    "⑤ 모델 평가",
    "⑥ 결과 해석",
])


# =========================================================
# 탭 1. 탐구 주제
# =========================================================
with tabs[0]:
    st.header("탐구 주제")

    st.subheader("전반전 득점으로 최종 득점을 예측할 수 있을까?")

    st.write(
        """
        축구 경기에서 전반전에 득점을 많이 하면 최종 득점도 많아질까?

        이 탐구에서는 대한민국 남자 축구 국가대표팀의 경기 기록을
        활용하여 전반전 득점과 최종 득점 사이의 관계를 알아본다.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("분석 대상 경기", f"{len(analysis_df)}경기")
        st.info(
            """
            **독립변수(X)**

            한국의 전반전 득점

            모델이 예측에 사용하는 변수
            """
        )

    with col2:
        st.metric("분석 기간", "2021~2025년")
        st.success(
            """
            **종속변수(Y)**

            한국의 최종 득점

            모델이 예측하려는 변수
            """
        )

    st.subheader("분석 방법")

    st.markdown(
        """
        1. 경기 데이터를 불러온다.
        2. 전반전 득점과 최종 득점을 숫자형으로 정리한다.
        3. 두 변수의 관계를 산점도로 확인한다.
        4. 선형회귀 모델을 학습한다.
        5. 회귀식과 평가 지표를 확인한다.
        6. 분석 결과와 한계점을 해석한다.
        """
    )

    st.subheader("선형회귀란?")
    st.write(
        "선형회귀는 독립변수와 종속변수 사이의 관계를 "
        "직선으로 나타내고, 그 직선을 이용해 값을 예측하는 방법이다."
    )

    st.latex(r"\hat{y} = b_0 + b_1x")

    st.write(
        "- x: 한국의 전반전 득점\n"
        "- ŷ: 모델이 예측한 한국의 최종 득점\n"
        "- b₀: 절편\n"
        "- b₁: 기울기"
    )


# =========================================================
# 탭 2. 데이터 탐색
# =========================================================
with tabs[1]:
    st.header("데이터 탐색")

    st.write(
        "직접 수집한 대한민국 남자 축구 국가대표팀의 "
        "경기 데이터를 살펴본다."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("전체 행", f"{df_raw.shape[0]:,}개")
    c2.metric("전체 열", f"{df_raw.shape[1]:,}개")
    c3.metric(
        "전체 결측치",
        f"{int(df_raw.isna().sum().sum()):,}개"
    )
    c4.metric(
        "회귀 분석 경기",
        f"{len(analysis_df):,}개"
    )

    st.subheader("데이터 미리보기")
    st.dataframe(
        df_raw,
        use_container_width=True,
        height=350,
    )

    st.subheader("열별 결측치 현황")

    missing_table = pd.DataFrame({
        "열 이름": df_raw.columns,
        "결측치 개수": [
            int(df_raw[col].isna().sum())
            for col in df_raw.columns
        ],
        "결측치 비율(%)": [
            round(df_raw[col].isna().mean() * 100, 2)
            for col in df_raw.columns
        ],
        "자료형": [
            str(df_raw[col].dtype)
            for col in df_raw.columns
        ],
    })

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("주요 변수의 기초통계")

    stats = analysis_df[[X_COL, Y_COL]].describe().T
    stats = stats.rename(columns={
        "count": "경기 수",
        "mean": "평균",
        "std": "표준편차",
        "min": "최솟값",
        "25%": "1사분위수",
        "50%": "중앙값",
        "75%": "3사분위수",
        "max": "최댓값",
    })

    st.dataframe(
        stats.round(2),
        use_container_width=True,
    )

    st.download_button(
        "분석 데이터 CSV 다운로드",
        data=analysis_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="축구_선형회귀_분석데이터.csv",
        mime="text/csv",
    )


# =========================================================
# 탭 3. 데이터 시각화
# =========================================================
with tabs[2]:
    st.header("데이터 시각화")

    st.write(
        "선형회귀 모델을 만들기 전에 두 변수의 분포와 관계를 "
        "그래프로 확인한다."
    )

    st.subheader("1. 전반전 득점 분포")

    fig_x = px.histogram(
        analysis_df,
        x=X_COL,
        nbins=max(1, analysis_df[X_COL].nunique()),
        title="한국의 전반전 득점 분포",
        labels={X_COL: "전반전 득점", "count": "경기 수"},
        text_auto=True,
    )

    fig_x.update_layout(
        height=400,
        bargap=0.15,
    )

    st.plotly_chart(
        fig_x,
        use_container_width=True,
    )

    st.caption(
        "전반전에 한국이 몇 골을 넣었는지, "
        "각 득점 횟수가 몇 경기에서 나타났는지 확인할 수 있다."
    )

    st.divider()

    st.subheader("2. 최종 득점 분포")

    fig_y = px.histogram(
        analysis_df,
        x=Y_COL,
        nbins=max(1, analysis_df[Y_COL].nunique()),
        title="한국의 최종 득점 분포",
        labels={Y_COL: "최종 득점", "count": "경기 수"},
        text_auto=True,
    )

    fig_y.update_layout(
        height=400,
        bargap=0.15,
    )

    st.plotly_chart(
        fig_y,
        use_container_width=True,
    )

    st.caption(
        "최종 득점이 어떤 값에 집중되어 있는지 확인할 수 있다."
    )

    st.divider()

    st.subheader("3. 전반전 득점과 최종 득점의 관계")

    fig_scatter = px.scatter(
        analysis_df,
        x=X_COL,
        y=Y_COL,
        hover_data=[
            col for col in ["날짜", "상대팀", "대회"]
            if col in analysis_df.columns
        ],
        title="전반전 득점과 최종 득점의 산점도",
        labels={
            X_COL: "한국의 전반전 득점",
            Y_COL: "한국의 최종 득점",
        },
    )

    fig_scatter.update_traces(
        marker=dict(size=11, opacity=0.75)
    )

    fig_scatter.update_layout(height=500)

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
    )

    st.caption(
        "점 하나는 축구 경기 하나를 의미한다. "
        "점들이 오른쪽 위 방향으로 모이는지 살펴보자."
    )


# =========================================================
# 탭 4. 선형회귀 모델
# =========================================================
with tabs[3]:
    st.header("선형회귀 모델")

    st.write(
        "학습 데이터로 선형회귀 모델을 만들고, "
        "전반전 득점과 최종 득점 사이의 직선 관계를 확인한다."
    )

    st.subheader("1. 학습 데이터와 평가 데이터")

    c1, c2 = st.columns(2)

    c1.metric("학습 데이터", f"{len(X_train)}경기")
    c2.metric("평가 데이터", f"{len(X_test)}경기")

    st.write(
        "전체 데이터를 무작위로 학습용 75%, 평가용 25%로 나누었다. "
        "평가 데이터는 모델을 학습할 때 사용하지 않는다."
    )

    st.divider()

    st.subheader("2. 선형회귀 결과")

    st.latex(
        rf"\hat{{y}} = {intercept:.3f} + ({slope:.3f})x"
    )

    c1, c2 = st.columns(2)

    c1.metric("기울기", f"{slope:.3f}")
    c2.metric("절편", f"{intercept:.3f}")

    st.write(
        f"전반전 득점이 1골 증가할 때, "
        f"모델이 예측하는 최종 득점은 평균적으로 "
        f"{slope:.3f}골 변하는 것으로 나타났다."
    )

    st.caption(
        "기울기는 학습 데이터에서 추정한 값이며, "
        "실제 경기에서 전반전 득점이 최종 득점을 변화시킨다는 "
        "인과관계를 증명하는 것은 아니다."
    )

    st.divider()

    st.subheader("3. 회귀 직선 그래프")

    line_x = pd.DataFrame({
        X_COL: [
            float(analysis_df[X_COL].min()),
            float(analysis_df[X_COL].max()),
        ]
    })
    line_y = model.predict(line_x)

    fig_model = go.Figure()

    fig_model.add_trace(
        go.Scatter(
            x=X_train[X_COL],
            y=y_train,
            mode="markers",
            name="학습 데이터",
            marker=dict(size=10),
        )
    )

    fig_model.add_trace(
        go.Scatter(
            x=X_test[X_COL],
            y=y_test,
            mode="markers",
            name="평가 데이터",
            marker=dict(size=11, symbol="diamond"),
        )
    )

    fig_model.add_trace(
        go.Scatter(
            x=line_x[X_COL],
            y=line_y,
            mode="lines",
            name="회귀 직선",
            line=dict(width=3),
        )
    )

    fig_model.update_layout(
        title="전반전 득점과 최종 득점의 선형회귀",
        xaxis_title="한국의 전반전 득점",
        yaxis_title="한국의 최종 득점",
        height=520,
        hovermode="closest",
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True,
    )

    st.caption(
        "회귀 직선은 학습 데이터로 계산했다. "
        "평가 데이터는 모델이 학습하지 않은 경기이므로 "
        "예측 성능을 확인하는 데 사용한다."
    )

    st.divider()

    st.subheader("4. 직접 최종 득점 예측하기")

    max_goal = max(
        10,
        int(analysis_df[X_COL].max()) + 3
    )

    user_goal = st.number_input(
        "한국의 전반전 득점을 입력하세요.",
        min_value=0,
        max_value=max_goal,
        value=1,
        step=1,
    )

    predicted_goal = float(
        model.predict(
            pd.DataFrame({X_COL: [user_goal]})
        )[0]
    )

    st.metric(
        "예측한 한국의 최종 득점",
        f"{predicted_goal:.2f}골",
    )

    st.caption(
        "예측값은 통계 모델의 계산 결과이며, "
        "실제 경기 결과를 보장하지 않는다. "
        "특히 관측된 전반전 득점 범위를 벗어난 입력은 "
        "신뢰하기 어렵다."
    )


# =========================================================
# 탭 5. 모델 평가
# =========================================================
with tabs[4]:
    st.header("모델 평가")

    st.write(
        "학습에 사용하지 않은 평가 데이터로 "
        "모델의 예측 성능을 확인한다."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("R² 결정계수", f"{r2:.3f}")
    c2.metric("MAE 평균 절대 오차", f"{mae:.3f}골")
    c3.metric("RMSE 평균 제곱근 오차", f"{rmse:.3f}골")

    st.subheader("평가 지표의 의미")

    st.markdown(
        """
        - **R² 결정계수:** 모델이 평가 데이터의 최종 득점 변동을
          얼마나 설명하는지 나타낸다. 1에 가까울수록 좋으며,
          음수가 나올 수도 있다.
        - **MAE:** 실제 최종 득점과 예측 득점 사이의
          평균적인 차이를 나타낸다. 0에 가까울수록 좋다.
        - **RMSE:** 예측 오차를 제곱한 뒤 평균을 내고
          다시 제곱근을 구한 값이다. 큰 오차에 더 민감하다.
        """
    )

    st.divider()

    st.subheader("실제 득점과 예측 득점 비교")

    result_df = pd.DataFrame({
        "실제 최종 득점": y_test.values,
        "예측 최종 득점": test_pred,
    })

    fig_compare = px.scatter(
        result_df,
        x="실제 최종 득점",
        y="예측 최종 득점",
        title="실제 득점과 모델 예측 득점",
        labels={
            "실제 최종 득점": "실제 최종 득점",
            "예측 최종 득점": "예측 최종 득점",
        },
    )

    min_value = min(
        result_df["실제 최종 득점"].min(),
        result_df["예측 최종 득점"].min(),
    )
    max_value = max(
        result_df["실제 최종 득점"].max(),
        result_df["예측 최종 득점"].max(),
    )

    fig_compare.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="완벽한 예측 기준선",
            line=dict(dash="dash"),
        )
    )

    fig_compare.update_layout(height=480)

    st.plotly_chart(
        fig_compare,
        use_container_width=True,
    )

    st.caption(
        "점이 점선에 가까울수록 실제 득점과 예측 득점이 비슷하다."
    )

    st.divider()

    st.subheader("평가 데이터의 예측 결과표")

    result_table = pd.DataFrame({
        "실제 전반전 득점": X_test[X_COL].values,
        "실제 최종 득점": y_test.values,
        "예측 최종 득점": test_pred.round(2),
        "예측 오차": (y_test.values - test_pred).round(2),
    })

    st.dataframe(
        result_table.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 탭 6. 결과 해석
# =========================================================
with tabs[5]:
    st.header("결과 해석")

    st.subheader("1. 분석 결과 요약")

    st.write(
        f"""
        이번 분석에서는 총 {len(analysis_df)}개의 경기 중
        {len(X_train)}개를 학습 데이터로 사용하고,
        {len(X_test)}개를 평가 데이터로 사용했다.

        선형회귀식은 다음과 같다.
        """
    )

    st.latex(
        rf"\hat{{y}} = {intercept:.3f} + ({slope:.3f})x"
    )

    st.write(
        f"""
        기울기는 {slope:.3f}이고, R² 결정계수는 {r2:.3f}이다.
        MAE는 {mae:.3f}골, RMSE는 {rmse:.3f}골로 나타났다.
        """
    )

    if slope > 0:
        st.info(
            "학습 데이터에서는 전반전 득점이 많을수록 "
            "최종 득점도 증가하는 양의 선형 관계가 나타났다."
        )
    elif slope < 0:
        st.info(
            "학습 데이터에서는 전반전 득점이 많을수록 "
            "최종 득점이 감소하는 음의 선형 관계가 나타났다."
        )
    else:
        st.info(
            "학습 데이터에서 전반전 득점에 따른 "
            "선형적인 변화가 거의 나타나지 않았다."
        )

    st.divider()

    st.subheader("2. 결과를 해석할 때 주의할 점")

    st.markdown(
        """
        - 경기 수가 70개로 많지 않으므로 결과를 모든 축구 경기에
          일반화하기 어렵다.
        - 전반전 득점 외에도 선수 구성, 상대 팀의 전력,
          경기 장소, 퇴장 등의 요인이 최종 득점에 영향을 줄 수 있다.
        - 전반전 득점은 최종 득점에 포함되는 값이므로
          두 변수 사이의 관계가 나타나는 것은 자연스러운 측면이 있다.
        - 선형회귀는 직선 관계를 가정하므로 실제 축구 경기의
          복잡한 상황을 모두 설명하지 못한다.
        - 이번 분석은 관계를 살펴보는 것이며 인과관계를 증명하지 않는다.
        """
    )

    st.divider()

    st.subheader("3. 나의 탐구 기록")

    conclusion = st.text_area(
        "분석 결과를 바탕으로 알게 된 점을 적어 보세요.",
        placeholder=(
            "예: 전반전 득점과 최종 득점 사이의 관계를 "
            "회귀분석으로 확인할 수 있었다..."
        ),
        height=130,
    )

    limitation = st.text_area(
        "이번 분석의 한계점을 적어 보세요.",
        placeholder=(
            "예: 경기 수가 많지 않고 전반전 득점 이외의 "
            "여러 요인을 반영하지 못했다..."
        ),
        height=130,
    )

    if st.button("탐구 기록 다운로드"):
        report = (
            "# 축구 선형회귀 탐구 기록\n\n"
            "## 분석 결과\n"
            f"{conclusion}\n\n"
            "## 한계점\n"
            f"{limitation}\n\n"
            "## 회귀식\n"
            f"최종 득점 = {intercept:.3f} + "
            f"({slope:.3f} × 전반전 득점)\n\n"
            f"R² = {r2:.3f}\n"
            f"MAE = {mae:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
        )

        st.download_button(
            "기록 파일 저장",
            data=report.encode("utf-8-sig"),
            file_name="축구_선형회귀_탐구기록.md",
            mime="text/markdown",
        )


# =========================================================
# 하단 안내
# =========================================================
st.divider()

st.caption(
    f"데이터 출처: GitHub CSV | "
    f"분석 경기 수: {len(analysis_df)}개"
)

st.caption(
    "이 앱은 교육용 데이터 분석 실습을 위한 것으로, "
    "통계적 관계와 예측 결과를 이해하는 데 목적이 있다."
)
