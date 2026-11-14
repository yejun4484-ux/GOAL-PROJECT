import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

from io import StringIO
from urllib.parse import unquote

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =========================================================
# 1. 기본 설정
# =========================================================
st.set_page_config(
    page_title="대한민국 축구 경기 데이터 분석",
    page_icon="⚽",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/yejun4484-ux/GOAL-PROJECT/"
    "refs/heads/main/data/"
    "%25EA%25BF%2588%25EB%2591%2590%25EB%25A0%2588_"
    "%25EB%258D%25B0%25EC%259D%25B4%25ED%2584%25B0_"
    "%25EC%25B6%2595%25EA%25B5%25AC.csv"
)

FEATURES = ["한국_전반득점", "상대_전반득점"]
RESULTS = ["승리", "무승부", "패배"]
SITUATIONS = ["앞섬", "동점", "뒤짐"]

SCORE_COLUMNS = [
    "한국_전반득점",
    "상대_전반득점",
    "한국_최종득점",
    "상대_최종득점",
]

REQUIRED_COLUMNS = [
    "날짜", "연도", "상대팀", "대회",
    "한국_전반득점", "상대_전반득점",
    "한국_최종득점", "상대_최종득점",
    "전반전상황", "최종결과",
]

FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans CJK KR, sans-serif"


# =========================================================
# 2. 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():
    for url in [DATA_URL, unquote(DATA_URL)]:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()

            for encoding in ["utf-8-sig", "utf-8", "cp949"]:
                try:
                    return pd.read_csv(
                        StringIO(response.content.decode(encoding))
                    )
                except UnicodeDecodeError:
                    continue

        except Exception:
            continue

    return pd.DataFrame()


original_df = load_data()

if original_df.empty:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "GitHub CSV 주소와 저장소 공개 여부를 확인해 주세요."
    )
    st.stop()

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in original_df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()


# =========================================================
# 3. 데이터 전처리
# =========================================================
def preprocess_data(data):
    df = data.copy()

    # 득점과 연도를 숫자형으로 변환
    for col in SCORE_COLUMNS + ["연도"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 문자열 앞뒤 공백 제거
    for col in ["상대팀", "대회", "전반전상황", "최종결과"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # 분석에 필요한 값이 없는 행 제거
    df = df.dropna(
        subset=FEATURES + ["최종결과", "전반전상황"]
    )

    # 중복 데이터 제거
    df = df.drop_duplicates()

    # 정해진 범주에 해당하는 데이터만 사용
    df = df[
        df["최종결과"].isin(RESULTS)
        & df["전반전상황"].isin(SITUATIONS)
    ].copy()

    return df.reset_index(drop=True)


clean_df = preprocess_data(original_df)

if clean_df.empty:
    st.error("전처리 후 분석할 데이터가 없습니다.")
    st.stop()

if clean_df["최종결과"].value_counts().reindex(
    RESULTS,
    fill_value=0
).min() < 5:

    st.error(
        "승리·무승부·패배 중 한 범주의 경기 수가 5개보다 적어 "
        "현재 설정한 5겹 교차검증을 진행하기 어렵습니다. "
        "데이터를 확인해 주세요."
    )
    st.stop()


# =========================================================
# 4. 최종 모델: 로지스틱 회귀
# =========================================================
def make_logistic_model():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000)),
    ])


@st.cache_data
def evaluate_logistic(data):

    X = data[FEATURES]
    y = data["최종결과"]

    # 5겹 교차검증
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # 기준선 모델
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline_pred = cross_val_predict(
        baseline,
        X,
        y,
        cv=cv
    )

    # 로지스틱 회귀 모델
    logistic = make_logistic_model()

    logistic_pred = cross_val_predict(
        logistic,
        X,
        y,
        cv=cv
    )

    # 평가 지표 계산
    rows = []

    for name, pred in [
        ("베이스라인", baseline_pred),
        ("로지스틱 회귀", logistic_pred),
    ]:

        rows.append({
            "모델": name,

            "정확도": accuracy_score(
                y, pred
            ),

            "정밀도": precision_score(
                y,
                pred,
                average="macro",
                zero_division=0
            ),

            "재현율": recall_score(
                y,
                pred,
                average="macro",
                zero_division=0
            ),

            "F1 점수": f1_score(
                y,
                pred,
                average="macro",
                zero_division=0
            ),
        })

    score_df = pd.DataFrame(rows)

    # 전체 데이터를 이용해 최종 모델 학습
    final_model = make_logistic_model()

    final_model.fit(X, y)

    # 실제 결과와 교차검증 예측 결과 정리
    prediction_df = data[
        ["날짜", "상대팀"] + FEATURES
    ].copy()

    prediction_df["실제 결과"] = y.values

    prediction_df["로지스틱 회귀 예측"] = logistic_pred

    return score_df, prediction_df, final_model


score_df, prediction_df, final_model = evaluate_logistic(clean_df)


# =========================================================
# 5. 공통 통계
# =========================================================

# 전반전 상황별 최종 결과 경기 수
counts = pd.crosstab(
    clean_df["전반전상황"],
    clean_df["최종결과"],
).reindex(
    index=SITUATIONS,
    columns=RESULTS,
    fill_value=0
)

# 전반전 상황별 최종 결과 비율
percentages = counts.div(
    counts.sum(axis=1).replace(0, 1),
    axis=0,
) * 100

# 최종 결과별 경기 수
result_counts = (
    clean_df["최종결과"]
    .value_counts()
    .reindex(RESULTS, fill_value=0)
    .rename_axis("최종 결과")
    .reset_index(name="경기 수")
)

result_counts["비율(%)"] = (
    result_counts["경기 수"] / len(clean_df) * 100
).round(1)

# 평가 점수를 퍼센트로 변환
display_scores = score_df.copy()

for col in ["정확도", "정밀도", "재현율", "F1 점수"]:
    display_scores[col] = (
        display_scores[col] * 100
    ).round(1)


# =========================================================
# 6. 제목과 탭
# =========================================================
st.title("⚽ 대한민국 축구 국가대표팀 경기 분석")

st.caption(
    "전반전 득점을 바탕으로 최종 경기 결과 "
    "(승리·무승부·패배)를 로지스틱 회귀로 분류한다."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "① 탐구 질문",
    "② 데이터와 전처리",
    "③ 경기 상황 분석",
    "④ 로지스틱 회귀 분석",
])


# =========================================================
# TAB 1. 탐구 질문
# =========================================================
with tab1:

    st.header("1. 탐구 질문")

    st.subheader(
        "대한민국 축구 국가대표팀의 전반전 경기 상황에 따라 "
        "최종 경기 결과는 어떻게 달라질까?"
    )

    st.write("""
    축구 경기를 보면서 전반전에 앞서거나 뒤지는 상황이
    최종 경기 결과와 어떤 관계가 있는지 궁금했다.

    전반전에 앞서면 실제로 승리할 가능성이 높은지,
    뒤지고 있어도 최종 결과가 달라질 수 있는지
    경기 데이터를 통해 확인하고자 했다.
    """)

    st.divider()

    st.subheader("분석 목표")

    st.markdown("""
    1. 전반전 상황별 최종 결과 비율을 비교한다.
    2. 한국과 상대 팀의 전반전 득점을 입력 변수로 사용한다.
    3. 로지스틱 회귀로 최종 결과를 승리·무승부·패배로 분류한다.
    4. 기준선과 비교하고 혼동 행렬 및 확률 그래프로 성능을 해석한다.
    """)

    st.divider()

    st.subheader("왜 분류 문제일까?")

    st.write("""
    이번 탐구에서 예측하려는 값은 연속적인 숫자가 아니라
    승리, 무승부, 패배라는 세 가지 범주이다.

    따라서 범주를 예측하는 분류가 적합하다.
    """)

    st.markdown("""
    - 회귀: 득점 수처럼 연속적인 숫자를 예측할 때 사용한다.
    - 분류: 승리·무승부·패배처럼 정해진 범주를 예측할 때 사용한다.
    - 군집: 정답 없이 비슷한 데이터끼리 묶을 때 사용한다.
    """)

    st.divider()

    st.subheader("최종 모델을 로지스틱 회귀로 정한 이유")

    st.markdown("""
    - 최종 결과가 세 범주로 나뉘므로 다중 분류에 사용할 수 있다.
    - 각 결과가 나올 확률을 확인할 수 있어 예측 근거를 설명하기 쉽다.
    - 입력 변수가 전반전 득점 2개로 단순하여 모델의 결과를 해석하기에 적절하다.
    - 표준화와 함께 사용하면 두 득점 변수의 단위 차이를 조정할 수 있다.
    - 기준선과 같은 교차검증 방식으로 성능을 비교할 수 있다.
    """)

    st.info(
        "로지스틱 회귀가 무조건 가장 정확하다는 뜻은 아니다. "
        "이번 탐구에서는 범주형 결과와 확률 해석에 적합하다는 점을 "
        "근거로 최종 분석 모델로 선택했다."
    )


# =========================================================
# TAB 2. 데이터와 전처리
# =========================================================
with tab2:

    st.header("2. 데이터와 전처리")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "원본 데이터 행 수",
        f"{len(original_df):,}개"
    )

    c2.metric(
        "데이터 열 수",
        f"{original_df.shape[1]}개"
    )

    c3.metric(
        "전처리 후 경기 수",
        f"{len(clean_df):,}경기"
    )

    st.subheader("원본 데이터 미리보기")

    st.dataframe(
        original_df.head(10),
        use_container_width=True
    )

    st.subheader("데이터 속성 유형")

    type_table = pd.DataFrame({
        "속성": REQUIRED_COLUMNS,

        "유형": [
            "날짜형", "수치형", "범주형", "범주형",
            "수치형(이산형)", "수치형(이산형)",
            "수치형(이산형)", "수치형(이산형)",
            "범주형", "범주형",
        ],

        "활용": [
            "경기 시점 확인", "연도별 구분", "상대 팀 구분", "대회 구분",
            "모델 입력 변수", "모델 입력 변수",
            "최종 득점 확인", "최종 득점 확인",
            "상황별 통계", "모델 예측 대상",
        ],
    })

    st.dataframe(
        type_table,
        use_container_width=True
    )

    st.subheader("결측치 확인")

    missing_df = pd.DataFrame({
        "열": original_df.columns,

        "결측치 개수": [
            int(original_df[col].isna().sum())
            for col in original_df.columns
        ],
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.write(
        f"원본 데이터의 전체 결측치: "
        f"**{int(original_df.isna().sum().sum()):,}개**"
    )

    st.subheader("전처리 방법")

    st.markdown("""
    1. 득점과 연도 열을 숫자형으로 변환했다.
    2. 문자열 앞뒤의 공백을 제거했다.
    3. 모델 분석에 필요한 전반전 득점 또는 결과가 비어 있는 행을 제거했다.
    4. 중복된 행을 제거했다.
    5. 최종 결과가 승리·무승부·패배이고 전반전 상황이 앞섬·동점·뒤짐인 기록만 사용했다.
    6. 득점이 크다는 이유만으로 실제 경기 기록을 이상치로 삭제하지 않았다.
    """)

    st.subheader("전처리 전후 비교")

    comparison = pd.DataFrame({

        "항목": [
            "행 수",
            "한국 전반전 득점 평균",
            "상대 전반전 득점 평균",
            "한국 최종 득점 평균",
            "상대 최종 득점 평균",
        ],

        "전처리 전": [
            len(original_df),

            pd.to_numeric(
                original_df["한국_전반득점"],
                errors="coerce"
            ).mean(),

            pd.to_numeric(
                original_df["상대_전반득점"],
                errors="coerce"
            ).mean(),

            pd.to_numeric(
                original_df["한국_최종득점"],
                errors="coerce"
            ).mean(),

            pd.to_numeric(
                original_df["상대_최종득점"],
                errors="coerce"
            ).mean(),
        ],

        "전처리 후": [
            len(clean_df),
            clean_df["한국_전반득점"].mean(),
            clean_df["상대_전반득점"].mean(),
            clean_df["한국_최종득점"].mean(),
            clean_df["상대_최종득점"].mean(),
        ],
    })

    st.dataframe(
        comparison.round(2),
        use_container_width=True
    )

    st.subheader("전처리된 데이터")

    st.dataframe(
        clean_df,
        use_container_width=True
    )

    st.download_button(
        "전처리 데이터 다운로드",

        data=clean_df.to_csv(
            index=False
        ).encode("utf-8-sig"),

        file_name="전처리_축구데이터.csv",

        mime="text/csv",
    )


# =========================================================
# TAB 3. 경기 상황 분석
# =========================================================
with tab3:

    st.header("3. 전반전 상황과 최종 결과")

    # 최종 결과별 경기 수
    st.subheader("최종 결과별 경기 수")

    st.dataframe(
        result_counts,
        use_container_width=True
    )

    fig_result = px.bar(
        result_counts,
        x="최종 결과",
        y="경기 수",
        color="최종 결과",
        text="경기 수",
        title="최종 결과별 경기 수",
    )

    fig_result.update_layout(
        font_family=FONT
    )

    st.plotly_chart(
        fig_result,
        use_container_width=True
    )

    st.caption(
        "해석: 막대의 높이를 비교하면 전체 경기에서 "
        "승리·무승부·패배가 각각 몇 번 나타났는지 알 수 있다. "
        "실제 수치는 위 표와 그래프에서 확인한다."
    )

    st.divider()

    # 전반전 상황별 최종 결과 비율
    st.subheader("전반전 상황별 최종 결과 비율")

    st.write(
        "전반전에 앞선 경기, 동점인 경기, 뒤진 경기에서 "
        "최종 결과가 어떻게 달라지는지 비교한다."
    )

    st.write("상황별 경기 수")

    st.dataframe(
        counts,
        use_container_width=True
    )

    st.write("상황별 최종 결과 비율(%)")

    st.dataframe(
        percentages.round(1),
        use_container_width=True
    )

    chart_df = percentages.reset_index().melt(
        id_vars="전반전상황",
        var_name="최종 결과",
        value_name="비율(%)",
    )

    fig_situation = px.bar(
        chart_df,
        x="전반전상황",
        y="비율(%)",
        color="최종 결과",
        text=chart_df["비율(%)"].round(1).astype(str) + "%",
        barmode="stack",

        category_orders={
            "전반전상황": SITUATIONS,
            "최종 결과": RESULTS,
        },

        title="전반전 상황별 최종 결과 비율",
    )

    fig_situation.update_layout(
        yaxis_range=[0, 100],
        font_family=FONT,
    )

    st.plotly_chart(
        fig_situation,
        use_container_width=True
    )

    st.info("""
    해석 방법:

    - 각 막대는 해당 전반전 상황의 전체 경기 수를 100%로 나타낸다.
    - 막대 안에서 승리·무승부·패배가 차지하는 비율을 비교한다.
    - 앞선 상황에서 승리 비율이 높게 나타나는지, 뒤진 상황에서도 승리한 경기가 있는지 확인한다.
    - 이 그래프는 두 변수의 관계를 보여 주며, 전반전 상황이 최종 결과의 직접적인 원인이라고 단정할 수는 없다.
    """)

    st.divider()

    # 전반전 득점과 최종 결과의 관계
    st.subheader("전반전 득점과 최종 결과의 분포")

    fig_scatter = px.scatter(
        clean_df,

        x="한국_전반득점",
        y="상대_전반득점",

        color="최종결과",

        hover_data=[
            "날짜",
            "상대팀",
            "전반전상황"
        ],

        category_orders={
            "최종결과": RESULTS
        },

        title="한국과 상대 팀의 전반전 득점에 따른 최종 결과",

        labels={
            "한국_전반득점": "대한민국 전반전 득점",
            "상대_전반득점": "상대 팀 전반전 득점",
            "최종결과": "최종 결과",
        },
    )

    fig_scatter.update_layout(
        font_family=FONT
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

    st.caption(
        "해석: 가로축은 한국의 전반전 득점, 세로축은 상대 팀의 "
        "전반전 득점이다. 색깔별 점의 분포를 살펴보면 "
        "어떤 전반전 득점 조합에서 각 결과가 나타났는지 확인할 수 있다."
    )


# =========================================================
# TAB 4. 로지스틱 회귀 분석
# =========================================================
with tab4:

    st.header("4. 로지스틱 회귀 분석")

    st.subheader("모델의 입력과 예측 대상")

    st.markdown("""
    **입력 변수(X)**

    - 한국의 전반전 득점
    - 상대 팀의 전반전 득점

    **예측 대상(y)**

    - 승리
    - 무승부
    - 패배
    """)

    st.write("""
    로지스틱 회귀는 입력값을 바탕으로 각 결과에 해당할 확률을 계산한다.
    세 결과의 확률을 비교하여 가장 높은 확률을 가진 결과를 최종 예측으로 선택한다.
    """)

    st.divider()

    st.subheader("모델을 선택한 이유")

    st.markdown("""
    1. 예측 대상이 숫자가 아닌 세 가지 범주이므로 분류 문제에 적합하다.
    2. 승리·무승부·패배 각각의 예측 확률을 확인할 수 있다.
    3. 표준화된 입력 변수의 계수를 살펴보며 예측에 어떤 방향으로 관련되는지 해석할 수 있다.
    4. 모델 구조가 비교적 단순해 포스터 발표에서 원리를 설명하기 쉽다.
    """)

    st.divider()

    # 평가 방법
    st.subheader("평가 방법과 지표")

    st.write("""
    데이터를 5개 부분으로 나누고, 4개 부분으로 학습한 뒤
    나머지 1개 부분으로 검증하는 과정을 반복하는 5겹 교차검증을 사용했다.
    기준선과 로지스틱 회귀에 같은 검증 방식을 적용했다.
    """)

    metric_help = pd.DataFrame({

        "평가 지표": [
            "정확도",
            "정밀도",
            "재현율",
            "매크로 F1 점수"
        ],

        "의미": [
            "전체 경기 중 결과를 맞힌 비율",
            "특정 결과라고 예측한 경기 중 실제로 그 결과인 비율",
            "실제 특정 결과 경기 중 모델이 찾아낸 비율",
            "정밀도와 재현율을 함께 고려한 값으로, 세 결과의 성능을 동일한 비중으로 평균",
        ],
    })

    st.dataframe(
        metric_help,
        use_container_width=True
    )

    st.divider()

    # 기준선과 로지스틱 회귀 성능 비교
    st.subheader("기준선과 로지스틱 회귀 성능 비교")

    st.dataframe(
        display_scores,
        use_container_width=True
    )

    metric_choice = st.selectbox(
        "비교할 평가 지표",
        ["정확도", "정밀도", "재현율", "F1 점수"],
    )

    fig_scores = px.bar(
        display_scores,

        x="모델",
        y=metric_choice,

        color="모델",
        text=metric_choice,

        title=f"베이스라인과 로지스틱 회귀의 {metric_choice} 비교",

        range_y=[0, 100],
    )

    fig_scores.update_layout(
        yaxis_title="점수(%)",
        font_family=FONT,
    )

    st.plotly_chart(
        fig_scores,
        use_container_width=True
    )

    # 기준선과 성능 차이 계산
    baseline_row = score_df[
        score_df["모델"] == "베이스라인"
    ].iloc[0]

    logistic_row = score_df[
        score_df["모델"] == "로지스틱 회귀"
    ].iloc[0]

    acc_diff = (
        logistic_row["정확도"]
        - baseline_row["정확도"]
    ) * 100

    f1_diff = (
        logistic_row["F1 점수"]
        - baseline_row["F1 점수"]
    ) * 100

    c1, c2 = st.columns(2)

    c1.metric(
        "정확도 차이(로지스틱 - 기준선)",
        f"{acc_diff:+.1f}%p"
    )

    c2.metric(
        "매크로 F1 차이(로지스틱 - 기준선)",
        f"{f1_diff:+.1f}%p"
    )

    if logistic_row["F1 점수"] > baseline_row["F1 점수"]:

        st.success(
            "로지스틱 회귀의 매크로 F1 점수가 기준선보다 높다. "
            "단순히 가장 많은 결과만 예측하는 방식보다 "
            "세 결과를 구분하는 데 나은 성능을 보였다."
        )

    elif logistic_row["F1 점수"] < baseline_row["F1 점수"]:

        st.warning(
            "로지스틱 회귀의 매크로 F1 점수가 기준선보다 낮다. "
            "현재 입력 변수와 데이터만으로는 "
            "기준선보다 나은 분류 성능을 얻지 못했다."
        )

    else:

        st.info(
            "로지스틱 회귀와 기준선의 매크로 F1 점수가 같다."
        )

    st.caption(
        "평가 점수는 실제 데이터로 계산된다. "
        "점수가 높더라도 모든 경기를 정확하게 예측한다는 뜻은 아니다."
    )

    st.divider()

    # 혼동 행렬
    st.subheader("혼동 행렬")

    cm = confusion_matrix(
        prediction_df["실제 결과"],
        prediction_df["로지스틱 회귀 예측"],
        labels=RESULTS,
    )

    cm_df = pd.DataFrame(
        cm,

        index=[
            f"실제 {x}"
            for x in RESULTS
        ],

        columns=[
            f"예측 {x}"
            for x in RESULTS
        ],
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="로지스틱 회귀 혼동 행렬",
    )

    fig_cm.update_layout(
        font_family=FONT
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    st.write("""
    - 세로축(행)은 실제 경기 결과이다.
    - 가로축(열)은 모델이 예측한 결과이다.
    - 대각선에 있는 숫자는 실제 결과를 맞힌 경기 수이다.
    - 대각선 밖의 숫자는 다른 결과로 잘못 분류한 경기 수이다.
    """)

    st.divider()

    # 로지스틱 회귀 계수
    st.subheader("로지스틱 회귀 계수 해석")

    fitted_logistic = final_model.named_steps["model"]

    coefficient_df = pd.DataFrame(
        fitted_logistic.coef_,

        columns=[
            "한국 전반전 득점",
            "상대 전반전 득점"
        ],

        index=fitted_logistic.classes_,
    )

    coefficient_long = (
        coefficient_df
        .rename_axis("결과")
        .reset_index()
        .melt(
            id_vars="결과",
            var_name="입력 변수",
            value_name="계수",
        )
    )

    fig_coef = px.bar(
        coefficient_long,

        x="입력 변수",
        y="계수",

        color="결과",
        barmode="group",

        title="결과 범주별 로지스틱 회귀 계수",
    )

    fig_coef.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )

    fig_coef.update_layout(
        font_family=FONT
    )

    st.plotly_chart(
        fig_coef,
        use_container_width=True
    )

    st.write("""
    계수는 표준화된 전반전 득점이 각 결과의 모델 점수에
    어떤 방향으로 관련되는지 보여 준다.

    양수 계수는 해당 결과의 상대적인 점수를 높이는 방향,
    음수 계수는 낮추는 방향으로 작용한다.

    계수의 크기만으로 인과관계나 실제 승리 확률의 변화량을
    단정해서는 안 된다.

    로지스틱 회귀는 여러 결과의 점수를 함께 계산하므로
    계수는 다른 결과와 비교해서 해석해야 한다.
    """)

    st.divider()

    # 새로운 경기 예측
    st.subheader("새로운 경기의 결과 예측")

    st.write(
        "전반전 종료 시점의 득점을 입력하면 "
        "최종 결과별 예측 확률을 확인할 수 있다."
    )

    input_col1, input_col2 = st.columns(2)

    with input_col1:

        korea_first = st.number_input(
            "대한민국 전반전 득점",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

    with input_col2:

        opponent_first = st.number_input(
            "상대 팀 전반전 득점",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
        )

    if st.button(
        "로지스틱 회귀로 예측하기",
        type="primary"
    ):

        new_match = pd.DataFrame(
            [[korea_first, opponent_first]],
            columns=FEATURES,
        )

        probabilities = final_model.predict_proba(
            new_match
        )[0]

        classes = final_model.named_steps["model"].classes_

        probability_df = pd.DataFrame({

            "최종 결과": classes,

            "예측 확률(%)": probabilities * 100,

        }).sort_values(
            "예측 확률(%)",
            ascending=False
        )

        predicted_result = final_model.predict(
            new_match
        )[0]

        st.success(
            f"모델의 예측 결과: **{predicted_result}**"
        )

        fig_probability = px.bar(
            probability_df,

            x="최종 결과",
            y="예측 확률(%)",

            color="최종 결과",

            text=probability_df[
                "예측 확률(%)"
            ].round(1).astype(str) + "%",

            title="입력한 전반전 득점에 대한 최종 결과별 예측 확률",

            range_y=[0, 100],
        )

        fig_probability.update_layout(
            font_family=FONT
        )

        st.plotly_chart(
            fig_probability,
            use_container_width=True
        )

        st.dataframe(
            probability_df.round(1),
            use_container_width=True
        )

        st.caption(
            "이 확률은 학습 데이터에서 학습한 모델의 추정치이며 "
            "실제 경기 결과를 보장하지 않는다."
        )

    st.divider()

    # 실제 결과와 예측 결과
    st.subheader("실제 결과와 교차검증 예측 비교")

    st.dataframe(
        prediction_df,
        use_container_width=True
    )

    st.download_button(
        "예측 결과 다운로드",

        data=prediction_df.to_csv(
            index=False
        ).encode("utf-8-sig"),

        file_name="로지스틱_회귀_예측결과.csv",

        mime="text/csv",
    )

    st.divider()

    # 최종 분석 및 한계
    st.subheader("최종 분석 및 한계")

    st.markdown("""
    **분석에서 확인할 점**

    - 전반전 상황별 최종 결과 비율을 비교하여 경기 흐름과 최종 결과의 관계를 살펴본다.
    - 로지스틱 회귀의 예측 확률을 통해 전반전 득점 조합에 따른 모델의 판단을 확인한다.
    - 기준선과 정확도·정밀도·재현율·F1 점수를 비교하여 모델의 성능을 판단한다.

    **한계**

    - 경기 수가 충분하지 않으면 결과를 다른 경기에도 일반화하기 어렵다.
    - 전반전 득점 두 변수만 사용했기 때문에 상대 팀의 실력, 경기 장소, 선수 구성, 퇴장, 부상 등은 반영하지 못했다.
    - 전반전 득점과 최종 결과의 관계가 있다고 해서 전반전 득점만이 최종 결과의 원인이라고 할 수 없다.
    - 모델의 예측 확률은 실제 경기의 확정된 결과가 아니라 학습 데이터에 근거한 추정치이다.
    """)

    st.subheader("탐구 결론 작성 안내")

    st.write(
        "위 그래프와 평가 결과에서 실제 수치를 확인한 뒤, "
        "‘전반전 상황 중 어떤 경우에 승리 비율이 높았는가’, "
        "‘로지스틱 회귀는 기준선보다 어떤 지표에서 높거나 낮았는가’, "
        "‘어떤 경기 요인이 추가되면 분석이 더 정확해질 수 있는가’를 "
        "중심으로 결론을 작성한다."
    )
