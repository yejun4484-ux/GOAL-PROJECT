
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.figure_factory as ff

from io import StringIO
from urllib.parse import unquote

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 기본 설정
# ==========================================
st.set_page_config(
    page_title="전반전 경기 상황과 최종 결과 분석",
    page_icon="⚽",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/yejun4484-ux/GOAL-PROJECT/"
    "refs/heads/main/data/"
    "%25EA%25BF%2588%25EB%2591%2590%25EB%25A0%2588_"
    "%25EB%258D%25B0%25EC%259D%25B4%25ED%2584%25B0_"
    "%25EC%25B6%2595%25EA%25B5%25AC.csv"
)

RESULTS = ["승리", "무승부", "패배"]
SITUATIONS = ["앞섬", "동점", "뒤짐"]

FEATURES = ["한국_전반득점", "상대_전반득점"]

# ==========================================
# 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():
    urls = [DATA_URL, unquote(DATA_URL)]

    for url in urls:
        try:
            response = requests.get(url, timeout=15)
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


df = load_data()

if df.empty:
    st.error("데이터를 불러오지 못했습니다. 데이터 주소를 확인해 주세요.")
    st.stop()

required_columns = [
    "날짜", "연도", "상대팀", "대회",
    "한국_전반득점", "상대_전반득점",
    "한국_최종득점", "상대_최종득점",
    "전반전상황", "최종결과"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()

numeric_columns = [
    "연도", "한국_전반득점", "상대_전반득점",
    "한국_최종득점", "상대_최종득점"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["상대팀", "대회", "전반전상황", "최종결과"]:
    df[col] = df[col].astype(str).str.strip()

analysis_df = df[
    df["전반전상황"].isin(SITUATIONS)
    & df["최종결과"].isin(RESULTS)
].copy()

analysis_df = analysis_df.dropna(
    subset=FEATURES + ["최종결과"]
)

analysis_df = analysis_df.reset_index(drop=True)

if analysis_df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()

# ==========================================
# 로지스틱 회귀 모델
# ==========================================
X = analysis_df[FEATURES]
y = analysis_df["최종결과"]

class_counts = y.value_counts()

if len(class_counts) < 2 or class_counts.min() < 5:
    st.error(
        "로지스틱 회귀를 수행하기에는 특정 결과의 경기 수가 부족합니다."
    )
    st.stop()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        max_iter=3000,
        solver="lbfgs"
    ))
])

# 5겹 교차검증
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 각 경기를 학습에 사용하지 않은 모델로 예측
predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict"
)

# 전체 데이터로 최종 모델 학습
model.fit(X, y)

accuracy = accuracy_score(y, predictions)
precision = precision_score(
    y, predictions, average="macro", zero_division=0
)
recall = recall_score(
    y, predictions, average="macro", zero_division=0
)
f1 = f1_score(
    y, predictions, average="macro", zero_division=0
)

cm = confusion_matrix(
    y,
    predictions,
    labels=RESULTS
)

report = classification_report(
    y,
    predictions,
    labels=RESULTS,
    output_dict=True,
    zero_division=0
)

# ==========================================
# 제목 및 메뉴
# ==========================================
st.title("⚽ 전반전 경기 상황과 최종 결과 분석")

st.caption(
    "대한민국 축구 국가대표팀의 전반전 경기 상황에 따라 "
    "최종 경기 결과는 어떻게 달라질까?"
)

menu = st.sidebar.radio(
    "분석 메뉴",
    [
        "탐구 주제",
        "데이터 탐색",
        "전반전 상황별 경기 수",
        "전반전 상황별 최종 결과",
        "로지스틱 회귀 분석",
        "경기 기록 확인",
        "결과 해석",
        "최종 보고서"
    ]
)

# ==========================================
# 1. 탐구 주제
# ==========================================
if menu == "탐구 주제":
    st.header("🔎 탐구 주제")

    st.subheader(
        "대한민국 축구 국가대표팀의 전반전 경기 상황에 따라 "
        "최종 경기 결과는 어떻게 달라질까?"
    )

    st.write("""
    축구 경기에서는 전반전이 끝났을 때 앞서고 있는지,
    동점인지, 뒤지고 있는지에 따라 경기의 흐름이 달라질 수 있다.

    이번 탐구에서는 대한민국 축구 국가대표팀의 경기 데이터를 활용하여
    전반전 상황별 최종 승리, 무승부, 패배 비율을 비교한다.

    또한 로지스틱 회귀 모델을 이용하여 전반전 득점으로
    최종 경기 결과를 분류할 수 있는지 알아본다.
    """)

    st.subheader("분석 방법")

    st.markdown("""
    1. 대한민국 국가대표팀 경기 데이터를 수집한다.
    2. 전반전 상황별 최종 결과의 비율을 비교한다.
    3. 한국과 상대 팀의 전반전 득점을 입력 변수로 사용한다.
    4. 로지스틱 회귀 모델로 최종 결과를 분류한다.
    5. 교차검증을 이용해 모델의 성능을 평가한다.
    """)

    st.info(
        "로지스틱 회귀는 전반전 득점을 바탕으로 "
        "최종 결과를 승리·무승부·패배로 분류하는 모델이다."
    )

# ==========================================
# 2. 데이터 탐색
# ==========================================
elif menu == "데이터 탐색":
    st.header("📊 데이터 탐색")

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 데이터", f"{len(df)}경기")
    col2.metric("분석 데이터", f"{len(analysis_df)}경기")
    col3.metric("데이터 열 개수", f"{df.shape[1]}개")

    st.subheader("데이터 미리 보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("결측치 확인")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["열 이름", "결측치 개수"]

    st.dataframe(missing, use_container_width=True)

    st.subheader("기초 통계")

    st.dataframe(
        df[numeric_columns].describe(),
        use_container_width=True
    )

    st.subheader("분석에 사용하는 변수")

    st.write("""
    - 입력 변수: 한국_전반득점, 상대_전반득점
    - 예측 대상: 최종결과
    - 분류 항목: 승리, 무승부, 패배
    """)

# ==========================================
# 3. 전반전 상황별 경기 수
# ==========================================
elif menu == "전반전 상황별 경기 수":
    st.header("📈 전반전 상황별 경기 수")

    situation_counts = (
        analysis_df["전반전상황"]
        .value_counts()
        .reindex(SITUATIONS, fill_value=0)
        .reset_index()
    )

    situation_counts.columns = ["전반전 상황", "경기 수"]

    fig = px.bar(
        situation_counts,
        x="전반전 상황",
        y="경기 수",
        text="경기 수",
        title="전반전 상황별 경기 수",
        color="전반전 상황"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        situation_counts,
        use_container_width=True
    )

# ==========================================
# 4. 전반전 상황별 최종 결과
# ==========================================
elif menu == "전반전 상황별 최종 결과":
    st.header("⚽ 전반전 상황별 최종 결과")

    counts = pd.crosstab(
        analysis_df["전반전상황"],
        analysis_df["최종결과"]
    ).reindex(
        index=SITUATIONS,
        columns=RESULTS,
        fill_value=0
    )

    percentages = counts.div(
        counts.sum(axis=1).replace(0, 1),
        axis=0
    ) * 100

    st.subheader("최종 결과 비율")

    display_percentages = percentages.round(1).reset_index()

    st.dataframe(
        display_percentages,
        use_container_width=True
    )

    chart_df = percentages.reset_index().melt(
        id_vars="전반전상황",
        var_name="최종 결과",
        value_name="비율"
    )

    fig = px.bar(
        chart_df,
        x="전반전상황",
        y="비율",
        color="최종 결과",
        text=chart_df["비율"].round(1).astype(str) + "%",
        title="전반전 상황에 따른 최종 결과 비율",
        barmode="stack",
        category_orders={
            "전반전상황": SITUATIONS,
            "최종 결과": RESULTS
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    count_df = counts.reset_index().melt(
        id_vars="전반전상황",
        var_name="최종 결과",
        value_name="경기 수"
    )

    fig2 = px.bar(
        count_df,
        x="전반전상황",
        y="경기 수",
        color="최종 결과",
        barmode="group",
        title="전반전 상황별 최종 결과 경기 수"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.download_button(
        "분석 결과 CSV 다운로드",
        data=display_percentages.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="전반전_상황별_최종결과.csv",
        mime="text/csv"
    )

# ==========================================
# 5. 로지스틱 회귀 분석
# ==========================================
elif menu == "로지스틱 회귀 분석":
    st.header("🤖 로지스틱 회귀 분석")

    st.write("""
    로지스틱 회귀는 입력된 데이터를 바탕으로
    어떤 범주에 속할지 분류하는 머신러닝 방법이다.

    이번 분석에서는 한국과 상대 팀의 전반전 득점을 입력하여
    최종 결과가 승리, 무승부, 패배 중 무엇인지 분류한다.
    """)

    st.subheader("1. 모델에 사용하는 데이터")

    col1, col2 = st.columns(2)

    col1.metric("분석 경기 수", f"{len(analysis_df)}경기")
    col2.metric("입력 변수 개수", "2개")

    st.write("**입력 변수:**")
    st.write("- 한국_전반득점")
    st.write("- 상대_전반득점")

    st.write("**예측 대상:** 승리 / 무승부 / 패배")

    st.divider()

    st.subheader("2. 모델 성능")

    st.caption(
        "5겹 교차검증으로 각 경기의 예측 결과를 만들고 평가했다."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("정확도", f"{accuracy:.1%}")
    c2.metric("정밀도", f"{precision:.1%}")
    c3.metric("재현율", f"{recall:.1%}")
    c4.metric("F1 점수", f"{f1:.1%}")

    st.write("""
    - 정확도: 전체 경기 중 정답을 맞힌 비율
    - 정밀도: 특정 결과라고 예측한 경기 중 실제로 해당 결과인 비율
    - 재현율: 실제 특정 결과인 경기 중 모델이 찾아낸 비율
    - F1 점수: 정밀도와 재현율을 함께 고려한 점수
    """)

    st.divider()

    st.subheader("3. 혼동 행렬")

    cm_df = pd.DataFrame(
        cm,
        index=[f"실제 {r}" for r in RESULTS],
        columns=[f"예측 {r}" for r in RESULTS]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="로지스틱 회귀 혼동 행렬"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    st.caption(
        "행은 실제 경기 결과, 열은 모델이 예측한 결과를 의미한다. "
        "대각선에 있는 숫자가 정답으로 분류한 경기 수이다."
    )

    st.divider()

    st.subheader("4. 결과별 성능")

    class_result = pd.DataFrame({
        "결과": RESULTS,
        "정밀도": [
            report[r]["precision"] for r in RESULTS
        ],
        "재현율": [
            report[r]["recall"] for r in RESULTS
        ],
        "F1 점수": [
            report[r]["f1-score"] for r in RESULTS
        ],
        "실제 경기 수": [
            int(report[r]["support"]) for r in RESULTS
        ]
    })

    for col in ["정밀도", "재현율", "F1 점수"]:
        class_result[col] = (
            class_result[col] * 100
        ).round(1)

    st.dataframe(
        class_result,
        use_container_width=True
    )

    st.divider()

    st.subheader("5. 전반전 점수로 최종 결과 예측하기")

    st.write(
        "전반전 점수를 직접 입력하면 학습된 모델이 "
        "최종 결과를 어떻게 분류하는지 확인할 수 있다."
    )

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        korea_score = st.number_input(
            "한국 전반전 득점",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

    with input_col2:
        opponent_score = st.number_input(
            "상대 팀 전반전 득점",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

    if st.button("최종 결과 예측하기", type="primary"):
        new_data = pd.DataFrame(
            [[korea_score, opponent_score]],
            columns=FEATURES
        )

        predicted = model.predict(new_data)[0]
        probabilities = model.predict_proba(new_data)[0]

        st.success(f"모델의 예측 결과: **{predicted}**")

        probability_df = pd.DataFrame({
            "최종 결과": model.classes_,
            "예측 확률": probabilities * 100
        })

        probability_df["예측 확률"] = (
            probability_df["예측 확률"].round(1)
        )

        fig_prob = px.bar(
            probability_df,
            x="최종 결과",
            y="예측 확률",
            text=probability_df["예측 확률"].astype(str) + "%",
            title="최종 결과별 예측 확률",
            color="최종 결과"
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )

        st.caption(
            "예측 확률은 모델이 학습한 데이터의 패턴을 바탕으로 "
            "계산한 값이며 실제 경기 결과를 보장하지 않는다."
        )

    st.divider()

    st.subheader("6. 실제 결과와 예측 결과 비교")

    comparison = analysis_df[
        ["날짜", "상대팀", "한국_전반득점", "상대_전반득점"]
    ].copy()

    comparison["실제 결과"] = y.values
    comparison["예측 결과"] = predictions
    comparison["예측 성공"] = (
        comparison["실제 결과"] == comparison["예측 결과"]
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.download_button(
        "예측 결과 CSV 다운로드",
        data=comparison.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="로지스틱_회귀_예측결과.csv",
        mime="text/csv"
    )

# ==========================================
# 6. 경기 기록 확인
# ==========================================
elif menu == "경기 기록 확인":
    st.header("🗂️ 경기 기록 확인")

    selected_situation = st.selectbox(
        "전반전 상황 선택",
        ["전체", "앞섬", "동점", "뒤짐"]
    )

    filtered = analysis_df.copy()

    if selected_situation != "전체":
        filtered = filtered[
            filtered["전반전상황"] == selected_situation
        ]

    st.write(f"검색된 경기: {len(filtered)}경기")

    st.dataframe(
        filtered[required_columns],
        use_container_width=True
    )

# ==========================================
# 7. 결과 해석
# ==========================================
elif menu == "결과 해석":
    st.header("💡 결과 해석")

    counts = pd.crosstab(
        analysis_df["전반전상황"],
        analysis_df["최종결과"]
    ).reindex(
        index=SITUATIONS,
        columns=RESULTS,
        fill_value=0
    )

    percentages = counts.div(
        counts.sum(axis=1).replace(0, 1),
        axis=0
    ) * 100

    for situation in SITUATIONS:
        game_count = int(counts.loc[situation].sum())

        if game_count == 0:
            continue

        st.subheader(f"전반전 {situation}")

        for result in RESULTS:
            count = int(counts.loc[situation, result])
            percent = percentages.loc[situation, result]

            st.write(
                f"- 최종 {result}: {count}경기 ({percent:.1f}%)"
            )

    st.divider()

    st.subheader("로지스틱 회귀 분석 결과")

    st.write(f"""
    로지스틱 회귀 모델의 교차검증 정확도는 **{accuracy:.1%}**,
    매크로 F1 점수는 **{f1:.1%}**로 나타났다.

    이 점수는 전반전 득점만으로 최종 결과를 분류했을 때의
    성능을 나타낸다.
    """)

    st.write("""
    다만 경기 수가 적고 승리, 무승부, 패배의 경기 수에도 차이가 있으므로
    모델의 성능을 일반화하기에는 한계가 있다.
    """)

# ==========================================
# 8. 최종 보고서
# ==========================================
elif menu == "최종 보고서":
    st.header("📝 최종 보고서")

    counts = pd.crosstab(
        analysis_df["전반전상황"],
        analysis_df["최종결과"]
    ).reindex(
        index=SITUATIONS,
        columns=RESULTS,
        fill_value=0
    )

    percentages = counts.div(
        counts.sum(axis=1).replace(0, 1),
        axis=0
    ) * 100

    st.subheader("1. 탐구 주제")

    st.write("""
    대한민국 축구 국가대표팀의 전반전 경기 상황에 따라
    최종 경기 결과는 어떻게 달라질까?
    """)

    st.subheader("2. 분석 방법")

    st.write("""
    2021년부터 2025년까지의 대한민국 축구 국가대표팀 경기 데이터를
    활용하였다.

    먼저 전반전 상황을 앞섬, 동점, 뒤짐으로 나누어
    각 상황에서 최종 승리, 무승부, 패배가 나타난 비율을 비교하였다.

    다음으로 한국과 상대 팀의 전반전 득점을 입력 변수로,
    최종 경기 결과를 예측 대상으로 설정하여 다중 로지스틱 회귀를
    수행하였다.

    모델 평가는 5겹 교차검증을 이용하였다.
    """)

    st.subheader("3. 전반전 상황별 분석 결과")

    report_table = percentages.round(1).copy()
    report_table.insert(
        0,
        "경기 수",
        counts.sum(axis=1)
    )

    st.dataframe(
        report_table,
        use_container_width=True
    )

    for situation in SITUATIONS:
        game_count = int(counts.loc[situation].sum())

        if game_count == 0:
            continue

        win_count = int(counts.loc[situation, "승리"])
        draw_count = int(counts.loc[situation, "무승부"])
        loss_count = int(counts.loc[situation, "패배"])

        st.markdown(f"**전반전 {situation}**")

        st.write(
            f"{game_count}경기 중 최종 승리는 {win_count}경기, "
            f"무승부는 {draw_count}경기, 패배는 {loss_count}경기였다."
        )

    st.subheader("4. 로지스틱 회귀 분석 결과")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric("정확도", f"{accuracy:.1%}")
    metric_col2.metric("정밀도", f"{precision:.1%}")
    metric_col3.metric("F1 점수", f"{f1:.1%}")

    st.write(f"""
    로지스틱 회귀 모델의 5겹 교차검증 결과 정확도는
    **{accuracy:.1%}**로 나타났다.

    매크로 정밀도는 **{precision:.1%}**,
    매크로 재현율은 **{recall:.1%}**,
    매크로 F1 점수는 **{f1:.1%}**였다.

    이를 통해 전반전 득점 정보를 이용하여 최종 경기 결과를
    어느 정도 분류할 수 있는지 확인하였다.
    """)

    st.subheader("5. 결론")

    st.write("""
    전반전 상황별 비율을 비교한 결과, 전반전에 앞서고 있었던 경기에서는
    최종 승리 비율이 높게 나타났으며, 뒤지고 있었던 경기에서는
    최종 패배 비율이 높게 나타났다.

    로지스틱 회귀를 활용하여 전반전 득점으로 최종 결과를 분류하는
    모델도 만들 수 있었다.

    따라서 전반전 득점 상황은 최종 경기 결과와 관련된 정보를
    포함하고 있음을 확인하였다.
    """)

    st.subheader("6. 한계")

    st.write("""
    첫째, 분석에 사용한 경기가 70경기로 많지 않기 때문에
    다른 경기에도 같은 결과가 나타난다고 단정하기 어렵다.

    둘째, 승리, 무승부, 패배의 경기 수가 서로 다르기 때문에
    모델이 특정 결과를 더 잘 분류할 가능성이 있다.

    셋째, 상대 팀의 실력, 경기 장소, 선수 구성, 퇴장 여부 등
    최종 결과에 영향을 줄 수 있는 다른 요소는 고려하지 않았다.

    넷째, 교차검증 점수는 현재 수집한 데이터 안에서 평가한 결과이므로
    앞으로 진행될 경기에서도 같은 성능이 나온다는 보장은 없다.

    따라서 분석 결과를 모든 축구 경기에 일반화해서는 안 된다.
    """)

    st.subheader("7. 느낀 점")

    st.write("""
    이번 탐구를 통해 축구 경기의 전반전 상황과 최종 결과 사이의
    관계를 실제 데이터로 분석할 수 있었다.

    처음에는 전반전에서 앞서거나 뒤지는 상황이 최종 결과와
    어떤 관계가 있을지 궁금했다. 데이터를 직접 정리하고
    비율을 비교하면서 예상했던 경향을 수치로 확인할 수 있었다.

    또한 로지스틱 회귀를 활용해 전반전 득점으로 최종 결과를
    분류하는 과정도 경험했다.

    단순히 정확도만 보는 것이 아니라 정밀도, 재현율, F1 점수와
    혼동 행렬을 함께 살펴봐야 모델의 성능을 더 정확하게
    이해할 수 있다는 점을 배웠다.
    """)

    st.divider()

    st.caption(
        "자료 출처: 대한민국 축구 국가대표팀 경기 데이터 "
        "(제공된 CSV 자료)"
    )

    st.caption(
        "분석 방법: 전반전 상황별 비율 분석, 다중 로지스틱 회귀, "
        "5겹 교차검증"
    )
