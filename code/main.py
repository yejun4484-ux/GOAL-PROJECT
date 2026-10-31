
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

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
).reset_index(drop=True)

if analysis_df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()

# ==========================================
# 모델 학습 및 평가
# ==========================================
X = analysis_df[FEATURES]
y = analysis_df["최종결과"]

class_counts = y.value_counts()

if len(class_counts) < 2 or class_counts.min() < 5:
    st.error(
        "특정 결과의 경기 수가 부족하여 5겹 교차검증을 "
        "안정적으로 수행할 수 없습니다."
    )
    st.stop()

# 로지스틱 회귀
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        max_iter=3000,
        solver="lbfgs"
    ))
])

# 베이스라인: 학습 데이터에서 가장 많은 결과만 예측
baseline_model = DummyClassifier(
    strategy="most_frequent"
)

# 두 모델에 동일한 교차검증 적용
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_predictions = cross_val_predict(
    logistic_model,
    X,
    y,
    cv=cv,
    method="predict"
)

baseline_predictions = cross_val_predict(
    baseline_model,
    X,
    y,
    cv=cv,
    method="predict"
)

# 전체 데이터로 최종 모델 학습
logistic_model.fit(X, y)

# ==========================================
# 성능 계산
# ==========================================
logistic_accuracy = accuracy_score(
    y, logistic_predictions
)

baseline_accuracy = accuracy_score(
    y, baseline_predictions
)

logistic_precision = precision_score(
    y,
    logistic_predictions,
    average="macro",
    zero_division=0
)

logistic_recall = recall_score(
    y,
    logistic_predictions,
    average="macro",
    zero_division=0
)

logistic_f1 = f1_score(
    y,
    logistic_predictions,
    average="macro",
    zero_division=0
)

baseline_f1 = f1_score(
    y,
    baseline_predictions,
    average="macro",
    zero_division=0
)

accuracy_difference = (
    logistic_accuracy - baseline_accuracy
) * 100

f1_difference = (
    logistic_f1 - baseline_f1
) * 100

cm = confusion_matrix(
    y,
    logistic_predictions,
    labels=RESULTS
)

report = classification_report(
    y,
    logistic_predictions,
    labels=RESULTS,
    output_dict=True,
    zero_division=0
)

# 베이스라인에서 가장 많이 예측한 결과
majority_result = y.value_counts().idxmax()

# ==========================================
# 해석에 사용할 통계
# ==========================================
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

ahead_win_rate = percentages.loc["앞섬", "승리"]
behind_loss_rate = percentages.loc["뒤짐", "패배"]

# ==========================================
# 메뉴
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

    st.subheader("주제를 선택한 이유")

    st.write("""
    평소 축구 경기를 보면서 전반전에 앞서거나 뒤지는 상황이
    최종 경기 결과와 어떤 관계가 있는지 궁금했다.

    전반전에 앞서고 있으면 실제로 승리할 가능성이 높은지,
    반대로 뒤지고 있어도 역전할 수 있는지 데이터를 통해
    직접 확인하고 싶어 이 주제를 선택했다.
    """)

    st.subheader("분석 방법")

    st.markdown("""
    1. 대한민국 국가대표팀 경기 데이터를 수집한다.
    2. 전반전 상황별 최종 결과 비율을 비교한다.
    3. 한국과 상대 팀의 전반전 득점을 입력 변수로 사용한다.
    4. 로지스틱 회귀 모델로 최종 결과를 분류한다.
    5. 베이스라인과 모델 성능을 비교한다.
    """)

# ==========================================
# 2. 데이터 탐색
# ==========================================
elif menu == "데이터 탐색":
    st.header("📊 데이터 탐색")

    c1, c2, c3 = st.columns(3)

    c1.metric("원본 경기 수", f"{len(df)}경기")
    c2.metric("분석 경기 수", f"{len(analysis_df)}경기")
    c3.metric("데이터 열 개수", f"{df.shape[1]}개")

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

    st.subheader("모델에 사용하는 변수")

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

    st.write("""
    이 그래프는 전반전에 앞선 경기, 동점인 경기,
    뒤진 경기의 수를 보여 준다.
    """)

# ==========================================
# 4. 전반전 상황별 최종 결과
# ==========================================
elif menu == "전반전 상황별 최종 결과":
    st.header("⚽ 전반전 상황별 최종 결과")

    display_percentages = percentages.round(1).reset_index()

    st.subheader("최종 결과 비율")

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
    한국과 상대 팀의 전반전 득점을 입력하여
    최종 결과가 승리, 무승부, 패배 중 무엇인지 분류한다.
    """)

    st.subheader("1. 모델 성능")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("정확도", f"{logistic_accuracy:.1%}")
    c2.metric("정밀도", f"{logistic_precision:.1%}")
    c3.metric("재현율", f"{logistic_recall:.1%}")
    c4.metric("F1 점수", f"{logistic_f1:.1%}")

    st.write("""
    - 정확도: 전체 경기 중 정답을 맞힌 비율
    - 정밀도: 특정 결과라고 예측한 경기 중 실제로 해당 결과인 비율
    - 재현율: 실제 특정 결과인 경기 중 모델이 찾아낸 비율
    - F1 점수: 정밀도와 재현율을 함께 고려한 점수
    """)

    st.divider()

    st.subheader("2. 베이스라인과 비교")

    st.write("""
    베이스라인은 복잡한 분석을 하지 않고
    학습 데이터에서 가장 많이 등장한 결과만 예측하는 기준 모델이다.

    두 모델은 동일한 5겹 교차검증으로 평가하였다.
    """)

    baseline_table = pd.DataFrame({
        "모델": ["베이스라인", "로지스틱 회귀"],
        "정확도": [
            baseline_accuracy * 100,
            logistic_accuracy * 100
        ],
        "매크로 F1": [
            baseline_f1 * 100,
            logistic_f1 * 100
        ]
    }).round(1)

    st.dataframe(
        baseline_table,
        use_container_width=True
    )

    comparison_chart = baseline_table.melt(
        id_vars="모델",
        var_name="평가 지표",
        value_name="점수"
    )

    fig_compare = px.bar(
        comparison_chart,
        x="평가 지표",
        y="점수",
        color="모델",
        barmode="group",
        text=comparison_chart["점수"].astype(str) + "%",
        title="베이스라인과 로지스틱 회귀 성능 비교"
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    if accuracy_difference > 0:
        st.success(
            f"로지스틱 회귀의 정확도가 베이스라인보다 "
            f"{accuracy_difference:.1f}%p 높다."
        )
    elif accuracy_difference < 0:
        st.warning(
            f"로지스틱 회귀의 정확도가 베이스라인보다 "
            f"{abs(accuracy_difference):.1f}%p 낮다."
        )
    else:
        st.info(
            "로지스틱 회귀와 베이스라인의 정확도가 같다."
        )

    st.write(
        f"매크로 F1 점수 차이: {f1_difference:+.1f}%p"
    )

    st.caption(
        "정확도 차이는 퍼센트포인트(%p)로 표시한다. "
        "두 모델의 성능은 동일한 교차검증 방식으로 비교했다."
    )

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

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    st.write("""
    행은 실제 경기 결과, 열은 모델이 예측한 결과를 뜻한다.
    대각선에 있는 숫자는 모델이 정답을 맞힌 경기 수이고,
    대각선 밖의 숫자는 잘못 분류한 경기 수이다.
    """)

    st.divider()

    st.subheader("4. 결과별 성능")

    class_result = pd.DataFrame({
        "결과": RESULTS,
        "정밀도": [
            report[r]["precision"] * 100 for r in RESULTS
        ],
        "재현율": [
            report[r]["recall"] * 100 for r in RESULTS
        ],
        "F1 점수": [
            report[r]["f1-score"] * 100 for r in RESULTS
        ],
        "실제 경기 수": [
            int(report[r]["support"]) for r in RESULTS
        ]
    }).round(1)

    st.dataframe(
        class_result,
        use_container_width=True
    )

    st.divider()

    st.subheader("5. 전반전 점수로 최종 결과 예측하기")

    col1, col2 = st.columns(2)

    with col1:
        korea_score = st.number_input(
            "한국 전반전 득점",
            min_value=0,
            max_value=20,
            value=1
        )

    with col2:
        opponent_score = st.number_input(
            "상대 팀 전반전 득점",
            min_value=0,
            max_value=20,
            value=0
        )

    if st.button("최종 결과 예측하기", type="primary"):
        new_data = pd.DataFrame(
            [[korea_score, opponent_score]],
            columns=FEATURES
        )

        predicted = logistic_model.predict(new_data)[0]
        probabilities = logistic_model.predict_proba(new_data)[0]

        st.success(f"모델의 예측 결과: **{predicted}**")

        probability_df = pd.DataFrame({
            "최종 결과": logistic_model.classes_,
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
            "이 확률은 학습 데이터의 패턴을 바탕으로 계산한 값이며 "
            "실제 경기 결과를 보장하지 않는다."
        )

    st.divider()

    st.subheader("6. 실제 결과와 예측 결과 비교")

    comparison = analysis_df[
        ["날짜", "상대팀", "한국_전반득점", "상대_전반득점"]
    ].copy()

    comparison["실제 결과"] = y.values
    comparison["예측 결과"] = logistic_predictions
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

    st.subheader("전반전 상황별 최종 결과")

    st.dataframe(
        percentages.round(1),
        use_container_width=True
    )

    st.write(f"""
    전반전에 앞선 경기의 최종 승리 비율은
    {ahead_win_rate:.1f}%였다.

    전반전에 뒤진 경기의 최종 패배 비율은
    {behind_loss_rate:.1f}%였다.
    """)

    st.divider()

    st.subheader("로지스틱 회귀와 베이스라인 비교")

    st.write(f"""
    베이스라인 정확도는 {baseline_accuracy:.1%}이고,
    로지스틱 회귀 정확도는 {logistic_accuracy:.1%}이다.

    따라서 로지스틱 회귀의 정확도는 베이스라인보다
    {accuracy_difference:+.1f}%p 차이가 났다.

    매크로 F1 점수는 베이스라인 {baseline_f1:.1%},
    로지스틱 회귀 {logistic_f1:.1%}로 나타났다.
    """)

    if accuracy_difference > 0:
        st.write(
            "로지스틱 회귀가 베이스라인보다 정확도에서 "
            "더 좋은 결과를 보였다."
        )
    elif accuracy_difference < 0:
        st.write(
            "로지스틱 회귀가 베이스라인보다 정확도에서 "
            "낮은 결과를 보였다. 복잡한 모델을 사용했다고 "
            "항상 성능이 좋아지는 것은 아니라는 점을 알 수 있다."
        )
    else:
        st.write(
            "두 모델의 정확도가 같았다. "
            "현재 데이터에서는 로지스틱 회귀의 추가적인 "
            "정확도 향상이 나타나지 않았다."
        )

    st.divider()

    st.subheader("분석의 한계")

    st.write("""
    분석에 사용한 경기 수가 많지 않고, 전반전 득점 두 가지 변수만
    사용했기 때문에 상대 팀의 실력이나 경기 장소 등 다른 요소를
    충분히 반영하지 못했다.

    따라서 모델의 결과를 모든 축구 경기에 적용하기는 어렵다.
    """)

# ==========================================
# 8. 최종 보고서
# ==========================================
elif menu == "최종 보고서":
    st.header("📝 최종 보고서")

    st.caption(
        "탐구 질문에 대한 답변을 실제 데이터 분석 결과를 바탕으로 정리했다."
    )

    # 1. 주제 선택 이유
    st.subheader("1. 주제를 왜 이것으로 정했습니까?")

    st.write("""
    평소 축구 경기를 보면서 전반전에 앞서거나 뒤지는 상황이
    최종 경기 결과에 어떤 영향을 주는지 궁금했다.

    전반전에 앞서면 실제로 승리할 가능성이 높은지,
    뒤지고 있어도 역전할 수 있는지 직접 확인하고 싶었다.

    그래서 대한민국 축구 국가대표팀의 경기 데이터를 이용해
    전반전 상황과 최종 결과의 관계를 분석하는 주제를 선택했다.
    """)

    # 2. 그래프와 숫자의 의미
    st.subheader("2. 이 그래프와 숫자는 무엇을 뜻합니까?")

    st.write("""
    전반전 상황별 최종 결과 비율 그래프는 전반전에 앞섬,
    동점, 뒤짐이라는 세 가지 상황에서 최종 승리, 무승부,
    패배가 각각 얼마나 나타났는지 보여 준다.
    """)

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

    st.write(f"""
    분석 결과, 전반전에 앞선 경기의 최종 승리 비율은
    {ahead_win_rate:.1f}%였고, 전반전에 뒤진 경기의 최종 패배 비율은
    {behind_loss_rate:.1f}%였다.

    로지스틱 회귀의 혼동 행렬은 실제 경기 결과와 모델이 예측한
    경기 결과가 얼마나 일치하는지 보여 준다.
    대각선의 숫자가 클수록 해당 결과를 정확히 분류한 경기가 많다는 뜻이다.
    """)

    st.image if False else None

    cm_df = pd.DataFrame(
        cm,
        index=[f"실제 {r}" for r in RESULTS],
        columns=[f"예측 {r}" for r in RESULTS]
    )

    fig_report_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="실제 결과와 예측 결과 비교"
    )

    st.plotly_chart(
        fig_report_cm,
        use_container_width=True
    )

    # 3. 베이스라인보다 얼마나 나아졌는가?
    st.subheader("3. 아무 모델도 쓰지 않았을 때보다 얼마나 나아졌습니까?")

    st.write(f"""
    베이스라인은 학습 데이터에서 가장 많이 등장한
    '{majority_result}'만 예측하는 단순한 기준 모델이다.

    베이스라인 정확도는 {baseline_accuracy:.1%}이고,
    로지스틱 회귀 정확도는 {logistic_accuracy:.1%}이다.

    따라서 로지스틱 회귀는 베이스라인보다
    **{accuracy_difference:+.1f}%p** 차이가 났다.
    """)

    baseline_report = pd.DataFrame({
        "모델": ["베이스라인", "로지스틱 회귀"],
        "정확도": [
            baseline_accuracy * 100,
            logistic_accuracy * 100
        ],
        "매크로 F1 점수": [
            baseline_f1 * 100,
            logistic_f1 * 100
        ]
    }).round(1)

    st.dataframe(
        baseline_report,
        use_container_width=True
    )

    if accuracy_difference > 0:
        st.success(
            "로지스틱 회귀가 베이스라인보다 정확도가 높았다."
        )
    elif accuracy_difference < 0:
        st.warning(
            "로지스틱 회귀가 베이스라인보다 정확도가 낮았다. "
            "모델을 사용했다고 반드시 성능이 향상되는 것은 아니다."
        )
    else:
        st.info(
            "두 모델의 정확도는 같았다."
        )

    st.write(f"""
    정확도뿐 아니라 매크로 F1 점수도 비교하였다.
    베이스라인은 {baseline_f1:.1%}, 로지스틱 회귀는
    {logistic_f1:.1%}였다.

    이를 통해 전체 정답률뿐 아니라 승리, 무승부, 패배를
    각각 얼마나 균형 있게 분류하는지도 확인하였다.
    """)

    # 4. 믿을 수 없는 이유
    st.subheader("4. 이 결과를 믿을 수 없는 이유를 하나 말해 보십시오.")

    st.write(f"""
    가장 큰 한계는 분석에 사용한 데이터가
    {len(analysis_df)}경기로 많지 않다는 점이다.

    또한 모델은 한국과 상대 팀의 전반전 득점만 사용했다.
    상대 팀의 실력, 경기 장소, 선수 구성, 퇴장 여부 등
    다른 경기 요인은 반영하지 못했다.

    따라서 이번 결과가 모든 국가대표팀 경기에서도
    똑같이 나타난다고 확신할 수 없다.
    """)

    # 5. 새롭게 알게 된 점
    st.subheader("5. 이 결과에서 무엇을 새로 알게 되었습니까?")

    st.write(f"""
    전반전 상황별 분석을 통해 앞서고 있는 경기와 뒤지고 있는 경기에서
    최종 결과의 비율이 다르게 나타난다는 점을 확인했다.

    특히 전반전에 앞선 경기의 최종 승리 비율은
    {ahead_win_rate:.1f}%였으며, 뒤진 경기의 최종 패배 비율은
    {behind_loss_rate:.1f}%였다.

    또한 전반전 득점만으로 최종 결과를 분류하는 로지스틱 회귀 모델을
    만들고, 베이스라인과 비교하면서 복잡한 모델을 사용하더라도
    항상 성능이 좋아지는 것은 아니라는 점을 알게 되었다.
    """)

    st.subheader("6. 느낀 점")

    st.write("""
    이번 탐구를 통해 축구 경기의 흐름을 단순한 느낌이 아니라
    실제 데이터와 통계로 확인할 수 있었다.

    데이터를 정리하고 그래프로 표현하는 과정에서 경기 상황별
    차이를 더 쉽게 이해할 수 있었다.

    또한 로지스틱 회귀를 이용해 최종 결과를 분류하고,
    베이스라인과 비교하면서 머신러닝 모델의 성능을
    객관적으로 평가하는 방법을 배웠다.

    앞으로는 더 많은 경기 데이터와 다양한 변수를 활용하면
    분석 결과를 더 깊이 이해할 수 있을 것이라고 생각했다.
    """)

    st.divider()

    st.caption(
        "자료: 대한민국 축구 국가대표팀 경기 데이터"
    )

    st.caption(
        "분석 방법: 전반전 상황별 비율 분석, "
        "다중 로지스틱 회귀, 베이스라인 비교, 5겹 교차검증"
    )
