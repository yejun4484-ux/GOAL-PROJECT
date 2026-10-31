
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
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="대한민국 축구 경기 데이터 분석",
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

SCORE_COLUMNS = [
    "한국_전반득점",
    "상대_전반득점",
    "한국_최종득점",
    "상대_최종득점"
]

REQUIRED_COLUMNS = [
    "날짜", "연도", "상대팀", "대회",
    "한국_전반득점", "상대_전반득점",
    "한국_최종득점", "상대_최종득점",
    "전반전상황", "최종결과"
]


# ==========================================
# 2. 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():
    urls = [DATA_URL, unquote(DATA_URL)]

    for url in urls:
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


df_original = load_data()

if df_original.empty:
    st.error("데이터를 불러오지 못했습니다. 데이터 주소를 확인해 주세요.")
    st.stop()

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df_original.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()


# ==========================================
# 3. 전처리 전 데이터 확인
# ==========================================
df_before = df_original.copy()

for col in SCORE_COLUMNS + ["연도"]:
    df_before[col] = pd.to_numeric(
        df_before[col], errors="coerce"
    )

# ==========================================
# 4. 전처리
# ==========================================
df_clean = df_original.copy()

# 숫자형으로 변환
for col in SCORE_COLUMNS + ["연도"]:
    df_clean[col] = pd.to_numeric(
        df_clean[col], errors="coerce"
    )

# 문자열 공백 정리
for col in ["상대팀", "대회", "전반전상황", "최종결과"]:
    df_clean[col] = (
        df_clean[col]
        .astype("string")
        .str.strip()
    )

# 빈 문자열은 결측치로 변경
for col in ["상대팀", "대회", "전반전상황", "최종결과"]:
    df_clean[col] = df_clean[col].replace("", pd.NA)

rows_before = len(df_clean)

# 분석에 꼭 필요한 열의 결측치가 있는 행 제거
df_clean = df_clean.dropna(
    subset=FEATURES + ["최종결과", "전반전상황"]
)

rows_after_missing = len(df_clean)

# 중복 행 제거
df_clean = df_clean.drop_duplicates()

rows_after = len(df_clean)

df_clean = df_clean.reset_index(drop=True)

# 분석에 사용할 데이터
analysis_df = df_clean[
    df_clean["전반전상황"].isin(SITUATIONS)
    & df_clean["최종결과"].isin(RESULTS)
].copy()

analysis_df = analysis_df.dropna(
    subset=FEATURES + ["최종결과"]
).reset_index(drop=True)

if analysis_df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()


# ==========================================
# 5. 이상치 탐색
# ==========================================
outlier_info = []

for col in SCORE_COLUMNS:
    values = df_clean[col].dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_count = (
        (values < lower) | (values > upper)
    ).sum()

    outlier_info.append({
        "열 이름": col,
        "1사분위수": q1,
        "3사분위수": q3,
        "이상치 판단 하한": lower,
        "이상치 판단 상한": upper,
        "이상치 후보 개수": int(outlier_count)
    })

outlier_df = pd.DataFrame(outlier_info)


# ==========================================
# 6. 로지스틱 회귀와 베이스라인
# ==========================================
X = analysis_df[FEATURES]
y = analysis_df["최종결과"]

class_counts = y.value_counts()

if len(class_counts) < 2 or class_counts.min() < 5:
    st.error(
        "일부 결과의 경기 수가 부족하여 "
        "5겹 교차검증을 수행할 수 없습니다."
    )
    st.stop()

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        max_iter=3000,
        solver="lbfgs"
    ))
])

baseline_model = DummyClassifier(
    strategy="most_frequent"
)

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

# 전체 데이터로 예측용 모델 학습
logistic_model.fit(X, y)

# 성능 계산
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

majority_result = y.value_counts().idxmax()


# ==========================================
# 7. 통계 계산
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
# 8. 메뉴
# ==========================================
st.title("⚽ 대한민국 축구 국가대표팀 경기 분석")

st.caption(
    "전반전 경기 상황과 최종 결과의 관계를 데이터로 알아본다."
)

menu = st.sidebar.radio(
    "분석 메뉴",
    [
        "탐구 주제",
        "데이터 살펴보기·전처리",
        "전반전 상황별 경기 수",
        "전반전 상황별 최종 결과",
        "로지스틱 회귀 분석",
        "경기 기록 확인",
        "최종 보고서"
    ]
)


# ==========================================
# 9. 탐구 주제
# ==========================================
if menu == "탐구 주제":

    st.header("🔎 탐구 주제")

    st.subheader(
        "대한민국 축구 국가대표팀의 전반전 경기 상황에 따라 "
        "최종 경기 결과는 어떻게 달라질까?"
    )

    st.write("""
    축구 경기를 보면서 전반전에 앞서거나 뒤지는 상황이
    최종 경기 결과와 어떤 관계가 있는지 궁금했다.

    전반전에 앞서면 실제로 승리할 가능성이 높은지,
    뒤지고 있어도 역전할 수 있는지 데이터를 통해 확인하고 싶었다.

    그래서 대한민국 축구 국가대표팀의 경기 기록을 이용하여
    전반전 상황과 최종 결과의 관계를 분석하는 주제를 선택했다.
    """)

    st.subheader("분석 방법")

    st.markdown("""
    1. 대한민국 국가대표팀 경기 데이터를 수집한다.
    2. 데이터의 결측치, 이상치, 편향을 확인한다.
    3. 전반전 상황별 최종 결과 비율을 비교한다.
    4. 로지스틱 회귀로 최종 결과를 분류한다.
    5. 베이스라인과 모델 성능을 비교한다.
    """)


# ==========================================
# 10. 데이터 살펴보기·전처리
# ==========================================
elif menu == "데이터 살펴보기·전처리":

    st.header("② 데이터 살펴보기·전처리")

    st.info(
        "데이터를 처음 열었을 때 발견한 점, 결측치와 이상치의 차이, "
        "처리 기준, 편향 및 전처리 전후 변화를 확인한다."
    )

    # 행과 열
    st.subheader("1. 데이터의 행과 열")

    c1, c2, c3 = st.columns(3)

    c1.metric("전처리 전 행 수", f"{len(df_before)}개")
    c2.metric("열 수", f"{df_before.shape[1]}개")
    c3.metric("전처리 후 행 수", f"{len(df_clean)}개")

    st.write(
        f"원본 데이터는 **{df_before.shape[0]}행 "
        f"{df_before.shape[1]}열**로 구성되어 있다."
    )

    st.subheader("원본 데이터 미리 보기")
    st.dataframe(df_original.head(10), use_container_width=True)

    # 결측치
    st.divider()
    st.subheader("2. 결측치 확인")

    missing_table = pd.DataFrame({
        "열 이름": df_original.columns,
        "결측치 개수": [
            int(df_original[col].isna().sum())
            for col in df_original.columns
        ]
    })

    st.dataframe(
        missing_table,
        use_container_width=True
    )

    total_missing = int(df_original.isna().sum().sum())

    st.write(f"전체 결측치 개수는 **{total_missing}개**이다.")

    st.markdown("""
    **결측치와 이상치의 차이**

    - 결측치: 데이터 값이 비어 있거나 기록되지 않은 경우
    - 이상치: 다른 데이터와 비교했을 때 유난히 크거나 작은 값

    결측치는 값이 없는 것이고, 이상치는 값이 존재하지만
    다른 값들과 차이가 큰 경우라는 점에서 다르다.
    """)

    st.subheader("결측치 처리 기준")

    st.write("""
    이번 분석에서는 한국 전반전 득점, 상대 전반전 득점,
    전반전 상황, 최종 결과처럼 분석에 꼭 필요한 값이 비어 있는
    행을 제거하였다.

    최종 결과가 없는 경기는 정답을 알 수 없으므로
    모델 학습과 평가에 사용할 수 없기 때문이다.
    """)

    # 이상치
    st.divider()
    st.subheader("3. 이상치 확인")

    st.write("""
    이상치는 사분위 범위(IQR)를 이용하여 탐색하였다.

    1사분위수에서 3사분위수를 뺀 값을 IQR이라고 한다.
    IQR을 이용해 일반적인 범위를 벗어나는 값을 이상치 후보로 표시한다.
    """)

    if not outlier_df.empty:
        st.dataframe(
            outlier_df.round(2),
            use_container_width=True
        )

        st.write(
            "아래 그래프는 득점 데이터의 분포와 이상치 후보를 확인하기 위한 것이다."
        )

        selected_score = st.selectbox(
            "확인할 득점 열",
            SCORE_COLUMNS
        )

        fig_box = px.box(
            df_clean,
            y=selected_score,
            points="all",
            title=f"{selected_score} 분포와 이상치 후보"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

    st.subheader("이상치 처리 기준")

    st.write("""
    득점 데이터에서 큰 값이 발견되더라도 실제 경기에서 나온
    정상적인 득점일 수 있다.

    따라서 IQR 기준에 해당하는 값은 이상치 후보로만 표시하고,
    실제 경기 기록을 확인하지 않은 상태에서 삭제하지 않았다.

    즉, 이번 탐구에서는 이상치 후보를 확인하되
    근거 없는 데이터 삭제는 하지 않았다.
    """)

    # 편향
    st.divider()
    st.subheader("4. 데이터의 편향 확인")

    st.write("""
    편향은 데이터가 특정 결과나 집단에 지나치게 많이 모여
    균형이 맞지 않는 상태를 뜻한다.

    이번 데이터에서는 승리, 무승부, 패배의 경기 수가
    서로 다른지 확인하였다.
    """)

    class_distribution = (
        analysis_df["최종결과"]
        .value_counts()
        .reindex(RESULTS, fill_value=0)
        .reset_index()
    )

    class_distribution.columns = ["최종 결과", "경기 수"]

    class_distribution["비율(%)"] = (
        class_distribution["경기 수"]
        / len(analysis_df) * 100
    ).round(1)

    st.dataframe(
        class_distribution,
        use_container_width=True
    )

    fig_bias = px.bar(
        class_distribution,
        x="최종 결과",
        y="경기 수",
        text="경기 수",
        title="최종 결과별 경기 수",
        color="최종 결과"
    )

    st.plotly_chart(
        fig_bias,
        use_container_width=True
    )

    min_count = class_distribution["경기 수"].min()
    max_count = class_distribution["경기 수"].max()

    if min_count == max_count:
        st.write(
            "세 결과의 경기 수가 같아 결과별 경기 수가 균형을 이룬다."
        )
    else:
        st.write(
            f"가장 많은 결과는 **{majority_result}**이고, "
            f"가장 적은 결과와 경기 수 차이는 "
            f"**{max_count - min_count}경기**이다."
        )

        st.write("""
        따라서 결과별 경기 수에 차이가 있는 편향이 존재한다.
        모델이 많은 결과를 더 잘 맞히는지 주의해서 살펴볼 필요가 있다.
        """)

    # 전처리 전후 비교
    st.divider()
    st.subheader("5. 전처리 전후 비교")

    before_after = pd.DataFrame({
        "항목": [
            "행 수",
            "한국 전반전 득점 평균",
            "상대 전반전 득점 평균",
            "한국 최종 득점 평균",
            "상대 최종 득점 평균"
        ],
        "전처리 전": [
            len(df_before),
            df_before["한국_전반득점"].mean(),
            df_before["상대_전반득점"].mean(),
            df_before["한국_최종득점"].mean(),
            df_before["상대_최종득점"].mean()
        ],
        "전처리 후": [
            len(df_clean),
            df_clean["한국_전반득점"].mean(),
            df_clean["상대_전반득점"].mean(),
            df_clean["한국_최종득점"].mean(),
            df_clean["상대_최종득점"].mean()
        ]
    })

    st.dataframe(
        before_after.round(2),
        use_container_width=True
    )

    removed_rows = len(df_before) - len(df_clean)

    st.write(
        f"전처리 과정에서 총 **{removed_rows}행**이 제거되었다."
    )

    st.write("""
    필요한 값이 비어 있는 행과 중복 행을 제거한 뒤
    행 수와 득점 평균이 어떻게 달라졌는지 비교하였다.

    이상치 후보는 실제 경기 기록일 가능성이 있어 삭제하지 않았다.
    """)

    # 전처리 결정 로그
    st.divider()
    st.subheader("6. 전처리 결정 기록")

    decision_log = """
[전처리 결정 기록]

1. 숫자형 열은 숫자 자료형으로 변환하였다.
2. 문자열 열의 앞뒤 공백을 제거하였다.
3. 분석에 필요한 전반전 득점과 최종 결과 등이 비어 있는 행은 제거하였다.
4. 중복된 행은 제거하였다.
5. 득점의 이상치 후보는 IQR 방식으로 확인하였다.
6. 실제 경기 기록일 가능성이 있으므로 이상치 후보를 임의로 삭제하지 않았다.
7. 최종 결과별 경기 수를 비교하여 데이터 편향을 확인하였다.
8. 전처리 전후 행 수와 득점 평균을 비교하였다.
9. 로지스틱 회귀에서는 StandardScaler를 학습 데이터에 맞춰 적용하도록 Pipeline을 사용하였다.
10. 교차검증을 통해 모델을 평가하였다.
"""

    st.code(decision_log, language="text")

    st.download_button(
        "전처리 결정 기록 다운로드",
        data=decision_log.encode("utf-8-sig"),
        file_name="전처리_결정기록.txt",
        mime="text/plain"
    )

    st.warning(
        "GitHub 커밋은 본인 계정에서 직접 해야 한다. "
        "아래 안내를 참고해 전처리 코드를 커밋하고 기록을 남기자."
    )

    st.code(
        "git add main.py\n"
        'git commit -m "데이터 전처리 및 분석 기능 추가"\n'
        "git push",
        language="bash"
    )


# ==========================================
# 11. 전반전 상황별 경기 수
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
# 12. 전반전 상황별 최종 결과
# ==========================================
elif menu == "전반전 상황별 최종 결과":

    st.header("⚽ 전반전 상황별 최종 결과")

    st.subheader("최종 결과 비율")

    st.dataframe(
        percentages.round(1),
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


# ==========================================
# 13. 로지스틱 회귀 분석
# ==========================================
elif menu == "로지스틱 회귀 분석":

    st.header("🤖 로지스틱 회귀 분석")

    st.write("""
    한국과 상대 팀의 전반전 득점을 입력 변수로 사용하여
    최종 결과를 승리, 무승부, 패배로 분류한다.
    """)

    st.subheader("1. 모델 성능")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("정확도", f"{logistic_accuracy:.1%}")
    c2.metric("정밀도", f"{logistic_precision:.1%}")
    c3.metric("재현율", f"{logistic_recall:.1%}")
    c4.metric("F1 점수", f"{logistic_f1:.1%}")

    st.divider()

    st.subheader("2. 베이스라인과 비교")

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
        title="베이스라인과 로지스틱 회귀 비교"
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    st.write(
        f"정확도 차이: **{accuracy_difference:+.1f}%p**"
    )

    st.write(
        f"매크로 F1 점수 차이: **{f1_difference:+.1f}%p**"
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
        title="실제 결과와 예측 결과"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    st.write("""
    행은 실제 경기 결과, 열은 모델의 예측 결과이다.
    대각선에 있는 숫자는 정답을 맞힌 경기 수를 뜻한다.
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

    st.subheader("5. 전반전 점수로 최종 결과 예측")

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

        st.success(f"예측 결과: **{predicted}**")

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

        st.plotly_chart(fig_prob, use_container_width=True)

        st.caption(
            "예측 확률은 학습 데이터의 패턴을 바탕으로 한 값이며 "
            "실제 경기 결과를 보장하지 않는다."
        )

    st.divider()

    st.subheader("6. 실제 결과와 예측 결과")

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
# 14. 경기 기록 확인
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
        filtered[REQUIRED_COLUMNS],
        use_container_width=True
    )


# ==========================================
# 15. 최종 보고서
# ==========================================
elif menu == "최종 보고서":

    st.header("📝 최종 보고서")

    st.subheader("1. 탐구 주제")

    st.write("""
    대한민국 축구 국가대표팀의 전반전 경기 상황에 따라
    최종 경기 결과는 어떻게 달라질까?
    """)

    st.subheader("2. 주제를 선택한 이유")

    st.write("""
    평소 축구 경기를 보면서 전반전에 앞서거나 뒤지는 상황이
    최종 경기 결과와 어떤 관계가 있는지 궁금했다.

    전반전에 앞서면 실제로 승리할 가능성이 높은지,
    뒤지고 있어도 역전할 수 있는지 데이터를 통해 확인하고 싶었다.

    그래서 대한민국 축구 국가대표팀의 경기 데이터를 활용하여
    전반전 상황과 최종 결과의 관계를 분석하는 주제를 선택했다.
    """)

    st.subheader("3. 데이터 살펴보기와 전처리")

    st.write(
        f"원본 데이터는 {len(df_original)}행 "
        f"{df_original.shape[1]}열로 구성되어 있었다."
    )

    st.write(
        f"분석에 필요한 열의 결측치와 중복 행을 확인하고 처리한 뒤 "
        f"{len(df_clean)}행이 남았다."
    )

    st.write("""
    결측치는 값이 비어 있는 경우이고, 이상치는 다른 데이터에 비해
    유난히 크거나 작은 값이다.

    분석에 필요한 값이 비어 있는 행은 제거하였고,
    중복 행도 제거하였다.

    득점의 이상치 후보는 IQR 방식으로 확인했지만
    실제 경기에서 나온 득점일 수 있으므로 임의로 삭제하지 않았다.

    또한 승리, 무승부, 패배의 경기 수를 비교하여
    데이터의 편향 여부를 확인하였다.
    """)

    st.subheader("4. 그래프와 숫자는 무엇을 뜻하는가?")

    st.write("""
    전반전 상황별 결과 그래프는 전반전에 앞섬, 동점, 뒤짐이라는
    세 가지 상황에서 최종 승리, 무승부, 패배가 각각 얼마나
    나타났는지 보여 준다.
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

    로지스틱 회귀의 혼동 행렬은 실제 경기 결과와
    모델이 예측한 결과가 얼마나 일치하는지를 보여 준다.
    대각선의 숫자는 정답을 맞힌 경기 수를 뜻한다.
    """)

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

    st.subheader("5. 베이스라인보다 얼마나 나아졌는가?")

    st.write(f"""
    베이스라인은 가장 많이 등장한 결과인 '{majority_result}'만
    예측하는 단순한 기준 모델이다.

    베이스라인 정확도는 {baseline_accuracy:.1%}이고,
    로지스틱 회귀 정확도는 {logistic_accuracy:.1%}이다.

    따라서 로지스틱 회귀의 정확도는 베이스라인보다
    **{accuracy_difference:+.1f}%p** 차이가 났다.

    매크로 F1 점수는 베이스라인 {baseline_f1:.1%},
    로지스틱 회귀 {logistic_f1:.1%}였다.
    """)

    if accuracy_difference > 0:
        st.write(
            "로지스틱 회귀가 베이스라인보다 정확도가 높았다."
        )
    elif accuracy_difference < 0:
        st.write(
            "로지스틱 회귀가 베이스라인보다 정확도가 낮았다. "
            "모델을 사용했다고 항상 성능이 향상되는 것은 아니다."
        )
    else:
        st.write(
            "두 모델의 정확도는 같았다."
        )

    st.subheader("6. 이 결과를 믿기 어려운 이유")

    st.write(f"""
    분석에 사용한 경기 수는 {len(analysis_df)}경기로 많지 않다.
    또한 한국과 상대 팀의 전반전 득점 두 가지 변수만 사용했기 때문에
    상대 팀의 실력, 경기 장소, 선수 구성, 퇴장 여부 등
    다른 요소는 충분히 반영하지 못했다.

    따라서 이번 결과가 모든 축구 경기에서도 똑같이 나타난다고
    확신하기 어렵다.
    """)

    st.subheader("7. 이 결과에서 새롭게 알게 된 점")

    st.write(f"""
    전반전 상황별 분석을 통해 앞서고 있는 경기와 뒤지고 있는 경기에서
    최종 결과의 비율이 다르게 나타난다는 점을 확인했다.

    특히 전반전에 앞선 경기의 최종 승리 비율은
    {ahead_win_rate:.1f}%였고, 뒤진 경기의 최종 패배 비율은
    {behind_loss_rate:.1f}%였다.

    또한 로지스틱 회귀와 베이스라인을 비교하면서
    복잡한 모델을 사용하더라도 항상 성능이 좋아지는 것은 아니라는
    점을 새롭게 알게 되었다.
    """)

    st.subheader("8. 느낀 점")

    st.write("""
    이번 탐구를 통해 축구 경기의 흐름을 단순한 느낌이 아니라
    실제 데이터와 통계로 확인할 수 있었다.

    데이터를 정리하고 그래프로 표현하면서 경기 상황별 차이를
    더 쉽게 이해할 수 있었다.

    또한 로지스틱 회귀를 이용해 최종 결과를 분류하고,
    베이스라인과 비교하면서 머신러닝 모델의 성능을
    객관적으로 평가하는 방법을 배웠다.

    앞으로는 더 많은 경기 데이터와 다양한 변수를 활용하면
    분석을 더 깊이 진행할 수 있을 것이라고 생각했다.
    """)

    st.divider()

    st.caption(
        "자료: 대한민국 축구 국가대표팀 경기 데이터"
    )

    st.caption(
        "분석 방법: 데이터 전처리, 기술통계, "
        "다중 로지스틱 회귀, 베이스라인 비교, 5겹 교차검증"
    )
