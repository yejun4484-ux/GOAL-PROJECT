
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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

FEATURES = ["한국_전반득점", "상대_전반득점"]
RESULTS = ["승리", "무승부", "패배"]
SITUATIONS = ["앞섬", "동점", "뒤짐"]

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
    st.error("데이터를 불러오지 못했습니다. GitHub 주소를 확인해 주세요.")
    st.stop()

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in original_df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()


# ==========================================
# 3. 전처리 함수
# ==========================================
def preprocess_data(data):
    df = data.copy()

    # 숫자형으로 변환
    for col in SCORE_COLUMNS + ["연도"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 문자열 공백 정리
    for col in ["상대팀", "대회", "전반전상황", "최종결과"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # 분석에 필요한 열의 결측치 제거
    df = df.dropna(
        subset=FEATURES + ["최종결과", "전반전상황"]
    )

    # 중복 행 제거
    df = df.drop_duplicates()

    # 분석 대상 범주만 사용
    df = df[
        df["최종결과"].isin(RESULTS)
        & df["전반전상황"].isin(SITUATIONS)
    ].copy()

    return df.reset_index(drop=True)


clean_df = preprocess_data(original_df)

if clean_df.empty:
    st.error("전처리 후 분석할 데이터가 없습니다.")
    st.stop()


# ==========================================
# 4. 모델 3개 선정
# ==========================================
def make_models(k_value=5):
    return {
        "로지스틱 회귀": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000))
        ]),

        "의사결정트리": DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        ),

        "k-최근접 이웃": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=k_value))
        ])
    }


# ==========================================
# 5. 의사결정트리 Plotly 시각화 함수
# ==========================================
def draw_decision_tree_plotly(model):
    """
    학습된 의사결정트리를 Plotly로 시각화한다.
    한국어 속성명과 실제 학습된 분기 기준을 표시한다.
    """

    tree = model.tree_

    class_names = [
        str(value) for value in model.classes_
    ]

    feature_names = {
        0: "한국 전반전 득점",
        1: "상대 팀 전반전 득점"
    }

    positions = {}
    leaf_order = [0]

    # 노드 위치 계산
    def assign_positions(node, depth):

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        # 자식 노드가 없으면 리프 노드
        if left_child == right_child:
            x_pos = leaf_order[0]
            leaf_order[0] += 1

        else:
            left_x = assign_positions(
                left_child,
                depth + 1
            )

            right_x = assign_positions(
                right_child,
                depth + 1
            )

            x_pos = (left_x + right_x) / 2

        positions[node] = (x_pos, -depth)

        return x_pos

    assign_positions(0, 0)

    fig = go.Figure()

    # 노드 사이 연결선
    for node in range(tree.node_count):

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        # 리프 노드는 연결선이 필요하지 않음
        if left_child == right_child:
            continue

        x_parent, y_parent = positions[node]
        x_left, y_left = positions[left_child]
        x_right, y_right = positions[right_child]

        # 왼쪽 가지: 조건을 만족하는 경우
        fig.add_trace(go.Scatter(
            x=[x_parent, x_left],
            y=[y_parent, y_left],
            mode="lines",
            line=dict(
                color="#2563EB",
                width=2
            ),
            hoverinfo="skip",
            showlegend=False
        ))

        # 오른쪽 가지: 조건을 만족하지 않는 경우
        fig.add_trace(go.Scatter(
            x=[x_parent, x_right],
            y=[y_parent, y_right],
            mode="lines",
            line=dict(
                color="#DC2626",
                width=2
            ),
            hoverinfo="skip",
            showlegend=False
        ))

        # 가지에 예 / 아니오 표시
        fig.add_annotation(
            x=(x_parent + x_left) / 2,
            y=(y_parent + y_left) / 2,
            text="예",
            showarrow=False,
            bgcolor="white",
            font=dict(
                size=11,
                color="#2563EB"
            )
        )

        fig.add_annotation(
            x=(x_parent + x_right) / 2,
            y=(y_parent + y_right) / 2,
            text="아니오",
            showarrow=False,
            bgcolor="white",
            font=dict(
                size=11,
                color="#DC2626"
            )
        )

    # 노드별 설명 상자
    for node in range(tree.node_count):

        x_pos, y_pos = positions[node]

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        is_leaf = left_child == right_child

        # 해당 노드에서 각 결과의 경기 수
        class_counts = tree.value[node][0]

        predicted_index = int(class_counts.argmax())
        predicted_class = class_names[predicted_index]

        sample_count = int(tree.n_node_samples[node])

        if is_leaf:

            label = (
                f"최종 예측: {predicted_class}<br>"
                f"해당 경기 수: {sample_count}"
            )

            background = "#DCFCE7"

        else:

            feature_index = int(tree.feature[node])

            feature_label = feature_names.get(
                feature_index,
                str(feature_index)
            )

            threshold = tree.threshold[node]

            label = (
                f"{feature_label} ≤ {threshold:.1f}?<br>"
                f"해당 경기 수: {sample_count}<br>"
                f"현재 주요 결과: {predicted_class}"
            )

            background = "#DBEAFE"

        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=label,
            showarrow=False,
            align="center",
            bgcolor=background,
            bordercolor="#64748B",
            borderwidth=1,
            borderpad=7,
            font=dict(
                family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans CJK KR, sans-serif",
                size=12,
                color="#111827"
            )
        )

    max_depth = int(tree.max_depth)
    leaf_total = max(leaf_order[0], 1)

    fig.update_layout(
        title="전반전 득점으로 최종 경기 결과를 분류하는 의사결정트리",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(
            520,
            180 * (max_depth + 1)
        ),
        margin=dict(
            l=30,
            r=30,
            t=80,
            b=30
        ),
        font=dict(
            family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans CJK KR, sans-serif",
            color="#111827"
        ),
        xaxis=dict(
            visible=False,
            range=[-1, leaf_total],
            fixedrange=True
        ),
        yaxis=dict(
            visible=False,
            range=[-max_depth - 0.7, 0.7],
            fixedrange=True
        )
    )

    return fig


# ==========================================
# 6. 모델 평가
# ==========================================
@st.cache_data
def evaluate_models(data, k_value=5):

    X = data[FEATURES]
    y = data["최종결과"]

    counts = y.value_counts()

    if len(counts) < 2 or counts.min() < 5:
        return None

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = make_models(k_value)

    # 베이스라인 모델
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline_predictions = cross_val_predict(
        baseline,
        X,
        y,
        cv=cv
    )

    predictions = {
        "베이스라인": baseline_predictions
    }

    scores = [{
        "모델": "베이스라인",
        "정확도": accuracy_score(
            y,
            baseline_predictions
        ),
        "정밀도": precision_score(
            y,
            baseline_predictions,
            average="macro",
            zero_division=0
        ),
        "재현율": recall_score(
            y,
            baseline_predictions,
            average="macro",
            zero_division=0
        ),
        "F1 점수": f1_score(
            y,
            baseline_predictions,
            average="macro",
            zero_division=0
        )
    }]

    # 세 모델 평가
    for name, model in models.items():

        pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv
        )

        predictions[name] = pred

        scores.append({
            "모델": name,
            "정확도": accuracy_score(y, pred),
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
            )
        })

    score_df = pd.DataFrame(scores)

    # 전체 데이터로 최종 모델 학습
    fitted_models = make_models(k_value)

    for model in fitted_models.values():
        model.fit(X, y)

    prediction_df = data[
        [
            "날짜",
            "상대팀",
            "한국_전반득점",
            "상대_전반득점"
        ]
    ].copy()

    prediction_df["실제 결과"] = y.values

    for name in models:
        prediction_df[name + " 예측"] = predictions[name]

    return score_df, prediction_df, fitted_models


# ==========================================
# 7. KNN의 K값 설정
# ==========================================
max_k = max(
    1,
    min(15, int(len(clean_df) * 0.8) - 1)
)

k_value = st.sidebar.slider(
    "kNN의 K값(K)",
    min_value=1,
    max_value=max_k,
    value=min(5, max_k),
    help=(
        "가까운 과거 경기 몇 개를 참고할지 정합니다. "
        "홀수를 사용하면 다수결 동률을 줄이는 데 도움이 됩니다."
    )
)

evaluation = evaluate_models(
    clean_df,
    k_value
)

if evaluation is None:
    st.error(
        "각 결과 범주의 경기 수가 부족하여 "
        "5겹 교차검증을 진행할 수 없습니다."
    )
    st.stop()

score_df, prediction_df, fitted_models = evaluation


# ==========================================
# 8. 공통 통계
# ==========================================
counts = pd.crosstab(
    clean_df["전반전상황"],
    clean_df["최종결과"]
).reindex(
    index=SITUATIONS,
    columns=RESULTS,
    fill_value=0
)

percentages = counts.div(
    counts.sum(axis=1).replace(0, 1),
    axis=0
) * 100


# ==========================================
# 9. 메뉴
# ==========================================
st.title("⚽ 대한민국 축구 국가대표팀 경기 분석")

st.caption(
    "전반전 경기 상황과 최종 결과의 관계를 데이터로 분석한다."
)

menu = st.sidebar.radio(
    "분석 메뉴",
    [
        "탐구 주제",
        "데이터 살펴보기",
        "데이터 전처리",
        "모델 선정과 원리",
        "모델 학습 결과 비교",
        "전반전 상황별 최종 결과",
        "모델 시각화 및 직접 예측",
        "경기 기록",
        "최종 보고서"
    ]
)


# ==========================================
# 10. 탐구 주제
# ==========================================
if menu == "탐구 주제":

    st.header("① 탐구 주제")

    st.subheader(
        "대한민국 축구 국가대표팀의 전반전 경기 상황에 따라 "
        "최종 경기 결과는 어떻게 달라질까?"
    )

    st.write("""
    축구 경기를 보면서 전반전에 앞서거나 뒤지는 상황이
    최종 경기 결과와 어떤 관계가 있는지 궁금했다.

    전반전에 앞서면 실제로 승리할 가능성이 높은지,
    뒤지고 있어도 역전할 수 있는지 데이터를 통해 확인하고 싶었다.

    그래서 대한민국 축구 국가대표팀의 경기 기록을 활용하여
    전반전 상황과 최종 결과의 관계를 분석하기로 했다.
    """)

    st.subheader("분석 목표")

    st.markdown("""
    - 전반전 상황별 최종 결과의 비율을 비교한다.
    - 전반전 득점을 이용해 최종 결과를 분류한다.
    - 세 가지 분류 모델의 성능을 비교한다.
    - 베이스라인보다 모델이 얼마나 나아졌는지 확인한다.
    """)

    st.subheader("회귀·분류·군집 중 무엇이 적합할까?")

    st.markdown("""
    **분류가 가장 적합하다.**

    이번 탐구에서 예측하고 싶은 값은 최종 득점처럼 연속적인 숫자가 아니라
    승리, 무승부, 패배라는 세 가지 범주이다.

    따라서 범주를 예측하는 분류 모델을 사용한다.

    - 회귀: 연속적인 숫자를 예측할 때 적합하다.
    - 분류: 정해진 범주를 예측할 때 적합하다.
    - 군집: 정답 없이 비슷한 데이터끼리 묶을 때 적합하다.
    """)


# ==========================================
# 11. 데이터 살펴보기
# ==========================================
elif menu == "데이터 살펴보기":

    st.header("② 데이터 살펴보기")

    c1, c2, c3 = st.columns(3)

    c1.metric("원본 행 수", len(original_df))
    c2.metric("열 수", original_df.shape[1])
    c3.metric("분석 데이터 행 수", len(clean_df))

    st.subheader("원본 데이터")

    st.dataframe(
        original_df.head(10),
        use_container_width=True
    )

    st.subheader("데이터 속성별 유형")

    type_table = pd.DataFrame({
        "속성": REQUIRED_COLUMNS,
        "유형": [
            "날짜형",
            "수치형",
            "범주형",
            "범주형",
            "수치형(이산형)",
            "수치형(이산형)",
            "수치형(이산형)",
            "수치형(이산형)",
            "범주형",
            "범주형"
        ],
        "적합한 시각화": [
            "선 그래프",
            "막대그래프",
            "막대그래프",
            "막대그래프",
            "막대그래프·산점도",
            "막대그래프·산점도",
            "막대그래프",
            "막대그래프",
            "막대그래프",
            "막대그래프"
        ]
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
        ]
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.write(
        f"전체 결측치 개수: **{int(original_df.isna().sum().sum())}개**"
    )


# ==========================================
# 12. 데이터 전처리
# ==========================================
elif menu == "데이터 전처리":

    st.header("③ 데이터 전처리")

    st.subheader("전처리 기준")

    st.markdown("""
    1. 득점과 연도 열을 숫자형으로 변환한다.
    2. 문자열 앞뒤의 공백을 제거한다.
    3. 분석에 필요한 값이 비어 있는 행을 제거한다.
    4. 중복 행을 제거한다.
    5. 분석 대상인 승리·무승부·패배 기록만 사용한다.
    6. 이상치 후보는 확인하되 실제 경기 기록일 수 있으므로 임의로 삭제하지 않는다.
    """)

    st.subheader("전처리 전후 비교")

    comparison = pd.DataFrame({
        "항목": [
            "행 수",
            "한국 전반전 득점 평균",
            "상대 전반전 득점 평균",
            "한국 최종 득점 평균",
            "상대 최종 득점 평균"
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
            ).mean()
        ],
        "전처리 후": [
            len(clean_df),
            clean_df["한국_전반득점"].mean(),
            clean_df["상대_전반득점"].mean(),
            clean_df["한국_최종득점"].mean(),
            clean_df["상대_최종득점"].mean()
        ]
    })

    st.dataframe(
        comparison.round(2),
        use_container_width=True
    )

    st.subheader("최종 결과별 데이터 편향")

    result_counts = (
        clean_df["최종결과"]
        .value_counts()
        .reindex(RESULTS, fill_value=0)
        .reset_index()
    )

    result_counts.columns = [
        "최종 결과",
        "경기 수"
    ]

    result_counts["비율(%)"] = (
        result_counts["경기 수"] / len(clean_df) * 100
    ).round(1)

    st.dataframe(
        result_counts,
        use_container_width=True
    )

    fig = px.bar(
        result_counts,
        x="최종 결과",
        y="경기 수",
        text="경기 수",
        title="최종 결과별 경기 수"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("이상치 확인")

    outlier_rows = []

    for col in SCORE_COLUMNS:

        values = clean_df[col].dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((values < lower) | (values > upper)).sum()
        )

        outlier_rows.append({
            "속성": col,
            "이상치 후보 개수": count,
            "하한": lower,
            "상한": upper
        })

    st.dataframe(
        pd.DataFrame(outlier_rows).round(2),
        use_container_width=True
    )

    st.write("""
    이상치는 다른 데이터에 비해 유난히 크거나 작은 값이다.
    결측치는 값이 비어 있는 경우라는 점에서 다르다.

    득점이 크다는 이유만으로 잘못된 데이터라고 판단할 수 없으므로
    이상치 후보를 표시하되 임의로 삭제하지 않았다.
    """)

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
        mime="text/csv"
    )


# ==========================================
# 13. 모델 선정과 원리
# ==========================================
elif menu == "모델 선정과 원리":

    st.header("④ 모델 선정과 원리")

    st.subheader("왜 분류 모델을 선택했을까?")

    st.write("""
    이번 탐구의 목표는 전반전 득점을 바탕으로 최종 결과를
    승리, 무승부, 패배 중 하나로 분류하는 것이다.

    예측 대상이 세 가지 범주이므로 분류 모델이 적합하다.
    """)

    st.divider()

    st.subheader("모델 1. 로지스틱 회귀")

    st.write("""
    로지스틱 회귀는 입력 데이터를 바탕으로 각 결과가 나올 확률을 계산한다.

    한국과 상대 팀의 전반전 득점을 입력하면 승리, 무승부, 패배일
    확률을 각각 계산하고 가장 높은 확률의 결과를 선택한다.
    """)

    st.code(
        "입력: 한국 전반전 득점, 상대 전반전 득점\n"
        "→ 각 결과의 확률 계산\n"
        "→ 가장 높은 확률의 결과 선택",
        language="text"
    )

    st.divider()

    st.subheader("모델 2. 의사결정트리")

    st.write("""
    의사결정트리는 여러 질문을 던지며 데이터를 나누는 모델이다.

    이번 탐구에서는 한국의 전반전 득점과 상대 팀의 전반전 득점을
    판단 기준으로 사용한다.

    예를 들어 한국 전반전 득점이 특정 기준 이하인지,
    상대 팀 전반전 득점이 특정 기준 이하인지 등을 판단한다.

    각 질문에 따라 데이터를 나누고 마지막 단계에서
    승리, 무승부, 패배 중 하나를 예측한다.

    트리의 분기 기준은 임의로 만든 것이 아니라
    실제 축구 경기 데이터로 학습하면서 찾아낸 값이다.
    """)

    st.code(
        "한국 전반전 득점 또는 상대 전반전 득점 확인\n"
        "→ 조건에 따라 경기 데이터 분리\n"
        "→ 다음 질문으로 이동\n"
        "→ 승리 / 무승부 / 패배 예측",
        language="text"
    )

    st.divider()

    st.subheader("모델 3. k-최근접 이웃(kNN)")

    st.write(f"""
    kNN은 새로운 데이터와 가장 가까운 기존 데이터 k개를 찾고,
    그 이웃들의 다수결로 결과를 분류한다.

    현재 K값은 **{k_value}**이며, 가장 가까운 경기 {k_value}개의
    최종 결과를 참고한다.

    거리 계산에서 큰 숫자의 영향이 지나치게 커지지 않도록
    StandardScaler로 득점 데이터를 표준화한다.
    """)

    st.code(
        "새로운 경기의 전반전 득점 입력\n"
        "→ 기존 경기와 거리 계산\n"
        f"→ 가까운 {k_value}개 경기 선택\n"
        "→ 다수결로 결과 분류",
        language="text"
    )

    st.divider()

    st.subheader("베이스라인 모델")

    st.write("""
    베이스라인은 가장 많이 등장한 최종 결과만 계속 예측하는
    단순한 기준 모델이다.

    세 분류 모델이 베이스라인보다 좋은 성능을 보이는지 비교한다.
    """)


# ==========================================
# 14. 모델 학습 결과 비교
# ==========================================
elif menu == "모델 학습 결과 비교":

    st.header("⑤ 모델 학습 결과 비교")

    st.info(
        "세 모델은 같은 데이터와 5겹 교차검증으로 평가했다. "
        "각 점수는 검증 과정에서 나온 예측 결과를 기준으로 계산했다."
    )

    st.subheader("모델별 성능")

    display_scores = score_df.copy()

    for col in [
        "정확도",
        "정밀도",
        "재현율",
        "F1 점수"
    ]:
        display_scores[col] = (
            display_scores[col] * 100
        ).round(1)

    st.dataframe(
        display_scores,
        use_container_width=True
    )

    st.subheader("모델별 정확도 비교")

    fig_accuracy = px.bar(
        display_scores,
        x="모델",
        y="정확도",
        text="정확도",
        title="모델별 정확도 비교",
        color="모델"
    )

    st.plotly_chart(
        fig_accuracy,
        use_container_width=True
    )

    st.subheader("모델별 F1 점수 비교")

    fig_f1 = px.bar(
        display_scores,
        x="모델",
        y="F1 점수",
        text="F1 점수",
        title="모델별 매크로 F1 점수 비교",
        color="모델"
    )

    st.plotly_chart(
        fig_f1,
        use_container_width=True
    )

    st.subheader("베이스라인보다 얼마나 나아졌을까?")

    baseline_row = score_df[
        score_df["모델"] == "베이스라인"
    ].iloc[0]

    model_rows = score_df[
        score_df["모델"] != "베이스라인"
    ]

    best_model_row = model_rows.sort_values(
        "F1 점수",
        ascending=False
    ).iloc[0]

    difference = (
        best_model_row["정확도"]
        - baseline_row["정확도"]
    ) * 100

    st.write(
        f"F1 점수가 가장 높은 모델은 **{best_model_row['모델']}**이다."
    )

    st.write(
        f"베이스라인 정확도는 {baseline_row['정확도']:.1%}이고, "
        f"가장 높은 F1 점수를 보인 모델의 정확도는 "
        f"{best_model_row['정확도']:.1%}이다."
    )

    st.write(
        f"정확도 차이는 **{difference:+.1f}%p**이다."
    )

    if difference > 0:
        st.success("해당 모델의 정확도가 베이스라인보다 높다.")

    elif difference < 0:
        st.warning("해당 모델의 정확도가 베이스라인보다 낮다.")

    else:
        st.info("해당 모델과 베이스라인의 정확도가 같다.")

    st.divider()

    st.subheader("혼동 행렬")

    selected_model = st.selectbox(
        "확인할 모델",
        [
            "로지스틱 회귀",
            "의사결정트리",
            "k-최근접 이웃"
        ]
    )

    cm = confusion_matrix(
        prediction_df["실제 결과"],
        prediction_df[selected_model + " 예측"],
        labels=RESULTS
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"실제 {x}" for x in RESULTS],
        columns=[f"예측 {x}" for x in RESULTS]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"{selected_model} 혼동 행렬"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    st.write("""
    혼동 행렬에서 행은 실제 결과이고 열은 모델의 예측 결과이다.
    대각선에 있는 숫자는 실제 결과를 맞힌 경기 수를 뜻한다.
    """)

    st.subheader("실제 결과와 모델 예측 비교")

    st.dataframe(
        prediction_df,
        use_container_width=True
    )

    st.download_button(
        "모델 예측 결과 다운로드",
        data=prediction_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="모델별_예측결과.csv",
        mime="text/csv"
    )

    st.warning("""
    경기 수가 많지 않기 때문에 결과를 일반화하기 어렵다.
    모델 점수가 높더라도 모든 경기에서 정확하게 예측한다고 볼 수 없다.
    """)


# ==========================================
# 15. 전반전 상황별 최종 결과
# ==========================================
elif menu == "전반전 상황별 최종 결과":

    st.header("⑥ 전반전 상황별 최종 결과")

    st.subheader("상황별 경기 수")

    st.dataframe(
        counts,
        use_container_width=True
    )

    st.subheader("상황별 최종 결과 비율")

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
        barmode="stack",
        title="전반전 상황별 최종 결과 비율"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("""
    전반전에 앞선 경기, 동점인 경기, 뒤진 경기에서
    최종 결과의 비율이 어떻게 다른지 비교할 수 있다.

    이 결과는 두 속성 사이의 관계를 보여 주며,
    전반전 상황이 최종 결과의 원인이라고 단정할 수는 없다.
    """)


# ==========================================
# 16. 모델 시각화 및 직접 예측
# ==========================================
elif menu == "모델 시각화 및 직접 예측":

    st.header("⑦ 모델 시각화 및 직접 예측")

    st.write(
        "전반전 득점을 직접 입력해 최종 결과를 예측하고, "
        "의사결정트리와 kNN이 데이터를 어떻게 판단하는지 살펴본다."
    )

    # --------------------------------------
    # 의사결정트리 시각화
    # --------------------------------------
    st.subheader("1. 의사결정트리 시각화")

    st.write("""
    한국 전반전 득점과 상대 팀 전반전 득점을 질문의 기준으로 삼아
    최종 결과인 승리·무승부·패배를 분류한다.

    아래 분기 숫자는 임의로 정한 값이 아니라
    실제 축구 경기 데이터로 학습한 트리가 찾아낸 기준이다.
    """)

    tree_model = fitted_models["의사결정트리"]

    fig_tree = draw_decision_tree_plotly(
        tree_model
    )

    st.plotly_chart(
        fig_tree,
        use_container_width=True
    )

    st.caption(
        "파란색 상자는 전반전 득점에 관한 판단 조건이고, "
        "초록색 상자는 마지막 예측 결과이다. "
        "파란색 '예' 선은 조건을 만족하는 경우, "
        "빨간색 '아니오' 선은 조건을 만족하지 않는 경우이다. "
        "분기 기준과 경기 수는 실제 학습 데이터에서 계산했다."
    )

    st.info("""
    의사결정트리는 전반전 득점만으로 최종 결과를 예측한다.
    상대 팀의 실력, 경기 장소, 선수 구성 등의 정보는 사용하지 않으므로
    실제 경기 결과를 확정적으로 예측하는 도구는 아니다.
    """)

    st.divider()

    # --------------------------------------
    # 새로운 경기 직접 예측
    # --------------------------------------
    st.subheader("2. 새로운 경기 결과 예측")

    st.write(
        "한국과 상대 팀의 전반전 득점을 입력해 보세요."
    )

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        korea_first = st.number_input(
            "대한민국 전반전 득점",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

    with input_col2:
        opponent_first = st.number_input(
            "상대 팀 전반전 득점",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

    if st.button(
        "두 모델로 경기 결과 예측하기",
        type="primary"
    ):

        new_match = pd.DataFrame(
            [[korea_first, opponent_first]],
            columns=FEATURES
        )

        tree_prediction = fitted_models[
            "의사결정트리"
        ].predict(new_match)[0]

        knn_prediction = fitted_models[
            "k-최근접 이웃"
        ].predict(new_match)[0]

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "의사결정트리 예측",
                tree_prediction
            )

        with result_col2:
            st.metric(
                f"kNN 예측 (K={k_value})",
                knn_prediction
            )

        if tree_prediction == knn_prediction:

            st.success(
                "두 모델이 같은 결과를 예측했습니다."
            )

        else:

            st.warning(
                "두 모델의 예측이 다릅니다. 모델마다 판단 방식이 다르기 때문이며, "
                "예측을 실제 경기 결과로 확정해서는 안 됩니다."
            )

        st.caption(
            "이 예측은 현재 데이터 전체로 학습한 모델의 결과입니다. "
            "실제 경기 결과를 보장하지 않으며, 입력한 경기는 "
            "학습 데이터와 비슷한 조건이라는 가정 아래 분류됩니다."
        )

        st.divider()

        # ----------------------------------
        # kNN이 참고한 가까운 경기
        # ----------------------------------
        st.subheader("3. kNN이 참고한 가까운 과거 경기")

        knn_pipeline = fitted_models["k-최근접 이웃"]

        scaler = knn_pipeline.named_steps["scaler"]

        query_scaled = scaler.transform(new_match)

        all_scaled = scaler.transform(
            clean_df[FEATURES]
        )

        distances = (
            (all_scaled - query_scaled) ** 2
        ).sum(axis=1) ** 0.5

        neighbor_df = clean_df.copy()

        neighbor_df["새 경기와의 거리"] = distances

        neighbor_df = neighbor_df.sort_values(
            "새 경기와의 거리"
        ).head(k_value)

        st.dataframe(
            neighbor_df[
                [
                    "날짜",
                    "상대팀"
                ] + FEATURES + [
                    "최종결과",
                    "새 경기와의 거리"
                ]
            ].round(3),
            use_container_width=True
        )

        vote_counts = (
            neighbor_df["최종결과"]
            .value_counts()
            .reindex(RESULTS, fill_value=0)
            .rename_axis("결과")
            .reset_index(name="가까운 경기 수")
        )

        st.write("가까운 경기들의 결과 분포")

        st.dataframe(
            vote_counts,
            use_container_width=True
        )

        # ----------------------------------
        # 가까운 경기 산점도
        # ----------------------------------
        fig_neighbors = px.scatter(
            clean_df,
            x="한국_전반득점",
            y="상대_전반득점",
            color="최종결과",
            hover_data=["날짜", "상대팀"],
            title=f"전반전 득점이 비슷한 과거 경기 (K={k_value})",
            labels={
                "한국_전반득점": "한국 전반전 득점",
                "상대_전반득점": "상대 팀 전반전 득점",
                "최종결과": "최종 결과"
            }
        )

        fig_neighbors.add_trace(
            go.Scatter(
                x=neighbor_df["한국_전반득점"],
                y=neighbor_df["상대_전반득점"],
                mode="markers",
                name="선택된 가까운 경기",
                marker=dict(
                    size=15,
                    symbol="circle-open",
                    line=dict(width=3)
                )
            )
        )

        fig_neighbors.add_trace(
            go.Scatter(
                x=[korea_first],
                y=[opponent_first],
                mode="markers",
                name="새 경기",
                marker=dict(
                    size=18,
                    symbol="star",
                    color="black"
                )
            )
        )

        st.plotly_chart(
            fig_neighbors,
            use_container_width=True
        )

    st.info(
        f"현재 kNN은 K={k_value}로 설정되어 있습니다. "
        "왼쪽 메뉴의 K값 슬라이더를 바꾸면 모델 평가와 예측에 반영됩니다."
    )


# ==========================================
# 17. 경기 기록
# ==========================================
elif menu == "경기 기록":

    st.header("⑧ 경기 기록")

    selected = st.selectbox(
        "전반전 상황 선택",
        ["전체"] + SITUATIONS
    )

    filtered = clean_df.copy()

    if selected != "전체":
        filtered = filtered[
            filtered["전반전상황"] == selected
        ]

    st.write(
        f"검색된 경기: {len(filtered)}경기"
    )

    st.dataframe(
        filtered[REQUIRED_COLUMNS],
        use_container_width=True
    )


# ==========================================
# 18. 최종 보고서
# ==========================================
elif menu == "최종 보고서":

    st.header("⑨ 최종 보고서")

    st.subheader("1. 탐구 주제")

    st.write("""
    대한민국 축구 국가대표팀의 전반전 경기 상황에 따라
    최종 경기 결과는 어떻게 달라질까?
    """)

    st.subheader("2. 분석 방법 선택 이유")

    st.write("""
    최종 결과는 승리, 무승부, 패배라는 세 가지 범주이므로
    회귀나 군집보다 분류가 적합하다고 판단했다.

    로지스틱 회귀, 의사결정트리, k-최근접 이웃 세 모델을 선정했다.
    각각 확률 계산, 조건에 따른 분리, 가까운 데이터의 다수결이라는
    서로 다른 원리로 작동한다.
    """)

    st.subheader("3. 데이터 전처리")

    st.write(f"""
    원본 데이터는 {len(original_df)}행 {original_df.shape[1]}열이다.
    분석에 필요한 결측치와 중복 행을 정리한 후
    {len(clean_df)}경기를 분석에 사용했다.

    이상치 후보는 확인했지만 실제 경기 기록일 수 있으므로
    임의로 삭제하지 않았다.
    """)

    st.subheader("4. 모델 학습 결과")

    report_scores = score_df.copy()

    for col in [
        "정확도",
        "정밀도",
        "재현율",
        "F1 점수"
    ]:
        report_scores[col] = (
            report_scores[col] * 100
        ).round(1)

    st.dataframe(
        report_scores,
        use_container_width=True
    )

    best_row = score_df[
        score_df["모델"] != "베이스라인"
    ].sort_values(
        "F1 점수",
        ascending=False
    ).iloc[0]

    st.write(
        f"매크로 F1 점수가 가장 높은 모델은 "
        f"**{best_row['모델']}**이다."
    )

    st.subheader("5. 의사결정트리와 kNN 추가 분석")

    st.write(f"""
    의사결정트리 그래프를 통해 전반전 득점에 따라
    데이터가 나뉘는 조건을 확인했다.

    kNN에서는 K={k_value}로 설정하고,
    새로운 경기와 가까운 과거 경기들의 결과를 비교해
    다수결로 예측하도록 했다.

    두 모델의 예측이 다를 수 있으므로
    한 가지 모델의 결과만으로 경기 결과를 단정하지 않았다.
    """)

    st.subheader("6. 결과를 믿기 어려운 이유")

    st.write("""
    분석 데이터의 경기 수가 많지 않고, 전반전 득점 두 가지 변수만
    사용했기 때문에 상대 팀의 실력, 경기 장소, 선수 구성,
    퇴장 여부 등 다른 요소를 충분히 반영하지 못했다.

    또한 데이터의 결과별 경기 수가 다를 수 있어
    모델 평가 결과를 신중하게 해석해야 한다.
    """)

    st.subheader("7. 새롭게 알게 된 점")

    st.write("""
    같은 데이터라도 모델마다 작동 원리가 다르고
    예측 결과와 성능이 달라질 수 있다는 것을 알게 되었다.

    또한 베이스라인과 비교해야 모델이 실제로 도움이 되는지
    판단할 수 있다는 점을 배웠다.
    """)

    st.subheader("8. 느낀 점")

    st.write("""
    이번 탐구를 통해 축구 경기의 흐름을 데이터로 분석하고,
    여러 분류 모델을 직접 비교해 볼 수 있었다.

    모델의 정확도만 보는 것이 아니라 정밀도, 재현율, F1 점수와
    혼동 행렬도 함께 살펴보는 것이 중요하다는 점을 알게 되었다.

    앞으로 더 많은 경기와 다양한 변수를 활용하면
    분석 결과를 더 깊이 이해할 수 있을 것이라고 생각한다.
    """)
