

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

FEATURES = [
    "한국_전반득점",
    "상대_전반득점"
]

RESULTS = ["승리", "무승부", "패배"]
SITUATIONS = ["앞섬", "동점", "뒤짐"]

SCORE_COLUMNS = [
    "한국_전반득점",
    "상대_전반득점",
    "한국_최종득점",
    "상대_최종득점"
]

REQUIRED_COLUMNS = [
    "날짜",
    "연도",
    "상대팀",
    "대회",
    "한국_전반득점",
    "상대_전반득점",
    "한국_최종득점",
    "상대_최종득점",
    "전반전상황",
    "최종결과"
]


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    for url in [DATA_URL, unquote(DATA_URL)]:

        try:
            response = requests.get(
                url,
                timeout=20
            )

            response.raise_for_status()

            for encoding in [
                "utf-8-sig",
                "utf-8",
                "cp949"
            ]:

                try:
                    return pd.read_csv(
                        StringIO(
                            response.content.decode(encoding)
                        )
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
        "GitHub CSV 주소를 확인해 주세요."
    )

    st.stop()


missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in original_df.columns
]

if missing_columns:

    st.error(
        f"필요한 열이 없습니다: {missing_columns}"
    )

    st.stop()


# ==========================================
# 3. 데이터 전처리
# ==========================================

def preprocess_data(data):

    df = data.copy()

    # 숫자형으로 변환
    for col in SCORE_COLUMNS + ["연도"]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 문자열 공백 제거
    for col in [
        "상대팀",
        "대회",
        "전반전상황",
        "최종결과"
    ]:

        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )

        df[col] = df[col].replace(
            "",
            pd.NA
        )

    # 분석에 필요한 결측치 제거
    df = df.dropna(
        subset=FEATURES + [
            "최종결과",
            "전반전상황"
        ]
    )

    # 중복 제거
    df = df.drop_duplicates()

    # 분석 대상 결과만 사용
    df = df[
        df["최종결과"].isin(RESULTS)
        & df["전반전상황"].isin(SITUATIONS)
    ].copy()

    return df.reset_index(drop=True)


clean_df = preprocess_data(original_df)

if clean_df.empty:

    st.error(
        "전처리 후 분석할 데이터가 없습니다."
    )

    st.stop()


# ==========================================
# 4. 모델 생성
# ==========================================

def make_models(k_value=5):

    return {

        "로지스틱 회귀": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=3000)
            )
        ]),

        "의사결정트리": DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        ),

        "k-최근접 이웃": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=k_value
                )
            )
        ])

    }


# ==========================================
# 5. 모델 평가 함수
# ==========================================

@st.cache_data
def evaluate_models(data, k_value=5):

    X = data[FEATURES]
    y = data["최종결과"]

    counts = y.value_counts()

    # 5겹 교차검증이 가능한지 확인
    if len(counts) < 2 or counts.min() < 5:
        return None

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = make_models(k_value)

    # -----------------------------
    # 베이스라인 평가
    # -----------------------------

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

    scores = [

        {
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
        }

    ]

    # -----------------------------
    # 세 가지 모델 평가
    # -----------------------------

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

            "정확도": accuracy_score(
                y,
                pred
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
            )

        })

    score_df = pd.DataFrame(scores)

    # -----------------------------
    # 전체 데이터로 최종 모델 학습
    # -----------------------------

    fitted_models = make_models(k_value)

    for model in fitted_models.values():

        model.fit(X, y)

    # 교차검증 예측 결과 저장
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

        prediction_df[name + " 예측"] = (
            predictions[name]
        )

    return (
        score_df,
        prediction_df,
        fitted_models
    )


# ==========================================
# 6. 의사결정트리 Plotly 시각화
# ==========================================

def draw_decision_tree_plotly(model):

    tree = model.tree_

    class_names = [
        str(value)
        for value in model.classes_
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

        positions[node] = (
            x_pos,
            -depth
        )

        return x_pos

    assign_positions(0, 0)

    fig = go.Figure()

    # 노드 연결선
    for node in range(tree.node_count):

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        if left_child == right_child:
            continue

        x_parent, y_parent = positions[node]
        x_left, y_left = positions[left_child]
        x_right, y_right = positions[right_child]

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

        # 예 / 아니오 표시
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

    # 노드 설명 상자
    for node in range(tree.node_count):

        x_pos, y_pos = positions[node]

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        is_leaf = left_child == right_child

        class_counts = tree.value[node][0]

        predicted_index = int(
            class_counts.argmax()
        )

        predicted_class = class_names[
            predicted_index
        ]

        sample_count = int(
            tree.n_node_samples[node]
        )

        if is_leaf:

            label = (
                f"최종 예측: {predicted_class}<br>"
                f"해당 경기 수: {sample_count}"
            )

            background = "#DCFCE7"

        else:

            feature_index = int(
                tree.feature[node]
            )

            feature_label = feature_names.get(
                feature_index,
                str(feature_index)
            )

            threshold = tree.threshold[node]

            label = (
                f"{feature_label} ≤ {threshold:.1f}?<br>"
                f"해당 경기 수: {sample_count}<br>"
                f"주요 결과: {predicted_class}"
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
# 7. K값 설정 및 모델 학습
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
        "5겹 교차검증을 진행할 수 없습니다. "
        "각 결과가 최소 5경기 이상 필요합니다."
    )

    st.stop()

score_df, prediction_df, fitted_models = evaluation


# ==========================================
# 8. 공통 분석 통계
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
# 9. 제목 및 네 개의 탭
# ==========================================

st.title("⚽ 대한민국 축구 국가대표팀 경기 분석")

st.caption(
    "전반전 경기 상황을 통해 최종 경기 결과를 분석하고 "
    "분류 모델이 기준선보다 좋은 성능을 보이는지 확인한다."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "① 질문",
    "② 데이터 분석",
    "③ 기준선",
    "④ 모델 평가"
])


# ==========================================
# TAB 1. 질문
# ==========================================

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
    뒤지고 있어도 역전할 수 있는지 데이터를 통해 확인하고 싶었다.

    그래서 대한민국 축구 국가대표팀의 경기 기록을 활용하여
    전반전 상황과 최종 결과의 관계를 분석하기로 했다.
    """)

    st.divider()

    st.subheader("분석 목표")

    st.markdown("""
    1. 전반전 상황별 최종 결과의 비율을 비교한다.
    2. 전반전 득점을 이용해 최종 결과를 분류한다.
    3. 세 가지 분류 모델의 성능을 비교한다.
    4. 기준선보다 모델이 얼마나 나아졌는지 확인한다.
    """)

    st.divider()

    st.subheader("회귀·분류·군집 중 어떤 방법이 적합할까?")

    st.write("""
    **분류가 가장 적합하다.**

    이번 탐구에서 예측하고 싶은 값은 최종 득점처럼
    연속적인 숫자가 아니라 승리, 무승부, 패배라는
    세 가지 범주이다.

    따라서 범주를 예측하는 분류 모델을 사용한다.
    """)

    st.markdown("""
    - 회귀: 연속적인 숫자를 예측할 때 적합하다.
    - 분류: 정해진 범주를 예측할 때 적합하다.
    - 군집: 정답 없이 비슷한 데이터끼리 묶을 때 적합하다.
    """)

    st.divider()

    st.subheader("질문에 답할 수 있는 데이터인가?")

    st.write("""
    이번 데이터에는 다음 정보가 포함되어 있다.

    - 한국의 전반전 득점
    - 상대 팀의 전반전 득점
    - 한국의 최종 득점
    - 상대 팀의 최종 득점
    - 전반전 상황
    - 최종 경기 결과

    따라서 전반전 상황과 최종 결과의 관계를 분석하는 것은 가능하다.
    또한 전반전 득점을 입력하여 최종 결과를 분류하는 것도 가능하다.
    """)

    st.success(
        "분석에 필요한 전반전 득점과 최종 결과가 있으므로 "
        "탐구 질문에 데이터로 답할 수 있다."
    )

    st.warning("""
    단, 이 데이터는 경기 사진을 직접 읽는 자료가 아니다.
    사진 속 경기 장면을 분석하거나 사진에 관한 질문에 답하려면
    별도의 이미지 분석 기능과 이미지 데이터가 필요하다.
    """)


# ==========================================
# TAB 2. 데이터 분석
# ==========================================

with tab2:

    st.header("2. 데이터 분석")

    st.subheader("데이터 기본 정보")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "원본 행 수",
        len(original_df)
    )

    c2.metric(
        "열 수",
        original_df.shape[1]
    )

    c3.metric(
        "전처리 후 경기 수",
        len(clean_df)
    )

    st.divider()

    st.subheader("원본 데이터 확인")

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

    st.divider()

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
        f"전체 결측치 개수: "
        f"**{int(original_df.isna().sum().sum())}개**"
    )

    st.divider()

    st.subheader("데이터 전처리")

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

    st.divider()

    st.subheader("최종 결과별 경기 수")

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
        result_counts["경기 수"]
        / len(clean_df) * 100
    ).round(1)

    st.dataframe(
        result_counts,
        use_container_width=True
    )

    fig_result = px.bar(
        result_counts,
        x="최종 결과",
        y="경기 수",
        text="경기 수",
        title="최종 결과별 경기 수"
    )

    st.plotly_chart(
        fig_result,
        use_container_width=True
    )

    st.divider()

    st.subheader("전반전 상황별 최종 결과 분석")

    st.write(
        "전반전에 앞선 경기, 동점인 경기, 뒤진 경기에서 "
        "최종 결과가 어떻게 달라지는지 확인한다."
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
        value_name="비율"
    )

    fig_situation = px.bar(
        chart_df,
        x="전반전상황",
        y="비율",
        color="최종 결과",
        text=chart_df["비율"].round(1).astype(str) + "%",
        barmode="stack",
        title="전반전 상황별 최종 결과 비율"
    )

    st.plotly_chart(
        fig_situation,
        use_container_width=True
    )

    st.info("""
    이 그래프를 통해 전반전에 앞선 경기와 뒤진 경기의
    최종 결과 비율을 비교할 수 있다.

    두 변수 사이의 관계를 보여 주는 것이며,
    전반전 상황이 최종 결과의 직접적인 원인이라고 단정할 수는 없다.
    """)

    st.divider()

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
# TAB 3. 기준선
# ==========================================

with tab3:

    st.header("3. 기준선")

    st.subheader("기준선이란?")

    st.write("""
    기준선(Baseline)은 내가 만든 모델이 얼마나 잘 작동하는지
    판단하기 위해 사용하는 단순한 비교 기준이다.

    이번 탐구에서는 가장 많이 등장한 최종 결과만 계속 예측하는
    DummyClassifier를 기준선으로 사용한다.
    """)

    st.divider()

    st.subheader("기준선의 예측 원리")

    most_common = (
        clean_df["최종결과"]
        .value_counts()
        .idxmax()
    )

    most_common_count = int(
        clean_df["최종결과"].value_counts().max()
    )

    st.write(f"""
    현재 데이터에서 가장 많이 등장한 최종 결과는
    **{most_common}**이다.

    기준선 모델은 전반전 득점이 어떻든 관계없이
    모든 경기의 결과를 **{most_common}**이라고 예측한다.
    """)

    c1, c2 = st.columns(2)

    c1.metric(
        "가장 많은 결과",
        most_common
    )

    c2.metric(
        "해당 결과 경기 수",
        most_common_count
    )

    st.divider()

    st.subheader("최종 결과 분포")

    baseline_chart = px.bar(
        result_counts,
        x="최종 결과",
        y="경기 수",
        text="경기 수",
        color="최종 결과",
        title="기준선이 참고하는 실제 결과 분포"
    )

    st.plotly_chart(
        baseline_chart,
        use_container_width=True
    )

    st.divider()

    st.subheader("기준선 모델의 평가 결과")

    baseline_row = score_df[
        score_df["모델"] == "베이스라인"
    ].iloc[0]

    baseline_display = pd.DataFrame({

        "평가 지표": [
            "정확도",
            "정밀도",
            "재현율",
            "F1 점수"
        ],

        "점수(%)": [
            baseline_row["정확도"] * 100,
            baseline_row["정밀도"] * 100,
            baseline_row["재현율"] * 100,
            baseline_row["F1 점수"] * 100
        ]

    })

    baseline_display["점수(%)"] = (
        baseline_display["점수(%)"].round(1)
    )

    st.dataframe(
        baseline_display,
        use_container_width=True
    )

    st.metric(
        "기준선 정확도",
        f"{baseline_row['정확도']:.1%}"
    )

    st.write("""
    정확도는 전체 예측 중 실제 결과를 맞힌 비율이다.

    기준선은 가장 많이 등장하는 결과만 예측하므로
    다른 결과를 구분하지 못한다.

    따라서 내가 만든 분류 모델이 기준선보다 높은 성능을 보이는지
    확인하는 것이 중요하다.
    """)

    st.divider()

    st.subheader("기준선의 한계")

    st.markdown("""
    - 전반전 득점을 고려하지 않는다.
    - 가장 많이 등장한 결과만 예측한다.
    - 적은 수로 등장하는 결과는 제대로 구분하지 못한다.
    - 실제 경기의 다양한 상황을 반영하지 못한다.
    """)

    st.info(
        "기준선은 뛰어난 예측 모델이 아니라 "
        "내가 만든 모델의 성능을 비교하기 위한 출발점이다."
    )


# ==========================================
# TAB 4. 모델 평가
# ==========================================

with tab4:

    st.header("4. 모델 평가")

    st.subheader("사용한 분류 모델")

    st.write("""
    전반전 득점을 입력하여 최종 결과인 승리, 무승부, 패배를
    분류하는 세 가지 모델을 사용했다.
    """)

    with st.expander("모델 1. 로지스틱 회귀"):

        st.write("""
        입력 데이터를 바탕으로 각 결과가 나올 확률을 계산한다.

        한국과 상대 팀의 전반전 득점을 입력하면
        승리, 무승부, 패배일 확률을 계산하고
        가장 높은 확률의 결과를 선택한다.
        """)

    with st.expander("모델 2. 의사결정트리"):

        st.write("""
        여러 질문을 던지며 데이터를 나누는 모델이다.

        한국의 전반전 득점과 상대 팀의 전반전 득점을
        판단 기준으로 사용한다.

        각 질문에 따라 데이터를 나누고 마지막 단계에서
        승리, 무승부, 패배 중 하나를 예측한다.
        """)

    with st.expander("모델 3. k-최근접 이웃(kNN)"):

        st.write(f"""
        새로운 데이터와 가장 가까운 기존 데이터 K개를 찾고
        이웃들의 다수결로 결과를 분류한다.

        현재 K값은 **{k_value}**이다.

        거리 계산에서 큰 숫자의 영향이 지나치게 커지지 않도록
        StandardScaler로 득점 데이터를 표준화한다.
        """)

    st.divider()

    st.subheader("평가 방법")

    st.write("""
    세 모델과 기준선을 같은 데이터에 적용하고
    5겹 교차검증으로 평가했다.

    교차검증은 데이터를 여러 부분으로 나누어
    학습과 검증을 반복하는 방법이다.

    평가 지표는 정확도, 정밀도, 재현율, 매크로 F1 점수를 사용한다.
    """)

    st.divider()

    st.subheader("모델별 성능 비교")

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

    st.divider()

    st.subheader("모델별 정확도 비교")

    fig_accuracy = px.bar(
        display_scores,
        x="모델",
        y="정확도",
        text="정확도",
        color="모델",
        title="기준선과 분류 모델의 정확도 비교"
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
        color="모델",
        title="기준선과 분류 모델의 매크로 F1 점수 비교"
    )

    st.plotly_chart(
        fig_f1,
        use_container_width=True
    )

    st.divider()

    st.subheader("기준선보다 성능이 좋은 모델은?")

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
        f"매크로 F1 점수가 가장 높은 모델은 "
        f"**{best_model_row['모델']}**이다."
    )

    st.write(
        f"기준선 정확도는 {baseline_row['정확도']:.1%}이고, "
        f"가장 높은 F1 점수를 보인 모델의 정확도는 "
        f"{best_model_row['정확도']:.1%}이다."
    )

    st.metric(
        "정확도 차이",
        f"{difference:+.1f}%p"
    )

    if difference > 0:

        st.success(
            "F1 점수가 가장 높은 모델의 정확도가 "
            "기준선보다 높다."
        )

    elif difference < 0:

        st.warning(
            "F1 점수가 가장 높은 모델의 정확도가 "
            "기준선보다 낮다."
        )

    else:

        st.info(
            "해당 모델과 기준선의 정확도가 같다."
        )

    st.caption(
        "가장 높은 F1 점수를 보인 모델을 기준으로 비교했다. "
        "정확도와 F1 점수는 서로 다른 평가 지표이므로 "
        "두 값을 함께 확인해야 한다."
    )

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

    st.divider()

    st.subheader("의사결정트리 시각화")

    tree_model = fitted_models["의사결정트리"]

    fig_tree = draw_decision_tree_plotly(
        tree_model
    )

    st.plotly_chart(
        fig_tree,
        use_container_width=True
    )

    st.caption("""
    파란색 상자는 전반전 득점에 관한 판단 조건이고,
    초록색 상자는 마지막 예측 결과이다.

    분기 기준과 경기 수는 실제 학습 데이터에서 계산했다.
    """)

    st.divider()

    st.subheader("새로운 경기 결과 예측")

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
                "두 모델의 예측이 다릅니다. "
                "모델마다 판단 방식이 다르기 때문이며, "
                "예측을 실제 경기 결과로 확정해서는 안 됩니다."
            )

        st.caption(
            "이 예측은 현재 데이터 전체로 학습한 모델의 결과이다. "
            "실제 경기 결과를 보장하지 않는다."
        )

        st.divider()

        st.subheader("kNN이 참고한 가까운 과거 경기")

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

    st.divider()

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

    st.divider()

    st.subheader("모델의 한계와 분석 결과 해석")

    st.warning("""
    경기 수가 많지 않기 때문에 결과를 일반화하기 어렵다.

    전반전 득점 두 가지 변수만 사용했기 때문에
    상대 팀의 실력, 경기 장소, 선수 구성, 퇴장 여부 등의
    다양한 요소를 충분히 반영하지 못했다.

    모델 점수가 높더라도 모든 경기에서 정확하게 예측한다고
    볼 수 없다.
    """)

    st.subheader("최종적으로 알게 된 점")

    st.write("""
    같은 데이터라도 모델마다 작동 원리가 다르고
    예측 결과와 성능이 달라질 수 있다는 것을 알게 되었다.

    또한 기준선과 비교해야 모델이 실제로 도움이 되는지
    판단할 수 있다는 점을 배웠다.

    정확도만 보는 것이 아니라 정밀도, 재현율, F1 점수와
    혼동 행렬도 함께 살펴보는 것이 중요하다.
    """)
