
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import BytesIO
from urllib.parse import unquote

# ==========================================
# 페이지 설정
# ==========================================

st.set_page_config(
    page_title="축구 경기 상황 분석",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ 전반전 경기 상황과 최종 결과 분석")
st.caption(
    "대한민국 남자 축구 국가대표팀 경기 데이터 | 2021~2025"
)

DATA_URL = (
    "https://raw.githubusercontent.com/yejun4484-ux/GOAL-PROJECT/"
    "refs/heads/main/data/"
    "%25EA%25BF%2588%25EB%2591%2590%25EB%25A0%2588_"
    "%25EB%258D%25B0%25EC%259D%25B4%25ED%2584%25B0_"
    "%25EC%25B6%2595%25EA%25B5%25AC.csv"
)

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

SITUATIONS = ["앞섬", "동점", "뒤짐"]
RESULTS = ["승리", "무승부", "패배"]

# ==========================================
# 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    urls = [
        DATA_URL,
        unquote(DATA_URL)
    ]

    last_error = None

    for url in urls:

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
                    df = pd.read_csv(
                        BytesIO(response.content),
                        encoding=encoding
                    )

                    df.columns = df.columns.str.strip()

                    if all(
                        col in df.columns
                        for col in REQUIRED_COLUMNS
                    ):
                        return df

                except Exception as e:
                    last_error = e

        except Exception as e:
            last_error = e

    raise ValueError(
        f"데이터를 불러오지 못했습니다: {last_error}"
    )


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()

# ==========================================
# 데이터 전처리
# ==========================================

numeric_columns = [
    "연도",
    "한국_전반득점",
    "상대_전반득점",
    "한국_최종득점",
    "상대_최종득점"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df["전반전상황"] = (
    df["전반전상황"]
    .astype(str)
    .str.strip()
)

df["최종결과"] = (
    df["최종결과"]
    .astype(str)
    .str.strip()
)

analysis_df = df[
    df["전반전상황"].isin(SITUATIONS)
    & df["최종결과"].isin(RESULTS)
].copy()

# ==========================================
# 사이드바
# ==========================================

st.sidebar.title("⚽ 분석 메뉴")

menu = st.sidebar.radio(
    "페이지 선택",
    [
        "탐구 주제",
        "데이터 탐색",
        "전반전 상황별 경기 수",
        "전반전 상황별 최종 결과",
        "경기 기록 확인",
        "결과 해석"
    ]
)

# ==========================================
# 1. 탐구 주제
# ==========================================

if menu == "탐구 주제":

    st.header("1. 탐구 주제")

    st.subheader(
        "전반전 경기 상황에 따라 최종 경기 결과는 어떻게 달라질까?"
    )

    st.write("""
    축구 경기에서는 전반전이 끝났을 때의 경기 상황이
    최종 결과와 어떤 관계가 있는지 알아보고자 한다.

    대한민국 남자 축구 국가대표팀의 2021년부터 2025년까지
    경기 데이터를 활용하여 전반전 상황을 앞섬, 동점, 뒤짐으로
    구분하고 각 상황에서 최종적으로 승리, 무승부, 패배한
    비율을 비교한다.
    """)

    st.divider()

    st.subheader("분석 변수")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**전반전 상황**")
        st.markdown("""
        - 앞섬
        - 동점
        - 뒤짐
        """)

    with col2:
        st.markdown("**최종 경기 결과**")
        st.markdown("""
        - 승리
        - 무승부
        - 패배
        """)

    st.divider()

    st.metric(
        "분석 가능한 경기 수",
        f"{len(analysis_df)}경기"
    )

    st.info(
        "이번 탐구는 예측 모델을 만드는 것이 아니라 "
        "실제 경기 기록을 이용해 상황별 결과 비율을 비교하는 탐구이다."
    )

# ==========================================
# 2. 데이터 탐색
# ==========================================

elif menu == "데이터 탐색":

    st.header("2. 데이터 탐색")

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 경기 수", f"{len(df)}경기")
    col2.metric("전체 열 수", f"{len(df.columns)}개")
    col3.metric("분석 가능 경기 수", f"{len(analysis_df)}경기")

    st.subheader("데이터 미리보기")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("결측치 확인")

    missing = pd.DataFrame({
        "열 이름": df.columns,
        "결측치 수": df.isnull().sum().values
    })

    st.dataframe(
        missing,
        use_container_width=True
    )

    st.subheader("기초 통계")

    st.dataframe(
        df[numeric_columns].describe(),
        use_container_width=True
    )

    st.subheader("분석에서 제외된 데이터")

    st.write(
        f"전체 {len(df)}경기 중 "
        f"{len(df) - len(analysis_df)}경기는 "
        "전반전 상황 또는 최종 결과가 분석에 필요한 값과 "
        "일치하지 않아 분석에서 제외했어."
    )

# ==========================================
# 3. 전반전 상황별 경기 수
# ==========================================

elif menu == "전반전 상황별 경기 수":

    st.header("3. 전반전 상황별 경기 수")

    situation_counts = (
        analysis_df["전반전상황"]
        .value_counts()
        .reindex(SITUATIONS, fill_value=0)
        .reset_index()
    )

    situation_counts.columns = [
        "전반전 상황",
        "경기 수"
    ]

    col1, col2, col3 = st.columns(3)

    for col, situation in zip(
        [col1, col2, col3],
        SITUATIONS
    ):
        count = int(
            situation_counts.loc[
                situation_counts["전반전 상황"] == situation,
                "경기 수"
            ].iloc[0]
        )

        col.metric(
            situation,
            f"{count}경기"
        )

    st.subheader("전반전 상황별 경기 수 그래프")

    fig = px.bar(
        situation_counts,
        x="전반전 상황",
        y="경기 수",
        color="전반전 상황",
        text="경기 수",
        title="전반전 상황별 경기 수"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        situation_counts,
        use_container_width=True
    )

# ==========================================
# 4. 전반전 상황별 최종 결과
# ==========================================

elif menu == "전반전 상황별 최종 결과":

    st.header("4. 전반전 상황별 최종 결과")

    st.write("""
    이번 탐구에서 가장 중요한 분석이다.

    전반전 상황이 앞섬, 동점, 뒤짐인 경우를 각각 나누고
    각 상황에서 최종적으로 승리, 무승부, 패배한 비율을 계산한다.
    """)

    # 상황별 최종 결과 경기 수
    count_table = pd.crosstab(
        analysis_df["전반전상황"],
        analysis_df["최종결과"]
    ).reindex(
        index=SITUATIONS,
        columns=RESULTS,
        fill_value=0
    )

    # 상황별 비율
    ratio_table = (
        count_table
        .div(count_table.sum(axis=1).replace(0, 1), axis=0)
        * 100
    )

    st.subheader("상황별 경기 수")

    st.dataframe(
        count_table,
        use_container_width=True
    )

    st.subheader("상황별 최종 결과 비율")

    display_ratio = ratio_table.round(1).astype(str) + "%"

    st.dataframe(
        display_ratio,
        use_container_width=True
    )

    st.caption(
        "각 행의 비율은 해당 전반전 상황에 속한 경기 수를 "
        "기준으로 계산했어."
    )

    st.divider()

    st.subheader("최종 결과 비율 누적 막대그래프")

    chart_df = (
        ratio_table
        .reset_index()
        .melt(
            id_vars="전반전상황",
            var_name="최종 결과",
            value_name="비율"
        )
    )

    fig = px.bar(
        chart_df,
        x="전반전상황",
        y="비율",
        color="최종 결과",
        text=chart_df["비율"].map(
            lambda x: f"{x:.1f}%"
        ),
        title="전반전 상황별 최종 경기 결과 비율",
        barmode="stack",
        category_orders={
            "전반전상황": SITUATIONS,
            "최종 결과": RESULTS
        }
    )

    fig.update_layout(
        yaxis_title="비율 (%)",
        xaxis_title="전반전 상황"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("승리·무승부·패배별 비교")

    fig = px.bar(
        chart_df,
        x="전반전상황",
        y="비율",
        color="최종 결과",
        barmode="group",
        text=chart_df["비율"].map(
            lambda x: f"{x:.1f}%"
        ),
        title="전반전 상황별 최종 결과 비율 비교",
        category_orders={
            "전반전상황": SITUATIONS,
            "최종 결과": RESULTS
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("주요 결과 요약")

    for situation in SITUATIONS:

        total = int(
            count_table.loc[situation].sum()
        )

        if total == 0:
            st.write(
                f"**{situation}:** 분석할 경기가 없어."
            )
            continue

        best_result = (
            ratio_table.loc[situation]
            .idxmax()
        )

        best_ratio = (
            ratio_table.loc[situation, best_result]
        )

        st.write(
            f"- 전반전 **{situation}** 상황은 "
            f"총 {total}경기이며, "
            f"가장 많이 나타난 최종 결과는 "
            f"**{best_result}**({best_ratio:.1f}%)야."
        )

    # 다운로드
    download_table = count_table.join(
        ratio_table.add_suffix("_비율(%)")
    )

    st.download_button(
        "상황별 분석 결과 CSV 다운로드",
        download_table.to_csv(
            encoding="utf-8-sig"
        ),
        file_name="전반전_상황별_최종결과.csv",
        mime="text/csv"
    )

# ==========================================
# 5. 경기 기록 확인
# ==========================================

elif menu == "경기 기록 확인":

    st.header("5. 경기 기록 확인")

    selected_situation = st.selectbox(
        "전반전 상황 선택",
        ["전체"] + SITUATIONS
    )

    selected_result = st.selectbox(
        "최종 결과 선택",
        ["전체"] + RESULTS
    )

    filtered = analysis_df.copy()

    if selected_situation != "전체":
        filtered = filtered[
            filtered["전반전상황"] == selected_situation
        ]

    if selected_result != "전체":
        filtered = filtered[
            filtered["최종결과"] == selected_result
        ]

    st.metric(
        "조건에 맞는 경기 수",
        f"{len(filtered)}경기"
    )

    show_columns = [
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

    st.dataframe(
        filtered[show_columns],
        use_container_width=True
    )

    st.download_button(
        "경기 기록 CSV 다운로드",
        filtered[show_columns].to_csv(
            index=False,
            encoding="utf-8-sig"
        ),
        file_name="축구_경기기록.csv",
        mime="text/csv"
    )

# ==========================================
# 6. 결과 해석
# ==========================================

elif menu == "결과 해석":

    st.header("6. 결과 해석")

    st.subheader("탐구에서 확인할 내용")

    st.markdown("""
    1. 대한민국 대표팀이 전반전을 앞선 상태로 마쳤을 때
       최종 승리하는 비율은 얼마나 되는가?
    2. 전반전을 동점으로 마쳤을 때 최종 결과는 어떻게 나타나는가?
    3. 전반전을 뒤진 상태로 마쳤을 때 역전승하는 비율은 얼마나 되는가?
    4. 전반전 상황에 따라 최종 경기 결과의 분포가 달라지는가?
    """)

    st.divider()

    st.subheader("분석 시 주의할 점")

    st.warning("""
    - 분석 대상은 약 70경기의 국가대표팀 경기로 한정된다.
    - 경기 수가 적은 상황의 비율은 몇 경기만 달라져도 크게 변할 수 있다.
    - 전반전 상황과 최종 결과 사이의 관계를 분석하는 것이며,
      전반전 상황이 최종 결과의 원인이라고 단정할 수 없다.
    - 상대 팀의 전력, 경기 장소, 선수 구성 등 다른 요인은
      이번 분석에 반영하지 않았다.
    """)

    st.subheader("탐구 결론 작성")

    st.write("""
    이번 탐구에서는 대한민국 남자 축구 국가대표팀의 경기 기록을
    활용하여 전반전 상황에 따른 최종 경기 결과의 비율을 비교하였다.

    전반전 상황을 앞섬, 동점, 뒤짐으로 구분하고
    각 상황에서 승리, 무승부, 패배가 차지하는 비율을 계산하였다.

    이를 통해 전반전 경기 상황과 최종 경기 결과 사이에
    어떤 관계가 나타나는지 살펴보았다.
    """)

    st.info(
        "실제 분석 결과를 확인한 뒤 마지막 결론 문장을 "
        "구체적인 수치로 수정하면 더 좋은 탐구 보고서가 돼."
    )
