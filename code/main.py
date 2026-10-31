
import io
from urllib.parse import unquote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ==========================================
# 기본 설정
# ==========================================
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


# ==========================================
# 아시아 국가 목록
# ==========================================
ASIA_TEAMS = {
    "Australia", "Bahrain", "Bangladesh", "Bhutan",
    "Brunei", "Cambodia", "China", "Chinese Taipei",
    "Hong Kong", "India", "Indonesia", "Iran",
    "Iraq", "Japan", "Jordan", "Kuwait",
    "Kyrgyzstan", "Laos", "Lebanon", "Macau",
    "Malaysia", "Maldives", "Mongolia", "Myanmar",
    "Nepal", "North Korea", "Oman", "Pakistan",
    "Palestine", "Philippines", "Qatar", "Saudi Arabia",
    "Singapore", "South Korea", "Sri Lanka", "Syria",
    "Tajikistan", "Thailand", "Timor-Leste",
    "Turkmenistan", "UAE", "Uzbekistan", "Vietnam",
    "Yemen",
}


# ==========================================
# 데이터 불러오기
# ==========================================
@st.cache_data(ttl=300)
def load_data():
    errors = []

    for url in [DATA_URL, unquote(DATA_URL)]:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()

            for encoding in ["utf-8-sig", "utf-8", "cp949"]:
                try:
                    df = pd.read_csv(
                        io.BytesIO(response.content),
                        encoding=encoding,
                    )
                    df.columns = df.columns.astype(str).str.strip()
                    return df, None
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue

        except Exception as e:
            errors.append(str(e))

    return None, "\n".join(errors)


# ==========================================
# 전처리
# ==========================================
def preprocess_data(raw):
    df = raw.copy()

    required = [
        "날짜",
        "연도",
        "상대팀",
        "한국_전반득점",
        "상대_전반득점",
        "한국_최종득점",
        "상대_최종득점",
    ]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: " + ", ".join(missing_columns)
        )

    # 문자열 앞뒤 공백 제거
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()

    # 날짜 형식 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str),
        errors="coerce",
    )

    # 득점 열 숫자형 변환
    score_cols = [
        "한국_전반득점",
        "상대_전반득점",
        "한국_최종득점",
        "상대_최종득점",
    ]

    for col in score_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # 연도 숫자형 변환
    df["연도"] = pd.to_numeric(
        df["연도"],
        errors="coerce",
    )

    # 상대팀을 아시아 / 비아시아로 분류
    def classify_team(team):
        if pd.isna(team):
            return "분류 확인 필요"

        team = str(team).strip()

        if team in ASIA_TEAMS:
            return "아시아"
        elif team:
            return "비아시아"

        return "분류 확인 필요"

    df["상대지역"] = df["상대팀"].apply(classify_team)

    # 전반전 상황을 득점으로 다시 계산
    def first_half_result(row):
        korea = row["한국_전반득점"]
        opponent = row["상대_전반득점"]

        if pd.isna(korea) or pd.isna(opponent):
            return "확인 필요"

        if korea > opponent:
            return "앞섬"
        elif korea == opponent:
            return "동점"
        return "뒤짐"

    df["전반전상황_전처리"] = df.apply(
        first_half_result,
        axis=1,
    )

    # 최종 결과를 득점으로 다시 계산
    def final_result(row):
        korea = row["한국_최종득점"]
        opponent = row["상대_최종득점"]

        if pd.isna(korea) or pd.isna(opponent):
            return "확인 필요"

        if korea > opponent:
            return "승리"
        elif korea == opponent:
            return "무승부"
        return "패배"

    df["최종결과_전처리"] = df.apply(
        final_result,
        axis=1,
    )

    # 전반전 상황과 최종 결과가 원본 기록과 일치하는지 확인
    if "전반전상황" in df.columns:
        df["전반전기록_일치"] = (
            df["전반전상황"].astype(str)
            == df["전반전상황_전처리"]
        )

    if "최종결과" in df.columns:
        df["최종기록_일치"] = (
            df["최종결과"].astype(str)
            == df["최종결과_전처리"]
        )

    return df


# ==========================================
# 공통 함수
# ==========================================
def make_crosstab(df):
    return pd.crosstab(
        df["전반전상황_전처리"],
        df["최종결과_전처리"],
    ).reindex(
        index=["앞섬", "동점", "뒤짐"],
        columns=["승리", "무승부", "패배"],
        fill_value=0,
    )


def percentage_table(df):
    table = make_crosstab(df)

    return table.div(
        table.sum(axis=1).replace(0, 1),
        axis=0,
    ) * 100


def show_metrics(df):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("경기 수", f"{len(df):,}경기")
    c2.metric(
        "평균 한국 득점",
        f"{df['한국_최종득점'].mean():.2f}골",
    )
    c3.metric(
        "평균 상대 득점",
        f"{df['상대_최종득점'].mean():.2f}골",
    )

    wins = (df["최종결과_전처리"] == "승리").sum()
    win_rate = wins / len(df) * 100 if len(df) else 0

    c4.metric("한국 승리 비율", f"{win_rate:.1f}%")


# ==========================================
# 데이터 읽기
# ==========================================
raw, error = load_data()

st.title("⚽ 대한민국 축구 국가대표팀 경기 분석")
st.caption(
    "전반전 상황은 최종 경기 결과에 어떤 영향을 미칠까?"
)

if raw is None:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(error)
    st.stop()

try:
    df = preprocess_data(raw)
except Exception as e:
    st.error(f"데이터 전처리 중 오류가 발생했습니다: {e}")
    st.stop()


# ==========================================
# 분석에 사용할 데이터
# ==========================================
analysis_df = df.dropna(
    subset=[
        "한국_전반득점",
        "상대_전반득점",
        "한국_최종득점",
        "상대_최종득점",
        "상대지역",
    ]
).copy()

analysis_df = analysis_df[
    analysis_df["상대지역"].isin(["아시아", "비아시아"])
].copy()


# ==========================================
# 상단 요약
# ==========================================
st.markdown(
    """
    대한민국 남자 축구 국가대표팀의 경기 기록을 바탕으로
    **전반전 상황과 최종 경기 결과의 관계**를 분석합니다.

    특히 상대팀이 아시아 국가인지 비아시아 국가인지에 따라
    전반전 상황이 최종 결과로 이어지는 경향이 달라지는지 알아봅니다.
    """
)

show_metrics(analysis_df)

st.caption(
    f"원본 데이터 {len(raw)}경기 · 분석 가능 데이터 {len(analysis_df)}경기"
)

st.divider()


# ==========================================
# 탭 구성
# ==========================================
tabs = st.tabs([
    "① 탐구 질문",
    "② 데이터 보기",
    "③ 전처리",
    "④ 분석 결과",
    "⑤ 해석·한계",
])


# ==========================================
# 탭 1: 탐구 질문
# ==========================================
with tabs[0]:
    st.subheader("우리의 탐구 질문")

    st.markdown(
        """
        대한민국 축구 국가대표팀은 전반전에 앞서거나 뒤졌을 때
        최종 경기에서도 그 상황을 유지하는 경향이 있을까?

        또한 상대팀이 아시아 국가인지 비아시아 국가인지에 따라
        이러한 경향에 차이가 있을까?
        """
    )

    st.subheader("분석 목적")

    st.markdown(
        """
        1. 전반전 상황별 최종 경기 결과를 확인한다.
        2. 아시아와 비아시아 상대팀의 결과를 비교한다.
        3. 전반전에 뒤졌을 때 최종적으로 승리하거나 무승부를
           기록한 경기의 비율을 확인한다.
        4. 전반전 상황과 최종 결과 사이의 관계를 해석한다.
        """
    )

    st.subheader("분석 데이터")

    c1, c2, c3 = st.columns(3)

    c1.metric("수집한 경기", f"{len(raw)}경기")
    c2.metric(
        "아시아 상대 경기",
        f"{(analysis_df['상대지역'] == '아시아').sum()}경기",
    )
    c3.metric(
        "비아시아 상대 경기",
        f"{(analysis_df['상대지역'] == '비아시아').sum()}경기",
    )

    st.info(
        "이 분석은 경기 결과 사이의 관계를 확인하는 탐구입니다. "
        "전반전 상황이 최종 결과의 직접적인 원인이라고 단정하지 않습니다."
    )


# ==========================================
# 탭 2: 데이터 보기와 분포
# ==========================================
with tabs[1]:
    st.subheader("원본 데이터 미리보기")

    st.dataframe(
        raw,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "원본 데이터 다운로드",
        data=raw.to_csv(index=False).encode("utf-8-sig"),
        file_name="축구_원본데이터.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("득점 데이터의 분포")

    score_choice = st.selectbox(
        "분포를 확인할 항목",
        [
            "한국_전반득점",
            "상대_전반득점",
            "한국_최종득점",
            "상대_최종득점",
        ],
        format_func=lambda x: {
            "한국_전반득점": "한국 전반 득점",
            "상대_전반득점": "상대 전반 득점",
            "한국_최종득점": "한국 최종 득점",
            "상대_최종득점": "상대 최종 득점",
        }[x],
    )

    values = analysis_df[score_choice].dropna()

    fig = px.histogram(
        analysis_df,
        x=score_choice,
        nbins=8,
        title="득점별 경기 수 분포",
        labels={score_choice: "득점"},
    )

    fig.update_layout(
        xaxis_title="득점",
        yaxis_title="경기 수",
        bargap=0.1,
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("평균", f"{values.mean():.2f}골")
    c2.metric("중앙값", f"{values.median():.1f}골")
    c3.metric(
        "최댓값",
        f"{values.max():.0f}골" if len(values) else "-",
    )

    st.divider()

    st.subheader("전반전 상황과 최종 결과의 분포")

    situation_counts = (
        analysis_df["전반전상황_전처리"]
        .value_counts()
        .reindex(["앞섬", "동점", "뒤짐"], fill_value=0)
        .rename_axis("전반전 상황")
        .reset_index(name="경기 수")
    )

    fig2 = px.bar(
        situation_counts,
        x="전반전 상황",
        y="경기 수",
        text="경기 수",
        title="전반전 상황별 경기 수",
    )

    st.plotly_chart(fig2, use_container_width=True)

    result_counts = (
        analysis_df["최종결과_전처리"]
        .value_counts()
        .reindex(["승리", "무승부", "패배"], fill_value=0)
        .rename_axis("최종 결과")
        .reset_index(name="경기 수")
    )

    fig3 = px.bar(
        result_counts,
        x="최종 결과",
        y="경기 수",
        text="경기 수",
        title="최종 결과별 경기 수",
    )

    st.plotly_chart(fig3, use_container_width=True)


# ==========================================
# 탭 3: 전처리
# ==========================================
with tabs[2]:
    st.subheader("데이터 전처리")

    st.markdown(
        """
        원본 데이터를 분석에 적합한 형태로 정리했습니다.

        전처리의 목적은 데이터를 임의로 바꾸는 것이 아니라,
        분석에 필요한 형식으로 통일하고 새로운 분석용 열을 만드는 것입니다.
        """
    )

    st.subheader("전처리 방법")

    methods = pd.DataFrame([
        ["날짜", "날짜 형식으로 변환", "연도별 경기 분석"],
        ["득점", "숫자형으로 변환", "평균·중앙값 계산"],
        ["상대팀", "아시아·비아시아로 분류", "상대 지역별 비교"],
        ["전반전 상황", "득점 비교로 다시 계산", "상황 분류의 일관성 확보"],
        ["최종 결과", "최종 득점 비교로 다시 계산", "결과 분류의 일관성 확보"],
    ], columns=["대상", "전처리 방법", "전처리 목적"])

    st.dataframe(
        methods,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("원본 데이터와 전처리 데이터 비교")

    compare_cols = [
        "날짜",
        "상대팀",
        "한국_전반득점",
        "상대_전반득점",
        "한국_최종득점",
        "상대_최종득점",
    ]

    if "전반전상황" in df.columns:
        compare_cols.append("전반전상황")

    if "최종결과" in df.columns:
        compare_cols.append("최종결과")

    compare_cols += [
        "상대지역",
        "전반전상황_전처리",
        "최종결과_전처리",
    ]

    st.dataframe(
        df[compare_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "전처리 데이터 다운로드",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="축구_전처리데이터.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("전처리 결과 확인")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "원본 경기 수",
        f"{len(raw)}경기",
    )

    c2.metric(
        "분석 가능 경기 수",
        f"{len(analysis_df)}경기",
    )

    c3.metric(
        "분석에서 제외된 경기",
        f"{len(raw) - len(analysis_df)}경기",
    )

    if "전반전기록_일치" in df.columns:
        mismatch = int((~df["전반전기록_일치"]).sum())
        st.write(
            f"전반전 상황 원본 기록과 계산 결과가 다른 경기: "
            f"**{mismatch}경기**"
        )

    if "최종기록_일치" in df.columns:
        mismatch = int((~df["최종기록_일치"]).sum())
        st.write(
            f"최종 결과 원본 기록과 계산 결과가 다른 경기: "
            f"**{mismatch}경기**"
        )

    unknown = df[
        df["상대지역"] == "분류 확인 필요"
    ]

    if not unknown.empty:
        st.warning(
            "상대 지역을 확인할 수 없는 데이터가 있습니다. "
            "아래 팀을 확인한 뒤 국가 분류를 보완해야 합니다."
        )
        st.dataframe(
            unknown[["상대팀"]],
            hide_index=True,
        )

    st.info(
        "득점 정보가 없는 경기는 분석에서 제외됩니다. "
        "결측치가 없거나 제외된 경기가 없다면 실제 표시되는 수치를 기준으로 해석하세요."
    )


# ==========================================
# 탭 4: 분석 결과
# ==========================================
with tabs[3]:
    st.subheader("아시아 vs 비아시아 상대팀 비교")

    st.markdown(
        """
        전반전 상황별로 최종 결과가 어떻게 나타났는지 비교합니다.
        표의 비율은 같은 전반전 상황 안에서 계산한 비율입니다.
        """
    )

    region = st.radio(
        "상대팀 구분",
        ["전체", "아시아", "비아시아"],
        horizontal=True,
    )

    if region == "전체":
        selected = analysis_df.copy()
    else:
        selected = analysis_df[
            analysis_df["상대지역"] == region
        ].copy()

    st.metric("선택한 경기 수", f"{len(selected)}경기")

    counts = make_crosstab(selected)
    rates = percentage_table(selected)

    st.subheader("경기 수")

    st.dataframe(
        counts,
        use_container_width=True,
    )

    st.subheader("전반전 상황별 최종 결과 비율")

    rate_display = rates.round(1).astype(str) + "%"

    st.dataframe(
        rate_display,
        use_container_width=True,
    )

    plot_data = rates.reset_index().melt(
        id_vars="전반전상황_전처리",
        var_name="최종 결과",
        value_name="비율",
    )

    fig = px.bar(
        plot_data,
        x="전반전상황_전처리",
        y="비율",
        color="최종 결과",
        barmode="stack",
        text_auto=".1f",
        title="전반전 상황별 최종 결과 비율",
        labels={
            "전반전상황_전처리": "전반전 상황",
            "비율": "비율(%)",
        },
    )

    fig.update_layout(
        yaxis_title="비율(%)",
        xaxis_title="전반전 상황",
        yaxis_range=[0, 100],
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("아시아와 비아시아 직접 비교")

    comparison_rows = []

    for group in ["아시아", "비아시아"]:
        part = analysis_df[
            analysis_df["상대지역"] == group
        ]

        for situation in ["앞섬", "동점", "뒤짐"]:
            subset = part[
                part["전반전상황_전처리"] == situation
            ]

            n = len(subset)

            if n == 0:
                continue

            wins = (
                subset["최종결과_전처리"] == "승리"
            ).sum()

            draws = (
                subset["최종결과_전처리"] == "무승부"
            ).sum()

            losses = (
                subset["최종결과_전처리"] == "패배"
            ).sum()

            comparison_rows.append({
                "상대 지역": group,
                "전반전 상황": situation,
                "경기 수": n,
                "승리 비율(%)": round(wins / n * 100, 1),
                "무승부 비율(%)": round(draws / n * 100, 1),
                "패배 비율(%)": round(losses / n * 100, 1),
            })

    comparison = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "분석 결과 CSV 다운로드",
        data=comparison.to_csv(index=False).encode("utf-8-sig"),
        file_name="아시아_비아시아_비교.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("전반전 열세 이후 경기 결과")

    behind = analysis_df[
        analysis_df["전반전상황_전처리"] == "뒤짐"
    ]

    if len(behind) > 0:
        comeback = (
            behind["최종결과_전처리"] == "승리"
        ).sum()

        draw = (
            behind["최종결과_전처리"] == "무승부"
        ).sum()

        loss = (
            behind["최종결과_전처리"] == "패배"
        ).sum()

        c1, c2, c3 = st.columns(3)

        c1.metric("역전 승리", f"{comeback}경기")
        c2.metric("무승부", f"{draw}경기")
        c3.metric("패배", f"{loss}경기")

        st.write(
            f"전반전에 뒤진 {len(behind)}경기 중 "
            f"최종적으로 승리하거나 무승부를 기록한 경기는 "
            f"**{comeback + draw}경기**입니다."
        )
    else:
        st.info("전반전에 뒤진 경기가 없습니다.")


# ==========================================
# 탭 5: 해석과 한계
# ==========================================
with tabs[4]:
    st.subheader("분석 결과 해석")

    st.markdown(
        """
        다음 내용을 중심으로 분석 결과를 해석할 수 있습니다.

        - 전반전에 앞선 경기 중 최종 승리한 비율
        - 전반전에 동점이었던 경기의 최종 결과 분포
        - 전반전에 뒤진 경기 중 역전 승리하거나 무승부를 기록한 비율
        - 아시아와 비아시아 상대팀 사이의 결과 차이
        """
    )

    st.subheader("실제 데이터에 따른 결과")

    for group in ["아시아", "비아시아"]:
        part = analysis_df[
            analysis_df["상대지역"] == group
        ]

        st.markdown(f"#### {group} 상대 경기")

        if part.empty:
            st.info("분석할 경기가 없습니다.")
            continue

        for situation in ["앞섬", "동점", "뒤짐"]:
            subset = part[
                part["전반전상황_전처리"] == situation
            ]

            if subset.empty:
                st.write(
                    f"- 전반전 {situation}: 해당 경기 없음"
                )
                continue

            win_rate = (
                subset["최종결과_전처리"] == "승리"
            ).mean() * 100

            draw_rate = (
                subset["최종결과_전처리"] == "무승부"
            ).mean() * 100

            loss_rate = (
                subset["최종결과_전처리"] == "패배"
            ).mean() * 100

            st.write(
                f"- 전반전 {situation}: "
                f"{len(subset)}경기, "
                f"승리 {win_rate:.1f}%, "
                f"무승부 {draw_rate:.1f}%, "
                f"패배 {loss_rate:.1f}%"
            )

    st.divider()

    st.subheader("분석의 한계")

    st.markdown(
        """
        1. 분석 대상은 2021~2025년의 대한민국 남자 축구 국가대표팀 경기로 한정되어 있다.
        2. 경기 수가 많지 않기 때문에 결과를 모든 축구 경기에 일반화하기 어렵다.
        3. 상대팀의 실력, 경기 장소, 선수 구성, 대회 중요도 등 다른 요인은 반영하지 않았다.
        4. 아시아와 비아시아의 분류만으로 상대팀의 실제 경기력을 판단할 수는 없다.
        5. 두 변수 사이의 관계가 확인되더라도 인과관계라고 단정할 수 없다.
        """
    )

    st.info(
        "이 분석은 전반전 상황과 최종 결과 사이의 경향을 탐색하는 "
        "기초 데이터 분석입니다."
    )


# ==========================================
# 하단
# ==========================================
st.divider()

st.caption(
    "대한민국 남자 축구 국가대표팀 경기 기록을 활용한 데이터 분석 탐구"
)

st.caption(
    "데이터를 직접 수집하고 전처리하여 분석에 활용했습니다."
)
