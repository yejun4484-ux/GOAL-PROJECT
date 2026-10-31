
import io
from urllib.parse import unquote

import pandas as pd
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
# 데이터 전처리
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
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    # 문자열 공백 정리
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce",
    )

    # 득점 숫자형 변환
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

    df["연도"] = pd.to_numeric(
        df["연도"],
        errors="coerce",
    )

    # 상대팀 지역 분류
    def classify_team(team):
        if pd.isna(team):
            return "분류 확인 필요"

        team = str(team).strip()

        if team in ASIA_TEAMS:
            return "아시아"

        if team:
            return "비아시아"

        return "분류 확인 필요"

    df["상대지역"] = df["상대팀"].apply(classify_team)

    # 전반전 상황 계산
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

    # 최종 결과 계산
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

    # 원본 기록과 계산 결과 비교
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
# 교차표 함수
# ==========================================
def make_crosstab(data):
    return pd.crosstab(
        data["전반전상황_전처리"],
        data["최종결과_전처리"],
    ).reindex(
        index=["앞섬", "동점", "뒤짐"],
        columns=["승리", "무승부", "패배"],
        fill_value=0,
    )


def percentage_table(data):
    table = make_crosstab(data)

    return table.div(
        table.sum(axis=1).replace(0, 1),
        axis=0,
    ) * 100


# ==========================================
# 상단 지표
# ==========================================
def show_metrics(data):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "경기 수",
        f"{len(data):,}경기",
    )

    c2.metric(
        "한국 평균 득점",
        f"{data['한국_최종득점'].mean():.2f}골",
    )

    c3.metric(
        "상대 평균 득점",
        f"{data['상대_최종득점'].mean():.2f}골",
    )

    wins = (
        data["최종결과_전처리"] == "승리"
    ).sum()

    win_rate = (
        wins / len(data) * 100
        if len(data) > 0 else 0
    )

    c4.metric(
        "한국 승리 비율",
        f"{win_rate:.1f}%",
    )


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

    st.write(
        "GitHub 저장소의 CSV 주소와 공개 설정을 확인해 주세요."
    )

    st.code(error or "오류 내용을 확인할 수 없습니다.")
    st.stop()

try:
    df = preprocess_data(raw)

except Exception as e:
    st.error(f"데이터 전처리 오류: {e}")
    st.stop()


# ==========================================
# 분석용 데이터
# ==========================================
analysis_df = df.dropna(
    subset=[
        "한국_전반득점",
        "상대_전반득점",
        "한국_최종득점",
        "상대_최종득점",
    ]
).copy()

analysis_df = analysis_df[
    analysis_df["상대지역"].isin(
        ["아시아", "비아시아"]
    )
].copy()


# ==========================================
# 탐구 소개
# ==========================================
st.markdown(
    """
    대한민국 남자 축구 국가대표팀의 경기 기록을 바탕으로
    **전반전 상황과 최종 경기 결과의 관계**를 분석합니다.

    상대팀이 아시아 국가인지 비아시아 국가인지에 따라서도
    결과에 차이가 나타나는지 살펴봅니다.
    """
)

show_metrics(analysis_df)

st.caption(
    f"원본 데이터 {len(raw)}경기 · "
    f"분석 가능 데이터 {len(analysis_df)}경기"
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
        대한민국 축구 국가대표팀은 전반전에 앞서거나
        뒤졌을 때 최종 경기에서도 그 상황을 유지하는 경향이 있을까?

        또한 상대팀이 아시아 국가인지 비아시아 국가인지에 따라
        이러한 경향에 차이가 있을까?
        """
    )

    st.subheader("분석 목적")

    st.markdown(
        """
        1. 전반전 상황별 최종 경기 결과를 확인한다.
        2. 아시아와 비아시아 상대팀의 결과를 비교한다.
        3. 전반전에 뒤졌을 때 최종적으로 승리하거나
           무승부를 기록한 경기의 비율을 확인한다.
        4. 전반전 상황과 최종 결과 사이의 관계를 해석한다.
        """
    )

    st.subheader("분석 데이터")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "수집한 경기",
        f"{len(raw)}경기",
    )

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
        "전반전 상황이 최종 결과의 직접적인 원인이라고 "
        "단정하지 않습니다."
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

    score_names = {
        "한국_전반득점": "한국 전반 득점",
        "상대_전반득점": "상대 전반 득점",
        "한국_최종득점": "한국 최종 득점",
        "상대_최종득점": "상대 최종 득점",
    }

    score_choice = st.selectbox(
        "분포를 확인할 항목",
        list(score_names.keys()),
        format_func=lambda x: score_names[x],
    )

    values = analysis_df[score_choice].dropna()

    if len(values) > 0:

        # 득점별 경기 수
        score_counts = (
            values.value_counts()
            .sort_index()
            .rename_axis("득점")
            .to_frame("경기 수")
        )

        st.bar_chart(
            score_counts,
            x_label="득점",
            y_label="경기 수",
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "평균",
            f"{values.mean():.2f}골",
        )

        c2.metric(
            "중앙값",
            f"{values.median():.1f}골",
        )

        c3.metric(
            "최댓값",
            f"{values.max():.0f}골",
        )

    else:
        st.info("분석할 득점 데이터가 없습니다.")

    st.divider()

    st.subheader("전반전 상황과 최종 결과의 분포")

    situation_counts = (
        analysis_df["전반전상황_전처리"]
        .value_counts()
        .reindex(
            ["앞섬", "동점", "뒤짐"],
            fill_value=0,
        )
        .rename_axis("전반전 상황")
        .to_frame("경기 수")
    )

    st.write("전반전 상황별 경기 수")

    st.bar_chart(
        situation_counts,
        x_label="전반전 상황",
        y_label="경기 수",
    )

    result_counts = (
        analysis_df["최종결과_전처리"]
        .value_counts()
        .reindex(
            ["승리", "무승부", "패배"],
            fill_value=0,
        )
        .rename_axis("최종 결과")
        .to_frame("경기 수")
    )

    st.write("최종 결과별 경기 수")

    st.bar_chart(
        result_counts,
        x_label="최종 결과",
        y_label="경기 수",
    )


# ==========================================
# 탭 3: 전처리
# ==========================================
with tabs[2]:

    st.subheader("데이터 전처리")

    st.markdown(
        """
        원본 데이터를 분석에 적합한 형태로 정리했습니다.

        데이터를 임의로 삭제하거나 바꾸는 것이 아니라,
        분석에 필요한 형식으로 통일하고 새로운 분석용 열을 만들었습니다.
        """
    )

    st.subheader("전처리 방법")

    methods = pd.DataFrame([
        ["날짜", "날짜 형식으로 변환", "연도별 경기 분석"],
        ["득점", "숫자형으로 변환", "평균·중앙값 계산"],
        ["상대팀", "아시아·비아시아로 분류", "상대 지역별 비교"],
        ["전반전 상황", "득점 비교로 다시 계산", "상황 분류의 일관성 확인"],
        ["최종 결과", "최종 득점 비교로 다시 계산", "결과 분류의 일관성 확인"],
    ], columns=[
        "대상",
        "전처리 방법",
        "전처리 목적",
    ])

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
        mismatch = int(
            (~df["전반전기록_일치"]).sum()
        )

        st.write(
            "전반전 상황 원본 기록과 계산 결과가 다른 경기: "
            f"**{mismatch}경기**"
        )

    if "최종기록_일치" in df.columns:
        mismatch = int(
            (~df["최종기록_일치"]).sum()
        )

        st.write(
            "최종 결과 원본 기록과 계산 결과가 다른 경기: "
            f"**{mismatch}경기**"
        )

    unknown = df[
        df["상대지역"] == "분류 확인 필요"
    ]

    if not unknown.empty:
        st.warning(
            "상대 지역을 확인할 수 없는 데이터가 있습니다. "
            "아래 팀의 분류를 확인해 주세요."
        )

        st.dataframe(
            unknown[["상대팀"]],
            hide_index=True,
        )

    st.info(
        "득점 정보가 없는 경기는 분석에서 제외됩니다. "
        "제외된 경기 수를 확인하고 결과를 해석하세요."
    )


# ==========================================
# 탭 4: 분석 결과
# ==========================================
with tabs[3]:

    st.subheader("아시아 vs 비아시아 상대팀 비교")

    st.markdown(
        """
        전반전 상황별로 최종 결과가 어떻게 나타났는지 비교합니다.
        비율은 같은 전반전 상황에 속한 경기들을 기준으로 계산합니다.
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

    st.metric(
        "선택한 경기 수",
        f"{len(selected)}경기",
    )

    counts = make_crosstab(selected)
    rates = percentage_table(selected)

    st.subheader("전반전 상황별 최종 결과 경기 수")

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

    st.write("최종 결과 비율 그래프")

    st.bar_chart(
        rates,
        x_label="전반전 상황",
        y_label="비율(%)",
    )

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
        1. 분석 대상은 2021~2025년 대한민국 남자 축구 국가대표팀 경기로 한정되어 있다.
        2. 경기 수가 많지 않아 모든 축구 경기에 일반화하기 어렵다.
        3. 상대팀의 실력, 경기 장소, 선수 구성, 대회 중요도 등은 반영하지 않았다.
        4. 아시아와 비아시아의 분류만으로 상대팀의 실제 경기력을 판단할 수는 없다.
        5. 두 변수 사이의 관계가 확인되더라도 인과관계라고 단정할 수 없다.
        """
    )

    st.info(
        "이 분석은 전반전 상황과 최종 결과 사이의 경향을 "
        "탐색하는 기초 데이터 분석입니다."
    )


# ==========================================
# 하단 안내
# ==========================================
st.divider()

st.caption(
    "대한민국 남자 축구 국가대표팀 경기 기록을 활용한 데이터 분석 탐구"
)

st.caption(
    "데이터를 직접 수집하고 전처리하여 분석에 활용했습니다."
)
