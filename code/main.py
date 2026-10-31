
import io
from urllib.parse import unquote

import pandas as pd
import requests
import streamlit as st


# ==========================================
# 기본 설정
# ==========================================
st.set_page_config(
    page_title="축구 데이터 기초통계 분석실",
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
# 아시아축구연맹(AFC) 소속 국가
# ==========================================
AFC_TEAMS = {
    "Australia",
    "Bahrain",
    "China",
    "Hong Kong",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Japan",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Malaysia",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi Arabia",
    "Singapore",
    "South Korea",
    "Syria",
    "Thailand",
    "Turkmenistan",
    "United Arab Emirates",
    "Uzbekistan",
    "Vietnam",
}


def classify_team(team):
    """상대팀을 아시아와 비아시아로 구분한다."""
    if pd.isna(team):
        return "분류 확인 필요"

    team = str(team).strip()

    if team in AFC_TEAMS:
        return "아시아 팀"

    return "비아시아 팀"


# ==========================================
# 데이터 불러오기
# ==========================================
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

                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue

        except Exception as exc:
            errors.append(f"{url}: {exc}")

    return None, None, "\n".join(errors)


# ==========================================
# 숫자형 데이터 자동 인식
# ==========================================
def prepare_numeric_columns(df):
    result = df.copy()

    for col in result.columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            continue

        name = str(col).lower()

        # 날짜 열은 숫자로 변환하지 않는다.
        if any(word in name for word in ("날짜", "date", "time", "일시")):
            continue

        original = result[col]

        cleaned = (
            original.astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        converted = pd.to_numeric(cleaned, errors="coerce")
        non_missing = original.notna().sum()

        if (
            non_missing > 0
            and converted.notna().sum() / non_missing >= 0.85
        ):
            result[col] = converted

    return result


# ==========================================
# 결측치 표 만들기
# ==========================================
def show_missing_table(df):
    missing = pd.DataFrame({
        "열 이름": df.columns,
        "결측치 개수": [
            int(df[col].isna().sum())
            for col in df.columns
        ],
        "결측치 비율(%)": [
            round(float(df[col].isna().mean() * 100), 2)
            for col in df.columns
        ],
        "자료형": [
            str(df[col].dtype)
            for col in df.columns
        ],
    })

    return missing.sort_values(
        ["결측치 개수", "열 이름"],
        ascending=[False, True],
    ).reset_index(drop=True)


# ==========================================
# 아시아 팀과 비아시아 팀 비교용 데이터
# ==========================================
def make_team_comparison_data(df):
    required_columns = [
        "상대팀",
        "한국_최종득점",
        "상대_최종득점",
        "최종결과",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        return None, missing_columns

    result = df.copy()

    result["한국_최종득점"] = pd.to_numeric(
        result["한국_최종득점"],
        errors="coerce",
    )

    result["상대_최종득점"] = pd.to_numeric(
        result["상대_최종득점"],
        errors="coerce",
    )

    result["대륙 구분"] = result["상대팀"].apply(classify_team)

    result["골득실"] = (
        result["한국_최종득점"]
        - result["상대_최종득점"]
    )

    result = result.dropna(
        subset=[
            "한국_최종득점",
            "상대_최종득점",
            "최종결과",
        ]
    )

    return result, []


# ==========================================
# 아시아 팀 vs 비아시아 팀 분석 화면
# ==========================================
def show_asia_comparison(df):
    st.subheader("🌏 아시아 팀 vs 비아시아 팀")

    st.write(
        "한국 남자 축구 국가대표팀이 아시아 팀과 경기했을 때와 "
        "비아시아 팀과 경기했을 때의 성적을 비교합니다."
    )

    comparison_df, missing_columns = make_team_comparison_data(df)

    if missing_columns:
        st.error(
            "비교 분석에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )
        return

    # 분류되지 않은 팀이 있는지 확인
    unknown_teams = comparison_df[
        comparison_df["대륙 구분"] == "분류 확인 필요"
    ]["상대팀"].dropna().unique().tolist()

    if unknown_teams:
        st.warning(
            "다음 팀은 아시아 여부를 확인할 수 없어 비교에서 제외합니다: "
            + ", ".join(unknown_teams)
        )

    comparison_df = comparison_df[
        comparison_df["대륙 구분"].isin(
            ["아시아 팀", "비아시아 팀"]
        )
    ].copy()

    if comparison_df.empty:
        st.warning("비교할 경기 데이터가 없습니다.")
        return

    # 연도 필터
    st.markdown("#### 분석 기간 선택")

    if "연도" in comparison_df.columns:
        comparison_df["연도"] = pd.to_numeric(
            comparison_df["연도"],
            errors="coerce",
        )

        years = sorted(
            comparison_df["연도"].dropna().astype(int).unique().tolist()
        )

        if years:
            selected_years = st.multiselect(
                "분석할 연도",
                options=years,
                default=years,
            )

            if selected_years:
                comparison_df = comparison_df[
                    comparison_df["연도"].isin(selected_years)
                ]
            else:
                st.info("분석할 연도를 하나 이상 선택해 주세요.")
                return

    if comparison_df.empty:
        st.warning("선택한 기간에 해당하는 경기가 없습니다.")
        return

    # 두 집단 데이터
    asia_df = comparison_df[
        comparison_df["대륙 구분"] == "아시아 팀"
    ]

    non_asia_df = comparison_df[
        comparison_df["대륙 구분"] == "비아시아 팀"
    ]

    if asia_df.empty or non_asia_df.empty:
        st.warning(
            "두 집단 중 한쪽의 경기 데이터가 없습니다. "
            "분석 기간을 다시 선택해 주세요."
        )
        return

    st.divider()

    # --------------------------------------
    # 1. 경기 수
    # --------------------------------------
    st.markdown("### 1. 집단별 경기 수")

    col1, col2 = st.columns(2)

    col1.metric(
        "아시아 팀과의 경기",
        f"{len(asia_df):,}경기",
    )

    col2.metric(
        "비아시아 팀과의 경기",
        f"{len(non_asia_df):,}경기",
    )

    count_df = pd.DataFrame({
        "경기 수": [
            len(asia_df),
            len(non_asia_df),
        ]
    }, index=["아시아 팀", "비아시아 팀"])

    st.bar_chart(count_df)

    st.caption(
        "집단별 경기 수가 다를 수 있으므로 전체 승리 횟수보다는 "
        "승리 비율과 경기당 평균 득점을 함께 비교하는 것이 좋습니다."
    )

    st.divider()

    # --------------------------------------
    # 2. 승리·무승부·패배
    # --------------------------------------
    st.markdown("### 2. 승리·무승부·패배 비교")

    result_order = ["승리", "무승부", "패배"]

    outcome_counts = pd.DataFrame({
        "아시아 팀": (
            asia_df["최종결과"]
            .value_counts()
            .reindex(result_order, fill_value=0)
        ),
        "비아시아 팀": (
            non_asia_df["최종결과"]
            .value_counts()
            .reindex(result_order, fill_value=0)
        ),
    })

    st.markdown("#### 결과별 경기 수")

    st.dataframe(
        outcome_counts,
        use_container_width=True,
    )

    outcome_rates = outcome_counts.copy()

    outcome_rates["아시아 팀"] = (
        outcome_rates["아시아 팀"] / len(asia_df) * 100
    )

    outcome_rates["비아시아 팀"] = (
        outcome_rates["비아시아 팀"] / len(non_asia_df) * 100
    )

    outcome_rates = outcome_rates.round(1)

    st.markdown("#### 결과별 비율(%)")

    st.dataframe(
        outcome_rates,
        use_container_width=True,
    )

    st.bar_chart(outcome_rates)

    st.caption(
        "승리율은 해당 집단과의 전체 경기 중 승리한 경기의 비율입니다."
    )

    st.divider()

    # --------------------------------------
    # 3. 평균 득점과 실점
    # --------------------------------------
    st.markdown("### 3. 경기당 평균 득점과 실점")

    score_summary = comparison_df.groupby("대륙 구분").agg(
        경기수=("상대팀", "count"),
        평균득점=("한국_최종득점", "mean"),
        평균실점=("상대_최종득점", "mean"),
        평균골득실=("골득실", "mean"),
    )

    score_summary = score_summary.reindex(
        ["아시아 팀", "비아시아 팀"]
    ).round(2)

    st.dataframe(
        score_summary,
        use_container_width=True,
    )

    score_chart = score_summary[
        ["평균득점", "평균실점"]
    ]

    st.markdown("#### 평균 득점과 실점 그래프")

    st.bar_chart(score_chart)

    st.markdown("#### 평균 골득실 그래프")

    st.bar_chart(
        score_summary[["평균골득실"]]
    )

    st.caption(
        "골득실은 한국의 최종 득점에서 상대팀의 최종 득점을 뺀 값입니다. "
        "양수이면 평균적으로 득점이 실점보다 많았다는 뜻입니다."
    )

    st.divider()

    # --------------------------------------
    # 4. 경기별 상세 데이터
    # --------------------------------------
    st.markdown("### 4. 경기별 결과 확인")

    show_columns = [
        col for col in [
            "날짜",
            "연도",
            "상대팀",
            "대륙 구분",
            "한국_최종득점",
            "상대_최종득점",
            "골득실",
            "최종결과",
        ]
        if col in comparison_df.columns
    ]

    detail_df = comparison_df[show_columns].copy()

    st.dataframe(
        detail_df.sort_values(
            "날짜",
            ascending=False,
        ) if "날짜" in detail_df.columns else detail_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "아시아·비아시아 비교 데이터 다운로드",
        data=comparison_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="아시아_비아시아_경기비교.csv",
        mime="text/csv",
    )

    st.divider()

    # --------------------------------------
    # 5. 분석 결과 해석 안내
    # --------------------------------------
    st.markdown("### 5. 분석할 때 주의할 점")

    st.info(
        """
        - 경기 수가 서로 다르면 전체 승리 횟수만으로 비교하면 안 됩니다.
        - 승리율과 경기당 평균 득점·실점을 함께 살펴보세요.
        - 상대팀의 실력, 대회 종류, 경기 장소 등 다른 요인도 결과에 영향을 줄 수 있습니다.
        - 이 분석만으로 아시아 팀과의 경기에서 성적이 달라지는 원인을 확정할 수는 없습니다.
        """
    )

    st.markdown("#### 탐구 질문")

    st.write(
        "한국 남자 축구 국가대표팀은 아시아 팀과 경기할 때와 "
        "비아시아 팀과 경기할 때 승리율과 경기당 평균 득점에 차이가 있을까?"
    )


# ==========================================
# 화면 제목
# ==========================================
st.title("⚽ 축구 데이터 기초통계 분석실")

st.caption(
    "축구 경기 데이터의 구조와 통계적 특징을 분석합니다."
)


# ==========================================
# 사이드바
# ==========================================
with st.sidebar:
    st.header("데이터 설정")
    st.write("데이터 출처: GitHub CSV")
    st.code(DATA_URL, language=None)

    if st.button("데이터 새로고침", use_container_width=True):
        load_data.clear()
        st.rerun()

    st.divider()

    st.markdown(
        """
        **분석 항목**
        - 행·열 개수
        - 결측치와 자료형
        - 평균과 중앙값
        - 범주별 빈도
        - 그래프와 상관관계
        - 아시아 팀과 비아시아 팀 비교
        """
    )


# ==========================================
# 데이터 읽기
# ==========================================
df_raw, loaded_url, error = load_data()

if df_raw is None:
    st.error("데이터를 불러오지 못했습니다.")

    st.write(
        "GitHub 저장소의 파일이 공개되어 있는지, "
        "CSV 파일 경로가 정확한지 확인해 주세요."
    )

    with st.expander("오류 내용 보기"):
        st.code(error or "원인을 확인할 수 없습니다.")

    st.info(
        "GitHub에서 CSV 파일을 열고 Raw 주소를 확인해 주세요."
    )

    st.stop()

df = prepare_numeric_columns(df_raw)


# ==========================================
# 데이터 기본 정보
# ==========================================
total_cells = df.shape[0] * df.shape[1]
missing_cells = int(df.isna().sum().sum())

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

category_cols = [
    col for col in df.columns
    if col not in numeric_cols
]


# ==========================================
# 핵심 지표 카드
# ==========================================
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("행 개수", f"{df.shape[0]:,}")
m2.metric("열 개수", f"{df.shape[1]:,}")
m3.metric("전체 결측치", f"{missing_cells:,}")
m4.metric("수치형 열", f"{len(numeric_cols):,}")
m5.metric("중복 행", f"{int(df.duplicated().sum()):,}")

if missing_cells == 0:
    st.success("현재 데이터에는 결측치가 없습니다.")
else:
    st.warning(
        f"전체 {total_cells:,}개 셀 중 "
        f"{missing_cells:,}개가 비어 있습니다."
    )


# ==========================================
# 분석 탭
# ==========================================
tabs = st.tabs([
    "📋 데이터 미리보기",
    "🧹 결측치·자료형",
    "🔢 수치형 기초통계",
    "🏷️ 범주형 빈도",
    "📈 그래프·관계",
    "🌏 아시아 vs 비아시아",
])


# ==========================================
# 탭 1: 데이터 미리보기
# ==========================================
with tabs[0]:
    st.subheader("데이터 미리보기")

    st.write(
        f"총 **{len(df):,}행 × {len(df.columns):,}열** 데이터입니다."
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=420,
    )

    st.download_button(
        "분석에 사용한 데이터 CSV 다운로드",
        data=df_raw.to_csv(index=False).encode("utf-8-sig"),
        file_name="축구_데이터.csv",
        mime="text/csv",
    )

    with st.expander("열 이름과 자료형"):
        schema = pd.DataFrame({
            "열 이름": df.columns,
            "자료형": [
                str(df[col].dtype)
                for col in df.columns
            ],
            "고유값 개수": [
                int(df[col].nunique(dropna=True))
                for col in df.columns
            ],
        })

        st.dataframe(
            schema,
            use_container_width=True,
            hide_index=True,
        )


# ==========================================
# 탭 2: 결측치와 자료형
# ==========================================
with tabs[1]:
    st.subheader("결측치 현황")

    missing_table = show_missing_table(df)

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True,
    )

    has_missing = missing_table[
        missing_table["결측치 개수"] > 0
    ]

    if has_missing.empty:
        st.success("모든 열에 결측치가 없습니다.")
    else:
        st.bar_chart(
            has_missing.set_index("열 이름")[["결측치 개수"]],
            use_container_width=True,
        )

    st.caption(
        "결측치 비율은 각 열의 전체 행 중 "
        "값이 비어 있는 행의 비율입니다."
    )


# ==========================================
# 탭 3: 수치형 기초통계
# ==========================================
with tabs[2]:
    st.subheader("수치형 열의 기초통계")

    if not numeric_cols:
        st.info("숫자로 인식된 열이 없습니다.")

    else:
        selected_numeric = st.multiselect(
            "통계를 볼 열 선택",
            options=numeric_cols,
            default=numeric_cols,
        )

        if selected_numeric:
            stats = df[selected_numeric].describe().T

            stats = stats.rename(columns={
                "count": "개수",
                "mean": "평균",
                "std": "표준편차",
                "min": "최솟값",
                "25%": "제1사분위수(25%)",
                "50%": "중앙값(50%)",
                "75%": "제3사분위수(75%)",
                "max": "최댓값",
            })

            stats.insert(
                0,
                "결측치 개수",
                [
                    int(df[col].isna().sum())
                    for col in stats.index
                ],
            )

            st.dataframe(
                stats.round(3),
                use_container_width=True,
            )

            st.markdown("#### 중앙값만 모아 보기")

            medians = (
                df[selected_numeric]
                .median(numeric_only=True)
                .rename("중앙값")
                .to_frame()
            )

            st.dataframe(
                medians.round(3),
                use_container_width=True,
            )

            st.download_button(
                "기초통계표 CSV 다운로드",
                data=stats.to_csv().encode("utf-8-sig"),
                file_name="축구_기초통계표.csv",
                mime="text/csv",
            )


# ==========================================
# 탭 4: 범주형 빈도 분석
# ==========================================
with tabs[3]:
    st.subheader("범주형 열의 빈도")

    if not category_cols:
        st.info("범주형 열이 없습니다.")

    else:
        selected_category = st.selectbox(
            "분석할 열 선택",
            category_cols,
        )

        top_n = st.slider(
            "표시할 항목 수",
            min_value=3,
            max_value=30,
            value=10,
        )

        counts = (
            df[selected_category]
            .fillna("(결측치)")
            .astype(str)
            .value_counts()
            .head(top_n)
            .rename_axis("항목")
            .reset_index(name="개수")
        )

        counts["비율(%)"] = (
            counts["개수"] / max(len(df), 1) * 100
        ).round(2)

        st.dataframe(
            counts,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            counts.set_index("항목")[["개수"]],
            use_container_width=True,
        )


# ==========================================
# 탭 5: 그래프와 상관관계
# ==========================================
with tabs[4]:
    st.subheader("수치형 열의 분포")

    if numeric_cols:
        chart_col = st.selectbox(
            "분포를 볼 수치형 열",
            numeric_cols,
        )

        values = df[chart_col].dropna()

        if values.empty:
            st.info("선택한 열에 분석할 숫자가 없습니다.")

        else:
            bin_count = min(20, max(1, values.nunique()))

            hist = (
                pd.cut(values, bins=bin_count)
                .value_counts()
                .sort_index()
            )

            hist_df = hist.rename_axis(
                "구간"
            ).reset_index(name="개수")

            hist_df["구간"] = hist_df["구간"].astype(str)

            st.bar_chart(
                hist_df.set_index("구간")[["개수"]],
                use_container_width=True,
            )

            st.write(
                f"**{chart_col}** — "
                f"평균 {values.mean():.2f}, "
                f"중앙값 {values.median():.2f}, "
                f"최솟값 {values.min():.2f}, "
                f"최댓값 {values.max():.2f}"
            )

        if len(numeric_cols) >= 2:
            st.subheader("수치형 열 사이의 상관관계")

            corr = df[numeric_cols].corr(
                numeric_only=True
            )

            st.dataframe(
                corr.round(3),
                use_container_width=True,
            )

            st.caption(
                "상관계수는 두 수치가 함께 변하는 경향을 나타냅니다. "
                "상관관계만으로 원인과 결과를 단정할 수는 없습니다."
            )

    else:
        st.info(
            "상관관계를 분석할 수 있는 수치형 열이 없습니다."
        )


# ==========================================
# 탭 6: 아시아 팀 vs 비아시아 팀
# ==========================================
with tabs[5]:
    show_asia_comparison(df)


# ==========================================
# 하단 안내
# ==========================================
st.divider()

st.caption(f"불러온 주소: {loaded_url}")

st.caption(
    "기초통계는 데이터의 구조를 이해하기 위한 출발점입니다. "
    "결측치와 자료형을 확인한 뒤 결과를 해석하세요."
)
