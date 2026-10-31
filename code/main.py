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
# 하단 안내
# ==========================================
st.divider()

st.caption(f"불러온 주소: {loaded_url}")

st.caption(
    "기초통계는 데이터의 구조를 이해하기 위한 출발점입니다. "
    "결측치와 자료형을 확인한 뒤 결과를 해석하세요."
)
