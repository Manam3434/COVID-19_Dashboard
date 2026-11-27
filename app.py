import os
import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import STOPWORDS, WordCloud

# -----------------------------------------------------------------------------
# Default Kaggle Credentials (for Streamlit deployment)
# -----------------------------------------------------------------------------
DEFAULT_KAGGLE_USERNAME = "manamsingh"
DEFAULT_KAGGLE_KEY = "KGAT_ff6295c3eaaaf1208c9701d19e8bb2aa"

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CORD-19 Research Dashboard",
    page_icon="🦠",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Data Loading & Caching (Memory-Safe Chunked Loading)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading and loading CORD-19 metadata...")
def load_data() -> pd.DataFrame:
    username = (
        st.secrets.get("KAGGLE_USERNAME")
        if "KAGGLE_USERNAME" in st.secrets
        else os.environ.get("KAGGLE_USERNAME", DEFAULT_KAGGLE_USERNAME)
    )
    key = (
        st.secrets.get("KAGGLE_KEY")
        if "KAGGLE_KEY" in st.secrets
        else os.environ.get("KAGGLE_KEY", DEFAULT_KAGGLE_KEY)
    )

    if username:
        os.environ["KAGGLE_USERNAME"] = str(username)
    if key:
        os.environ["KAGGLE_KEY"] = str(key)

    path = kagglehub.dataset_download(
        "allen-institute-for-ai/CORD-19-research-challenge",
        path="metadata.csv",
    )

    cols = [
        "cord_uid",
        "title",
        "abstract",
        "publish_time",
        "authors",
        "journal",
        "source_x",
    ]

    chunks = []
    rows_collected = 0
    max_rows = 100_000

    for chunk in pd.read_csv(
        path, usecols=cols, chunksize=50_000, low_memory=False
    ):
        chunk["publish_time"] = pd.to_datetime(
            chunk["publish_time"], errors="coerce"
        )
        chunk["year"] = chunk["publish_time"].dt.year
        chunk = chunk.dropna(subset=["title", "year"])
        chunk = chunk[(chunk["year"] >= 2020) & (chunk["year"] <= 2023)]
        if chunk.empty:
            continue

        chunk["month"] = (
            chunk["publish_time"].dt.to_period("M").dt.to_timestamp()
        )
        chunk["source_x"] = chunk["source_x"].fillna("Unknown")
        chunk["journal"] = chunk["journal"].fillna("Unknown")
        chunk["has_abstract"] = chunk["abstract"].notna()
        chunk["abstract_words"] = (
            chunk["abstract"].fillna("").str.split().str.len()
        )

        # Drop full abstract and publish_time columns to reduce memory usage
        chunk = chunk.drop(columns=["abstract", "publish_time"])
        chunks.append(chunk)
        rows_collected += len(chunk)

        if rows_collected >= max_rows:
            break

    expected_cols = [
        "cord_uid",
        "title",
        "authors",
        "journal",
        "source_x",
        "year",
        "month",
        "has_abstract",
        "abstract_words",
    ]

    if not chunks:
        return pd.DataFrame(columns=expected_cols)

    df = pd.concat(chunks, ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    return df


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
def main():
    st.title("🦠 CORD-19 Research Dashboard")
    st.caption(
        "An interactive dashboard exploring the CORD-19 COVID-19 research papers dataset."
    )

    try:
        df = load_data()
    except Exception as e:
        st.error(
            f"Failed to load dataset: {e}\n\n"
            "Check your Kaggle username and API token configuration."
        )
        st.stop()

    if df.empty:
        st.warning("No papers loaded for the selected date range.")
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    default_start = max(min_year, 2020)
    year_range = st.sidebar.slider(
        "Publication Years",
        min_value=min_year,
        max_value=max_year,
        value=(default_start, max_year),
    )

    sources = ["All"] + sorted(df["source_x"].unique().tolist())
    selected_source = st.sidebar.selectbox("Source", options=sources, index=0)

    top_n = st.sidebar.slider("Top N (Journals / Authors / Sources)", 5, 25, 10)

    # --- Filter Data ---
    filtered = df[
        (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    ]
    if selected_source != "All":
        filtered = filtered[filtered["source_x"] == selected_source]

    if filtered.empty:
        st.warning("No papers match the selected filters.")
        st.stop()

    # --- KPI Metrics ---
    n_papers = len(filtered)
    n_journals = filtered.loc[
        filtered["journal"] != "Unknown", "journal"
    ].nunique()
    pct_abs = filtered["has_abstract"].mean() * 100
    avg_words = filtered.loc[filtered["has_abstract"], "abstract_words"].mean()
    avg_words_display = f"{avg_words:.0f} words" if pd.notna(avg_words) else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Papers", f"{n_papers:,}")
    col2.metric("📚 Journals", f"{n_journals:,}")
    col3.metric("📝 Have Abstract", f"{pct_abs:.0f}%")
    col4.metric("✍️ Avg Abstract Length", avg_words_display)

    st.divider()

    # --- 1. Papers Over Time ---
    monthly = filtered.groupby("month").size().reset_index(name="papers")
    fig_time = px.line(
        monthly,
        x="month",
        y="papers",
        title="Papers Published per Month",
        labels={"month": "Month", "papers": "Papers"},
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # --- 2 & 3. Top Journals & Top Authors ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        journals = (
            filtered[filtered["journal"] != "Unknown"]["journal"]
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        journals.columns = ["journal", "papers"]
        fig_journals = px.bar(
            journals.sort_values("papers"),
            x="papers",
            y="journal",
            orientation="h",
            title=f"Top {top_n} Journals",
            labels={"papers": "Papers", "journal": "Journal"},
        )
        st.plotly_chart(fig_journals, use_container_width=True)

    with chart_col2:
        authors = (
            filtered["authors"]
            .dropna()
            .str.split(";")
            .explode()
            .str.strip()
        )
        authors = authors[authors != ""].value_counts().head(top_n).reset_index()
        authors.columns = ["author", "papers"]
        fig_authors = px.bar(
            authors.sort_values("papers"),
            x="papers",
            y="author",
            orientation="h",
            title=f"Top {top_n} Authors",
            labels={"papers": "Papers", "author": "Author"},
        )
        st.plotly_chart(fig_authors, use_container_width=True)

    # --- 4 & 5. Papers by Source & Abstract Length ---
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        source_counts = (
            filtered["source_x"].value_counts().head(top_n).reset_index()
        )
        source_counts.columns = ["source", "papers"]
        fig_source = px.pie(
            source_counts,
            names="source",
            values="papers",
            title="Where Do the Papers Come From?",
        )
        st.plotly_chart(fig_source, use_container_width=True)

    with chart_col4:
        fig_hist = px.histogram(
            filtered[filtered["abstract_words"].between(1, 600)],
            x="abstract_words",
            nbins=40,
            title="How Long Are the Abstracts? (Words)",
            labels={"abstract_words": "Word Count"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- 6. Word Cloud of Titles ---
    st.subheader("Most Common Words in Titles")
    sample_titles = filtered["title"].sample(
        min(len(filtered), 2500), random_state=1
    )
    text = " ".join(sample_titles.astype(str))
    stopwords = STOPWORDS.union(
        {"covid", "covid-19", "sars", "cov", "2", "study", "using"}
    )
    wc = WordCloud(
        width=800,
        height=300,
        background_color="white",
        stopwords=stopwords,
    ).generate(text)

    fig_wc, ax = plt.subplots(figsize=(9, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)
    plt.close(fig_wc)

    # --- Guide ---
    with st.expander("ℹ️ What each chart shows"):
        st.markdown(
            """
            - **Papers per month** – How research output grew over time
            - **Top journals / authors** – Who publishes the most
            - **Source pie** – Which databases the papers came from
            - **Abstract length** – How detailed the abstracts are
            - **Word cloud** – The most common topics in titles
            """
        )


if __name__ == "__main__":
    main()
