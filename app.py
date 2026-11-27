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
# Data Loading & Caching
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
    df = pd.read_csv(path, usecols=cols, low_memory=False)

    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df["year"] = df["publish_time"].dt.year
    df["month"] = df["publish_time"].dt.to_period("M").dt.to_timestamp()

    df = df.dropna(subset=["title", "year"])
    df = df[(df["year"] >= 1990) & (df["year"] <= 2023)]

    df["source_x"] = df["source_x"].fillna("Unknown")
    df["journal"] = df["journal"].fillna("Unknown")
    df["has_abstract"] = df["abstract"].notna()
    df["abstract_words"] = df["abstract"].fillna("").str.split().str.len()

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

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider(
        "Publication Years",
        min_value=min_year,
        max_value=max_year,
        value=(2019, max_year),
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Papers", f"{n_papers:,}")
    col2.metric("📚 Journals", f"{n_journals:,}")
    col3.metric("📝 Have Abstract", f"{pct_abs:.0f}%")
    col4.metric("✍️ Avg Abstract Length", f"{avg_words:.0f} words")

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
        min(len(filtered), 20000), random_state=1
    )
    text = " ".join(sample_titles)
    stopwords = STOPWORDS.union(
        {"covid", "covid-19", "sars", "cov", "2", "study", "using"}
    )
    wc = WordCloud(
        width=1200,
        height=450,
        background_color="white",
        stopwords=stopwords,
    ).generate(text)

    fig_wc, ax = plt.subplots(figsize=(12, 4.5))
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
