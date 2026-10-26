"""
CleanSpeak: YouTube Video Analytics and Audience Insights Application
Streamlit Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt

# Custom modules
from youtube_extractor import (
    extract_video_id,
    fetch_video_details_api,
    fetch_video_comments_api,
    get_sample_video_and_comments
)
from sentiment_analyzer import (
    analyze_comments_sentiment,
    calculate_sentiment_summary,
    calculate_engagement_metrics
)
from text_emoji_analyzer import (
    generate_wordcloud_figure,
    get_top_keywords,
    analyze_emojis
)
from ml_predictor import (
    train_rf_models,
    predict_engagement,
    FEATURE_LABELS
)
from sample_data import SAMPLE_VIDEOS

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="CleanSpeak - YouTube Analytics & Audience Insights",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header Gradient Banner */
    .banner-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #312e81 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    .banner-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .banner-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 0px;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #3b82f6;
        font-weight: 500;
    }

    /* Pill Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-positive { background-color: #dcfce7; color: #166534; }
    .badge-neutral { background-color: #f1f5f9; color: #475569; }
    .badge-negative { background-color: #fee2e2; color: #991b1b; }
    .badge-tag { background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }

    /* Comment Cards */
    .comment-box {
        background: white;
        border-left: 4px solid #cbd5e1;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .comment-box-positive { border-left-color: #10b981; }
    .comment-box-neutral { border-left-color: #64748b; }
    .comment-box-negative { border-left-color: #ef4444; }

    .comment-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 6px;
    }
    .comment-author {
        font-weight: 700;
        color: #1e293b;
    }
    .comment-body {
        font-size: 0.95rem;
        color: #334155;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached ML Model Training
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Training Random Forest models on benchmark data...")
def get_trained_models():
    """Trains and caches the Random Forest models for fast interactive use."""
    return train_rf_models()


# Load models
rf_likes, rf_comments, ml_metrics, feature_importance_df = get_trained_models()


# ---------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/youtube-play.png", width=64)
    st.title("CleanSpeak")
    st.caption("YouTube Video Analytics & Audience Insights")
    st.markdown("---")

    data_source_mode = st.radio(
        "Select Data Source:",
        ["🧪 Demo Sample Mode", "🚀 Live YouTube API"],
        index=0,
        help="Use Demo mode for instant evaluation without an API key, or Live API to analyze any public YouTube video."
    )

    if data_source_mode == "🚀 Live YouTube API":
        api_key_input = st.text_input(
            "YouTube Data API v3 Key",
            type="password",
            help="Enter your Google Cloud YouTube Data API v3 key."
        )
        with st.expander("🔑 How to get a free API Key"):
            st.markdown("""
            1. Go to [Google Cloud Console](https://console.cloud.google.com/).
            2. Create a free project & enable **YouTube Data API v3**.
            3. Go to **Credentials** > **Create Credentials** > **API Key**.
            4. Paste your key above. Free quota allows 10,000 units/day.
            """)

        video_url_input = st.text_input(
            "YouTube Video Link or ID:",
            value="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            placeholder="e.g. https://www.youtube.com/watch?v=..."
        )
        max_comments_to_fetch = st.slider(
            "Max Comments to Collect:",
            min_value=20,
            max_value=300,
            value=100,
            step=20,
            help="Higher counts retrieve more comments but consume more API quota."
        )
        selected_sample_key = None

    else:
        # Demo mode
        api_key_input = ""
        st.info("💡 **Instant Demo Mode**: Pre-loaded with verified video metrics and audience comments with emojis.")
        sample_options = {
            "tech_ai_crash_course": "💻 Full-Stack AI Apps Crash Course (Tech)",
            "tech_gadget_review": "📱 Smartphone 16 Ultra Review (Gadget)",
            "fitness_workout_guide": "🏃 20-Minute HIIT Bodyweight Workout (Fitness)"
        }
        selected_sample_key = st.selectbox(
            "Choose a Sample Video:",
            options=list(sample_options.keys()),
            format_func=lambda x: sample_options[x]
        )
        video_url_input = SAMPLE_VIDEOS[selected_sample_key]["id"]
        max_comments_to_fetch = 100

    analyze_button = st.button("🔍 Analyze Video", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Core Capabilities:**
    - 📊 Performance & Engagement KPIs
    - 💬 TextBlob Sentiment Analysis
    - ☁️ Audience Word Cloud & Keywords
    - 😊 Emoji Sentiment Analytics
    - 🤖 Random Forest Engagement Predictor
    """)


# ---------------------------------------------------------
# Session State Initialization & Data Fetching
# ---------------------------------------------------------
if "video_data" not in st.session_state:
    st.session_state.video_data = None
if "comments_df" not in st.session_state:
    st.session_state.comments_df = None
if "summary_metrics" not in st.session_state:
    st.session_state.summary_metrics = None
if "engagement_metrics" not in st.session_state:
    st.session_state.engagement_metrics = None


def run_analysis(source_mode, api_key, url_or_id, sample_key, max_comm):
    with st.spinner("Collecting video data and analyzing audience comments..."):
        try:
            if source_mode == "🧪 Demo Sample Mode":
                v_data, c_list = get_sample_video_and_comments(sample_key)
            else:
                if not api_key:
                    st.error("⚠️ Please provide a valid YouTube Data API key or switch to Demo Mode.")
                    return
                extracted_id = extract_video_id(url_or_id)
                if not extracted_id:
                    st.error("⚠️ Invalid YouTube URL or Video ID. Please check the input.")
                    return
                v_data = fetch_video_details_api(api_key, extracted_id)
                c_list = fetch_video_comments_api(api_key, extracted_id, max_results=max_comm)

            # Process sentiment
            df_comments = analyze_comments_sentiment(c_list)
            summary = calculate_sentiment_summary(df_comments)
            eng = calculate_engagement_metrics(v_data["views"], v_data["likes"], v_data["comment_count"])

            st.session_state.video_data = v_data
            st.session_state.comments_df = df_comments
            st.session_state.summary_metrics = summary
            st.session_state.engagement_metrics = eng
            st.toast("Analysis completed successfully!", icon="✅")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")


# Trigger initial load if nothing is loaded yet
if st.session_state.video_data is None:
    run_analysis(data_source_mode, api_key_input, video_url_input, selected_sample_key or "tech_ai_crash_course", max_comments_to_fetch)
elif analyze_button:
    run_analysis(data_source_mode, api_key_input, video_url_input, selected_sample_key or "tech_ai_crash_course", max_comments_to_fetch)


# ---------------------------------------------------------
# Main UI Dashboard
# ---------------------------------------------------------
video_data = st.session_state.video_data
df_comments = st.session_state.comments_df
summary = st.session_state.summary_metrics
eng = st.session_state.engagement_metrics

if video_data and df_comments is not None:
    # Header Banner
    st.markdown(f"""
    <div class="banner-container">
        <div class="banner-title">CleanSpeak: YouTube Analytics & Audience Insights</div>
        <div class="banner-subtitle">
            Analyzing <strong>{video_data['title']}</strong> by <em>{video_data['channel_title']}</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Views</div>
            <div class="metric-value">{video_data['views']:,}</div>
            <div class="metric-sub">⏱️ Duration: {video_data['duration']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Likes</div>
            <div class="metric-value">{video_data['likes']:,}</div>
            <div class="metric-sub">👍 {eng['like_ratio']}% Like-to-View</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Comments</div>
            <div class="metric-value">{video_data['comment_count']:,}</div>
            <div class="metric-sub">💬 {eng['comment_ratio']}% Comment-to-View</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Engagement Rate</div>
            <div class="metric-value" style="color: {eng['health_color']};">{eng['engagement_rate']}%</div>
            <div class="metric-sub">{eng['health_status']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        dom_color = "#10b981" if summary['dominant_sentiment'] == "Positive" else ("#ef4444" if summary['dominant_sentiment'] == "Negative" else "#64748b")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Audience Sentiment</div>
            <div class="metric-value" style="color: {dom_color};">{summary['dominant_sentiment']}</div>
            <div class="metric-sub">Score: {summary['avg_polarity']:+.2f} ({summary['positive_pct']}% Pos)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs Interface
    tab_overview, tab_sentiment, tab_wordcloud, tab_emoji, tab_ml, tab_data = st.tabs([
        "📊 Video Overview",
        "💬 Sentiment Analysis",
        "☁️ Word Cloud & Keywords",
        "😊 Emoji Analysis",
        "🤖 ML Engagement Predictor",
        "📥 Data Explorer & Export"
    ])

    # -----------------------------------------------------
    # TAB 1: Video Overview
    # -----------------------------------------------------
    with tab_overview:
        st.subheader("Video Details & Performance Overview")
        col_vid_left, col_vid_right = st.columns([1, 1])

        with col_vid_left:
            if video_data.get("thumbnail_url"):
                st.image(video_data["thumbnail_url"], use_container_width=True, caption=video_data["title"])
            else:
                st.video(f"https://www.youtube.com/watch?v={video_data['id']}")

        with col_vid_right:
            st.markdown(f"### {video_data['title']}")
            st.markdown(f"**Channel:** `{video_data['channel_title']}`")
            st.markdown(f"**Published Date:** `{video_data['published_at'][:10] if video_data.get('published_at') else 'N/A'}`")
            st.markdown(f"**Category:** `{video_data.get('category', 'Science & Technology')}`")
            st.markdown(f"**Duration:** `{video_data['duration']}` ({video_data['duration_seconds']} seconds)")

            # Tags display
            tags = video_data.get("tags", [])
            if tags:
                st.markdown("**Tags:**")
                tags_html = " ".join([f"<span class='badge badge-tag'>#{t}</span>" for t in tags[:12]])
                st.markdown(tags_html, unsafe_allow_html=True)

            with st.expander("📄 View Video Description"):
                st.write(video_data.get("description", "No description provided."))

        st.markdown("---")
        st.subheader("📈 Performance Benchmarking")
        bench_col1, bench_col2 = st.columns(2)

        with bench_col1:
            # Benchmark gauge / comparison
            gauge_max = max(10.0, float(eng["engagement_rate"]) * 1.2)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=eng["engagement_rate"],
                title={"text": "Overall Engagement Rate (%)", "font": {"size": 20}},
                delta={"reference": 3.5, "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
                gauge={
                    "axis": {"range": [0, gauge_max], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": "#3b82f6"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, gauge_max * 0.2], "color": "#fee2e2"},
                        {"range": [gauge_max * 0.2, gauge_max * 0.4], "color": "#fef3c7"},
                        {"range": [gauge_max * 0.4, gauge_max * 0.7], "color": "#e0e7ff"},
                        {"range": [gauge_max * 0.7, gauge_max], "color": "#dcfce7"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 3.5
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with bench_col2:
            # Comparison Bar Chart: Video vs Industry Averages
            categories = ["Like Rate (%)", "Comment Rate (%)", "Total Engagement (%)"]
            video_vals = [eng["like_ratio"], eng["comment_ratio"], eng["engagement_rate"]]
            industry_vals = [3.2, 0.35, 3.55]

            fig_comp = go.Figure(data=[
                go.Bar(name="This Video", x=categories, y=video_vals, marker_color="#3b82f6"),
                go.Bar(name="YouTube Benchmark Avg", x=categories, y=industry_vals, marker_color="#94a3b8")
            ])
            fig_comp.update_layout(
                barmode="group",
                title="Video Performance vs. YouTube Industry Benchmarks",
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # -----------------------------------------------------
    # TAB 2: Sentiment Analysis
    # -----------------------------------------------------
    with tab_sentiment:
        st.subheader("Audience Sentiment Analysis (TextBlob)")
        st.write(
            "CleanSpeak uses **TextBlob** natural language processing to evaluate the polarity "
            "(-1.0 = highly negative, +1.0 = highly positive) and subjectivity (0.0 = objective, 1.0 = opinionated) "
            "of collected viewer comments."
        )

        sent_stat1, sent_stat2, sent_stat3, sent_stat4 = st.columns(4)
        sent_stat1.metric("Analyzed Comments", f"{summary['total_comments']}")
        sent_stat2.metric("Positive Sentiment", f"{summary['positive_pct']}%", f"{summary['positive_count']} comments", delta_color="normal")
        sent_stat3.metric("Neutral Sentiment", f"{summary['neutral_pct']}%", f"{summary['neutral_count']} comments", delta_color="off")
        sent_stat4.metric("Negative Sentiment", f"{summary['negative_pct']}%", f"{summary['negative_count']} comments", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            # Donut chart of sentiment proportions
            sentiment_labels = ["Positive", "Neutral", "Negative"]
            sentiment_counts = [summary["positive_count"], summary["neutral_count"], summary["negative_count"]]
            colors = ["#10b981", "#64748b", "#ef4444"]

            fig_donut = go.Figure(data=[go.Pie(
                labels=sentiment_labels,
                values=sentiment_counts,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hoverinfo="label+value+percent"
            )])
            fig_donut.update_layout(
                title="Audience Sentiment Distribution",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with chart_col2:
            # Polarity distribution histogram
            fig_hist = px.histogram(
                df_comments,
                x="polarity",
                nbins=25,
                color="sentiment",
                color_discrete_map={"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#ef4444"},
                title="Sentiment Polarity Score Distribution (-1.0 to +1.0)",
                labels={"polarity": "TextBlob Polarity Score", "count": "Comment Count"}
            )
            fig_hist.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), bargap=0.08)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Interactive Comments Explorer")

        sentiment_filter = st.radio(
            "Filter Comments by Category:",
            ["All", "Positive 😊", "Neutral 😐", "Negative 😠"],
            horizontal=True
        )

        filter_mapping = {
            "All": None,
            "Positive 😊": "Positive",
            "Neutral 😐": "Neutral",
            "Negative 😠": "Negative"
        }
        selected_category = filter_mapping[sentiment_filter]

        search_query = st.text_input("🔎 Search comment text or author:", placeholder="Type keywords here...")

        filtered_comments = df_comments.copy()
        if selected_category:
            filtered_comments = filtered_comments[filtered_comments["sentiment"] == selected_category]
        if search_query:
            filtered_comments = filtered_comments[
                filtered_comments["original_text"].str.contains(search_query, case=False, na=False) |
                filtered_comments["author"].str.contains(search_query, case=False, na=False)
            ]

        st.caption(f"Showing {len(filtered_comments)} matching comments:")

        # Render top 12 comments as clean visual cards
        for _, row in filtered_comments.head(12).iterrows():
            sentiment_class = f"comment-box-{row['sentiment'].lower()}"
            badge_class = f"badge-{row['sentiment'].lower()}"
            st.markdown(f"""
            <div class="comment-box {sentiment_class}">
                <div class="comment-header">
                    <span class="comment-author">👤 {row['author']}</span>
                    <span>
                        <span class="badge {badge_class}">{row['sentiment']} ({row['polarity']:+.2f})</span>
                        &nbsp;👍 {row['likes']} likes
                    </span>
                </div>
                <div class="comment-body">{row['original_text']}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(filtered_comments) > 12:
            st.info(f"Showing top 12 of {len(filtered_comments)} comments. View full table in the 'Data Explorer & Export' tab.")

    # -----------------------------------------------------
    # TAB 3: Word Cloud & Keywords
    # -----------------------------------------------------
    with tab_wordcloud:
        st.subheader("Audience Discourse: Word Cloud & Top Keywords")
        st.write(
            "Word cloud and keyword extraction highlight recurring discussion topics, praised features, "
            "and audience feedback by eliminating generic filler and YouTube domain stopwords."
        )

        wc_col1, wc_col2 = st.columns([1.2, 0.8])

        with wc_col1:
            st.markdown("#### ☁️ Frequently Used Words (Word Cloud)")
            comments_text_list = df_comments["original_text"].tolist()
            wc_fig = generate_wordcloud_figure(comments_text_list, width=800, height=450, colormap="viridis")

            if wc_fig is not None:
                st.pyplot(wc_fig)
                plt.close(wc_fig)
            else:
                st.warning("Not enough textual data to generate word cloud.")

        with wc_col2:
            st.markdown("#### 📊 Top 15 Keywords Frequency")
            df_keywords = get_top_keywords(comments_text_list, top_n=15)

            if not df_keywords.empty:
                fig_kw = px.bar(
                    df_keywords.sort_values(by="frequency", ascending=True),
                    x="frequency",
                    y="keyword",
                    orientation="h",
                    color="frequency",
                    color_continuous_scale="Viridis",
                    labels={"frequency": "Mentions", "keyword": "Keyword"},
                    title="Most Mentioned Keywords"
                )
                fig_kw.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("No frequent keywords found.")

    # -----------------------------------------------------
    # TAB 4: Emoji Analysis
    # -----------------------------------------------------
    with tab_emoji:
        st.subheader("Audience Emotional Expression: Emoji Analytics")
        st.write(
            "Emojis reveal emotional tone beyond written text. CleanSpeak extracts audience emojis, "
            "quantifies their usage frequencies, and connects them with comment sentiment."
        )

        df_emojis = analyze_emojis(df_comments)

        if not df_emojis.empty:
            emo_col1, emo_col2 = st.columns([1, 1])

            with emo_col1:
                st.markdown("#### 🏆 Top Emojis by Frequency")
                fig_emoji_bar = px.bar(
                    df_emojis.head(10).sort_values(by="count", ascending=True),
                    x="count",
                    y="name",
                    orientation="h",
                    color="avg_polarity",
                    color_continuous_scale="Temps",
                    labels={"count": "Usage Count", "name": "Emoji Meaning", "avg_polarity": "Avg Polarity"},
                    title="Top 10 Emojis Used by Viewers"
                )
                fig_emoji_bar.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_emoji_bar, use_container_width=True)

            with emo_col2:
                st.markdown("#### 📋 Detailed Emoji Breakdown")
                # Format dataframe for display
                display_emoji_df = df_emojis.head(12)[["emoji", "name", "count", "avg_polarity", "sentiment_vibe"]]
                display_emoji_df.columns = ["Emoji", "Expression Name", "Count", "Sentiment Score", "Vibe"]
                st.dataframe(
                    display_emoji_df,
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.info("No emojis were detected in the collected comments.")

    # -----------------------------------------------------
    # TAB 5: ML Engagement Predictor (Random Forest)
    # -----------------------------------------------------
    with tab_ml:
        st.subheader("Machine Learning Engagement Predictor (Random Forest Regression)")
        st.write(
            "CleanSpeak employs **Random Forest Regression** to model the non-linear relationship between "
            "video-related features (views, title length, description length, tags, publish hour, duration, and audience sentiment) "
            "and engagement outcomes (potential likes and comments)."
        )

        # Model Performance Cards
        st.markdown("#### 🎯 Model Evaluation Metrics")
        ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)

        ml_col1.metric("Likes Model R² Score", f"{ml_metrics['likes']['r2']:.3f}", "Good Fit")
        ml_col2.metric("Likes Model MAE", f"±{ml_metrics['likes']['mae']:,.0f}", "Mean Abs Error")
        ml_col3.metric("Comments Model R² Score", f"{ml_metrics['comments']['r2']:.3f}", "High Fit")
        ml_col4.metric("Comments Model MAE", f"±{ml_metrics['comments']['mae']:,.0f}", "Mean Abs Error")

        st.markdown("---")

        # Feature Importance Chart
        st.markdown("#### 🔍 Random Forest Feature Importance")
        st.caption("Identifies which video characteristics have the largest relative predictive impact on viewer likes and comments.")

        fig_fi = px.bar(
            feature_importance_df,
            x="importance_avg",
            y="feature",
            orientation="h",
            color="importance_avg",
            color_continuous_scale="Purples",
            labels={"importance_avg": "Relative Importance (%)", "feature": "Video Feature"},
            title="Feature Importance in Engagement Prediction"
        )
        fig_fi.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
        fig_fi.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("---")

        # Interactive What-If Simulator
        st.markdown("#### 🎛️ Interactive 'What-If' Engagement Simulator")
        st.write(
            "Plan your upcoming video! Adjust the parameters below to predict potential likes, comments, and engagement rate "
            "using the trained Random Forest Regression models."
        )

        with st.container():
            sim_col1, sim_col2, sim_col3 = st.columns(3)

            curr_views = int(video_data.get("views", 10000))
            max_views_allowed = max(50_000_000_000, curr_views * 2)

            curr_title_len = int(np.clip(len(video_data.get("title", "")), 5, 200))

            curr_dur_sec = int(video_data.get("duration_seconds", 300))
            curr_dur_min = max(1, curr_dur_sec // 60)
            max_dur_min = max(180, curr_dur_min + 60)

            curr_tags = int(len(video_data.get("tags", [])))
            max_tags_allowed = max(40, curr_tags + 10)

            curr_desc_len = int(len(video_data.get("description", "")))
            max_desc_allowed = max(5000, curr_desc_len + 1000)

            curr_sentiment = float(np.clip(summary.get("avg_polarity", 0.25), -1.0, 1.0))

            with sim_col1:
                input_views = st.number_input(
                    "Target / Projected Views:",
                    min_value=0,
                    max_value=max_views_allowed,
                    value=curr_views,
                    step=max(10000, curr_views // 100 if curr_views > 0 else 10000)
                )
                input_title_len = st.slider(
                    "Title Length (Characters):",
                    min_value=5,
                    max_value=200,
                    value=curr_title_len
                )
                input_duration_min = st.slider(
                    "Video Duration (Minutes):",
                    min_value=1,
                    max_value=max_dur_min,
                    value=curr_dur_min
                )

            with sim_col2:
                input_tags = st.slider(
                    "Number of Tags:",
                    min_value=0,
                    max_value=max_tags_allowed,
                    value=curr_tags
                )
                input_desc_len = st.slider(
                    "Description Length (Characters):",
                    min_value=0,
                    max_value=max_desc_allowed,
                    value=curr_desc_len
                )
                input_hour = st.slider(
                    "Publish Hour (0 = Midnight, 14 = 2 PM):",
                    min_value=0,
                    max_value=23,
                    value=14
                )

            with sim_col3:
                input_sentiment = st.slider(
                    "Expected Sentiment Polarity (-1.0 to +1.0):",
                    min_value=-1.0,
                    max_value=1.0,
                    value=curr_sentiment,
                    step=0.05
                )

                st.markdown("<br>", unsafe_allow_html=True)
                calc_pred = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

            # Perform prediction
            input_dict = {
                "views": input_views,
                "title_length": input_title_len,
                "desc_length": input_desc_len,
                "tag_count": input_tags,
                "publish_hour": input_hour,
                "duration_sec": input_duration_min * 60,
                "sentiment_score": input_sentiment
            }

            predictions = predict_engagement(rf_likes, rf_comments, input_dict)

            st.markdown("<br>", unsafe_allow_html=True)
            res_c1, res_c2, res_c3 = st.columns(3)

            with res_c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Potential Likes</div>
                    <div class="metric-value" style="color: #3b82f6;">{predictions['predicted_likes']:,}</div>
                    <div class="metric-sub">Est. Like Rate: {predictions['predicted_like_ratio']}%</div>
                </div>
                """, unsafe_allow_html=True)

            with res_c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Comments</div>
                    <div class="metric-value" style="color: #8b5cf6;">{predictions['predicted_comments']:,}</div>
                    <div class="metric-sub">Est. Comment Rate: {predictions['predicted_comment_ratio']}%</div>
                </div>
                """, unsafe_allow_html=True)

            with res_c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Engagement Rate</div>
                    <div class="metric-value" style="color: #10b981;">{predictions['predicted_engagement_rate']}%</div>
                    <div class="metric-sub">Combined Engagement</div>
                </div>
                """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # TAB 6: Data Explorer & Export
    # -----------------------------------------------------
    with tab_data:
        st.subheader("Raw Data & Processed Comments Export")
        st.write("Browse all parsed comments with TextBlob sentiment scores or download as CSV / JSON for research and offline analysis.")

        export_df = df_comments[[
            "author", "original_text", "sentiment", "polarity", "subjectivity", "likes", "published_at"
        ]].copy()

        st.dataframe(export_df, use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Analyzed Comments (CSV)",
                data=csv_data,
                file_name=f"cleanspeak_comments_{video_data['id']}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_dl2:
            json_data = export_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                label="📥 Download Analyzed Comments (JSON)",
                data=json_data,
                file_name=f"cleanspeak_comments_{video_data['id']}.json",
                mime="application/json",
                use_container_width=True
            )

else:
    st.info("Please enter a YouTube video URL or select a sample video in the sidebar and click 'Analyze Video'.")
