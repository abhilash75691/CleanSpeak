"""
Text and Emoji Analysis module for CleanSpeak.
Provides WordCloud generation, top keyword extraction, and audience emoji frequency/sentiment analytics.
"""

import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import emoji

# Domain-specific common filler words to exclude from keyword / wordcloud analysis
CUSTOM_STOPWORDS = {
    "video", "youtube", "channel", "watch", "watching", "subscribe", "subscribers",
    "thank", "thanks", "make", "really", "will", "please", "can", "good", "great",
    "just", "also", "one", "people", "would", "much", "even", "still", "well",
    "see", "know", "get", "got", "way", "first", "think", "time", "video's"
}


def clean_text_for_wordcloud(text: str) -> str:
    """
    Cleans text by removing URLs, mentions, emojis, special characters, and digits.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove emojis
    text = emoji.replace_emoji(text, replace="")
    # Remove special characters, punctuation, and digits
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Normalize spaces and lowercase
    text = re.sub(r"\s+", " ", text).lower().strip()
    return text


def generate_wordcloud_figure(
    texts: List[str],
    width: int = 900,
    height: int = 450,
    background_color: str = "#ffffff",
    colormap: str = "magma"
) -> Optional[plt.Figure]:
    """
    Generates a high-resolution Matplotlib figure containing the WordCloud.
    """
    combined = " ".join([clean_text_for_wordcloud(t) for t in texts if t])
    if not combined.strip():
        return None

    stopwords = set(STOPWORDS).union(CUSTOM_STOPWORDS)

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        stopwords=stopwords,
        max_words=120,
        min_font_size=10,
        random_state=42,
        collocations=False
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=background_color)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def get_top_keywords(texts: List[str], top_n: int = 15) -> pd.DataFrame:
    """
    Extracts the top N most frequent keywords from comment texts, excluding stopwords.
    """
    stopwords = set(STOPWORDS).union(CUSTOM_STOPWORDS)
    tokens = []

    for t in texts:
        cleaned = clean_text_for_wordcloud(t)
        words = cleaned.split()
        for w in words:
            if len(w) > 2 and w not in stopwords:
                tokens.append(w)

    counter = Counter(tokens)
    most_common = counter.most_common(top_n)

    if not most_common:
        return pd.DataFrame(columns=["keyword", "frequency"])

    df = pd.DataFrame(most_common, columns=["keyword", "frequency"])
    return df


def analyze_emojis(df_comments: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes emoji usage in comments:
    - Identifies all emojis present
    - Counts their frequencies
    - Converts emoji codes to human-readable names
    - Calculates average sentiment polarity of comments containing each emoji
    """
    if df_comments.empty or "original_text" not in df_comments.columns:
        return pd.DataFrame(columns=["emoji", "name", "count", "avg_polarity", "sentiment_vibe"])

    emoji_data = {}  # emoji_char -> list of polarity scores

    for _, row in df_comments.iterrows():
        text = str(row.get("original_text", ""))
        polarity = float(row.get("polarity", 0.0))
        extracted = emoji.emoji_list(text)

        # Count unique emojis per comment to avoid spam distortion
        unique_in_comment = set(item["emoji"] for item in extracted)
        for emo in unique_in_comment:
            if emo not in emoji_data:
                emoji_data[emo] = []
            emoji_data[emo].append(polarity)

    if not emoji_data:
        return pd.DataFrame(columns=["emoji", "name", "count", "avg_polarity", "sentiment_vibe"])

    rows = []
    for emo, pol_list in emoji_data.items():
        count = len(pol_list)
        avg_pol = round(sum(pol_list) / count, 3)

        # Human readable name
        raw_name = emoji.demojize(emo).strip(":")
        clean_name = raw_name.replace("_", " ").title()

        if avg_pol > 0.15:
            vibe = "Positive 😊"
        elif avg_pol < -0.10:
            vibe = "Negative 😠"
        else:
            vibe = "Neutral 😐"

        rows.append({
            "emoji": emo,
            "name": clean_name,
            "count": count,
            "avg_polarity": avg_pol,
            "sentiment_vibe": vibe
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(by="count", ascending=False).reset_index(drop=True)
    return result_df
