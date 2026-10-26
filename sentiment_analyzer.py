"""
Sentiment Analysis module for CleanSpeak.
Uses TextBlob to compute polarity, subjectivity, and sentiment classifications
(Positive, Neutral, Negative) on YouTube audience comments.
"""

import re
import pandas as pd
from typing import List, Dict, Tuple
from textblob import TextBlob


def clean_text_for_sentiment(text: str) -> str:
    """
    Cleans comment text while retaining emojis and sentence structure
    essential for sentiment context.
    """
    if not isinstance(text, str):
        return ""
    # Strip URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Strip HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Normalize extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_sentiment(polarity: float) -> str:
    """
    Categorizes polarity into Positive, Neutral, or Negative.
    """
    if polarity > 0.10:
        return "Positive"
    elif polarity < -0.10:
        return "Negative"
    else:
        return "Neutral"


def analyze_comments_sentiment(comments: List[Dict]) -> pd.DataFrame:
    """
    Processes a list of comment dictionaries and returns an enriched pandas DataFrame
    with polarity, subjectivity, and sentiment label.
    """
    records = []
    for c in comments:
        raw_text = c.get("text", "")
        cleaned = clean_text_for_sentiment(raw_text)
        blob = TextBlob(cleaned)
        polarity = round(blob.sentiment.polarity, 4)
        subjectivity = round(blob.sentiment.subjectivity, 4)
        sentiment_label = classify_sentiment(polarity)

        records.append({
            "author": c.get("author", "Anonymous"),
            "original_text": raw_text,
            "cleaned_text": cleaned,
            "likes": int(c.get("likes", 0)),
            "published_at": c.get("published_at", ""),
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment_label
        })

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=[
            "author", "original_text", "cleaned_text", "likes",
            "published_at", "polarity", "subjectivity", "sentiment"
        ])
    return df


def calculate_sentiment_summary(df: pd.DataFrame) -> Dict:
    """
    Calculates aggregate summary metrics from the analyzed comments DataFrame.
    """
    if df.empty:
        return {
            "total_comments": 0,
            "avg_polarity": 0.0,
            "avg_subjectivity": 0.0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "positive_pct": 0.0,
            "neutral_pct": 0.0,
            "negative_pct": 0.0,
            "dominant_sentiment": "None"
        }

    total = len(df)
    pos_count = int((df["sentiment"] == "Positive").sum())
    neu_count = int((df["sentiment"] == "Neutral").sum())
    neg_count = int((df["sentiment"] == "Negative").sum())

    pos_pct = round((pos_count / total) * 100, 1)
    neu_pct = round((neu_count / total) * 100, 1)
    neg_pct = round((neg_count / total) * 100, 1)

    avg_pol = round(df["polarity"].mean(), 3)
    avg_sub = round(df["subjectivity"].mean(), 3)

    counts = {"Positive": pos_count, "Neutral": neu_count, "Negative": neg_count}
    dominant = max(counts, key=counts.get)

    return {
        "total_comments": total,
        "avg_polarity": avg_pol,
        "avg_subjectivity": avg_sub,
        "positive_count": pos_count,
        "neutral_count": neu_count,
        "negative_count": neg_count,
        "positive_pct": pos_pct,
        "neutral_pct": neu_pct,
        "negative_pct": neg_pct,
        "dominant_sentiment": dominant
    }


def calculate_engagement_metrics(views: int, likes: int, comments: int) -> Dict:
    """
    Calculates video performance KPIs and engagement rates.
    """
    views = max(1, views)
    engagement_rate = round(((likes + comments) / views) * 100, 2)
    like_ratio = round((likes / views) * 100, 2)
    comment_ratio = round((comments / views) * 100, 3)

    # Industry benchmark evaluation
    # Average YouTube engagement rate: 1.5% - 3.5%, high: >4%, exceptional: >7%
    if engagement_rate >= 7.0:
        health_status = "Exceptional 🔥"
        health_color = "#10B981"  # Emerald
    elif engagement_rate >= 4.0:
        health_status = "High Growth 🚀"
        health_color = "#3B82F6"  # Blue
    elif engagement_rate >= 2.0:
        health_status = "Healthy / Normal 👍"
        health_color = "#6366F1"  # Indigo
    else:
        health_status = "Below Average ⚠️"
        health_color = "#F59E0B"  # Amber

    return {
        "engagement_rate": engagement_rate,
        "like_ratio": like_ratio,
        "comment_ratio": comment_ratio,
        "health_status": health_status,
        "health_color": health_color
    }
