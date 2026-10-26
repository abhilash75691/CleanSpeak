"""
Machine Learning module for CleanSpeak.
Uses Random Forest Regression to predict potential likes and comments
based on video-related features (views, title length, description length, tags, publish hour, duration, sentiment).
"""

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sample_data import get_benchmark_dataset

FEATURE_COLUMNS = [
    "views",
    "title_length",
    "desc_length",
    "tag_count",
    "publish_hour",
    "duration_sec",
    "sentiment_score"
]

FEATURE_LABELS = {
    "views": "Views Count",
    "title_length": "Title Length (chars)",
    "desc_length": "Description Length (chars)",
    "tag_count": "Tag Count",
    "publish_hour": "Publish Hour (0-23)",
    "duration_sec": "Duration (seconds)",
    "sentiment_score": "Sentiment Score (-1 to 1)"
}


def train_rf_models(df: pd.DataFrame = None) -> Tuple[RandomForestRegressor, RandomForestRegressor, Dict, pd.DataFrame]:
    """
    Trains two Random Forest Regressors:
    1. Predicts potential Likes
    2. Predicts potential Comments

    Returns:
      (model_likes, model_comments, evaluation_metrics_dict, feature_importance_df)
    """
    if df is None or df.empty:
        df = get_benchmark_dataset(num_samples=600)

    X = df[FEATURE_COLUMNS]
    y_likes = df["likes"]
    y_comments = df["comments"]

    # 80/20 train/test split
    X_train, X_test, y_likes_train, y_likes_test, y_comm_train, y_comm_test = train_test_split(
        X, y_likes, y_comments, test_size=0.2, random_state=42
    )

    # Model 1: Random Forest for Likes
    rf_likes = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_likes.fit(X_train, y_likes_train)
    y_likes_pred = rf_likes.predict(X_test)

    # Model 2: Random Forest for Comments
    rf_comments = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_comments.fit(X_train, y_comm_train)
    y_comm_pred = rf_comments.predict(X_test)

    # Metrics computation
    metrics = {
        "likes": {
            "r2": round(float(r2_score(y_likes_test, y_likes_pred)), 3),
            "mae": round(float(mean_absolute_error(y_likes_test, y_likes_pred)), 1),
            "rmse": round(float(root_mean_squared_error(y_likes_test, y_likes_pred)), 1),
        },
        "comments": {
            "r2": round(float(r2_score(y_comm_test, y_comm_pred)), 3),
            "mae": round(float(mean_absolute_error(y_comm_test, y_comm_pred)), 1),
            "rmse": round(float(root_mean_squared_error(y_comm_test, y_comm_pred)), 1),
        }
    }

    # Feature Importance aggregation
    fi_df = pd.DataFrame({
        "feature": [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
        "feature_raw": FEATURE_COLUMNS,
        "importance_likes": np.round(rf_likes.feature_importances_ * 100, 2),
        "importance_comments": np.round(rf_comments.feature_importances_ * 100, 2)
    })
    fi_df["importance_avg"] = np.round((fi_df["importance_likes"] + fi_df["importance_comments"]) / 2, 2)
    fi_df = fi_df.sort_values(by="importance_avg", ascending=False).reset_index(drop=True)

    return rf_likes, rf_comments, metrics, fi_df


def predict_engagement(
    rf_likes: RandomForestRegressor,
    rf_comments: RandomForestRegressor,
    input_data: Dict[str, float]
) -> Dict[str, float]:
    """
    Predicts potential likes and comments given a dictionary of video features.
    """
    row = [input_data.get(col, 0) for col in FEATURE_COLUMNS]
    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    pred_likes = max(0, int(round(rf_likes.predict(input_df)[0])))
    pred_comments = max(0, int(round(rf_comments.predict(input_df)[0])))

    views = max(1, input_data.get("views", 1))
    predicted_engagement_rate = round(((pred_likes + pred_comments) / views) * 100, 2)
    predicted_like_ratio = round((pred_likes / views) * 100, 2)
    predicted_comment_ratio = round((pred_comments / views) * 100, 3)

    return {
        "predicted_likes": pred_likes,
        "predicted_comments": pred_comments,
        "predicted_engagement_rate": predicted_engagement_rate,
        "predicted_like_ratio": predicted_like_ratio,
        "predicted_comment_ratio": predicted_comment_ratio
    }
