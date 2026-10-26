"""
YouTube Data Extractor module for CleanSpeak.
Handles parsing YouTube video URLs and extracting video metadata and comments
using the YouTube Data API v3, with fallback to built-in sample data.
"""

import re
from typing import Dict, List, Tuple, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sample_data import SAMPLE_VIDEOS, SAMPLE_COMMENTS


def extract_video_id(url: str) -> Optional[str]:
    """
    Extracts the 11-character YouTube video ID from various URL formats
    or validates direct ID inputs.

    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/live/VIDEO_ID
      - Direct 11-char ID
    """
    if not url:
        return None

    url = url.strip()

    # Direct 11 character alphanumeric/hyphen/underscore ID
    if re.fullmatch(r"^[a-zA-Z0-9_-]{11}$", url):
        return url

    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/|\/live\/)([a-zA-Z0-9_-]{11})",
        r"(?:watch\?.*v=)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def parse_iso_duration(iso_duration: str) -> Tuple[str, int]:
    """
    Converts ISO 8601 duration string (e.g. PT42M15S, PT1H2M3S) to (formatted_str, total_seconds).
    Pure python implementation without external dependency.
    """
    if not iso_duration:
        return "0:00", 0

    match = re.match(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
    if not match:
        return "0:00", 0

    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0) + (days * 24)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)

    total_seconds = hours * 3600 + minutes * 60 + seconds

    if hours > 0:
        formatted = f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        formatted = f"{minutes}:{seconds:02d}"

    return formatted, total_seconds


def fetch_video_details_api(api_key: str, video_id: str) -> Dict:
    """
    Fetches video metadata and statistics from YouTube Data API v3.
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    response = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=video_id
    ).execute()

    items = response.get("items", [])
    if not items:
        raise ValueError(f"No video found for ID: {video_id}. Please check the link.")

    video = items[0]
    snippet = video.get("snippet", {})
    stats = video.get("statistics", {})
    content = video.get("contentDetails", {})

    iso_dur = content.get("duration", "PT0S")
    formatted_dur, total_seconds = parse_iso_duration(iso_dur)

    thumbnails = snippet.get("thumbnails", {})
    thumb_url = (
        thumbnails.get("maxres", {}).get("url")
        or thumbnails.get("high", {}).get("url")
        or thumbnails.get("medium", {}).get("url")
        or thumbnails.get("default", {}).get("url")
        or ""
    )

    tags = snippet.get("tags", [])
    views = int(stats.get("viewCount", 0))
    likes = int(stats.get("likeCount", 0))
    comment_count = int(stats.get("commentCount", 0))

    return {
        "id": video_id,
        "title": snippet.get("title", "Unknown Title"),
        "channel_title": snippet.get("channelTitle", "Unknown Channel"),
        "published_at": snippet.get("publishedAt", ""),
        "views": views,
        "likes": likes,
        "comment_count": comment_count,
        "duration": formatted_dur,
        "duration_seconds": total_seconds,
        "category_id": snippet.get("categoryId", "0"),
        "thumbnail_url": thumb_url,
        "description": snippet.get("description", ""),
        "tags": tags
    }


def fetch_video_comments_api(api_key: str, video_id: str, max_results: int = 100) -> List[Dict]:
    """
    Fetches comments for the specified video ID using YouTube Data API v3.
    Retrieves up to `max_results` top-level comments ordered by relevance.
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    page_token = None

    try:
        while len(comments) < max_results:
            page_batch_size = min(100, max_results - len(comments))
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=page_batch_size,
                order="relevance",
                pageToken=page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response.get("items", []):
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                author = snippet.get("authorDisplayName", "Anonymous")
                text = snippet.get("textDisplay", "")
                like_count = int(snippet.get("likeCount", 0))
                published_at = snippet.get("publishedAt", "")

                if text.strip():
                    comments.append({
                        "author": author,
                        "text": text,
                        "likes": like_count,
                        "published_at": published_at
                    })

                if len(comments) >= max_results:
                    break

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    except HttpError as err:
        if "commentsDisabled" in str(err):
            raise ValueError("Comments are disabled for this video.")
        elif "quotaExceeded" in str(err):
            raise ValueError("YouTube API quota exceeded for your API key. Try using Demo Mode.")
        else:
            raise ValueError(f"YouTube API Error: {err}")

    return comments


def get_sample_video_and_comments(sample_key: str = "tech_ai_crash_course") -> Tuple[Dict, List[Dict]]:
    """
    Returns built-in sample video data and comments for offline demonstration.
    """
    video_data = SAMPLE_VIDEOS.get(sample_key, SAMPLE_VIDEOS["tech_ai_crash_course"])
    comments_data = SAMPLE_COMMENTS.get(sample_key, SAMPLE_COMMENTS["tech_ai_crash_course"])
    return video_data, comments_data
