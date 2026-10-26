"""
Sample data module for CleanSpeak.
Provides realistic sample YouTube video metadata, comments, and benchmark
dataset for machine learning, enabling immediate execution without requiring an API key.
"""

import pandas as pd
import numpy as np

SAMPLE_VIDEOS = {
    "tech_ai_crash_course": {
        "id": "dQw4w9WgXcQ",
        "title": "Building Full-Stack AI Applications in 2026 - Complete Crash Course",
        "channel_title": "CodeCraft Academy",
        "published_at": "2026-03-15T14:30:00Z",
        "views": 485200,
        "likes": 27400,
        "comment_count": 1820,
        "duration": "42:15",
        "duration_seconds": 2535,
        "category": "Science & Technology",
        "thumbnail_url": "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=800&auto=format&fit=crop&q=60",
        "description": (
            "Learn how to build, deploy, and scale production-ready AI applications in this comprehensive tutorial. "
            "We cover LLMs, agentic workflows, vector databases, FastAPI backends, and interactive Streamlit frontends. "
            "All source code is available in the GitHub repository linked below!"
        ),
        "tags": ["AI", "Python", "Streamlit", "Machine Learning", "Full Stack", "Tutorial", "FastAPI", "Deep Learning"]
    },
    "tech_gadget_review": {
        "id": "abc123XYZ89",
        "title": "Smartphone 16 Ultra Review: The Honest Truth After 30 Days",
        "channel_title": "TechVanguard",
        "published_at": "2026-02-10T18:00:00Z",
        "views": 890400,
        "likes": 39200,
        "comment_count": 3410,
        "duration": "18:45",
        "duration_seconds": 1125,
        "category": "Technology & Gadgets",
        "thumbnail_url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=800&auto=format&fit=crop&q=60",
        "description": (
            "Here is my full 30-day in-depth review of the new flagship device. Is the battery upgrade worth it? "
            "We test the camera in low light, stress test the thermal throttling, and compare it against competitors."
        ),
        "tags": ["Smartphone", "Review", "Tech", "Camera Test", "Battery Life", "Comparison", "Flagship"]
    },
    "fitness_workout_guide": {
        "id": "fit987LMN45",
        "title": "20-Minute High Intensity Full Body HIIT Workout (No Equipment Needed)",
        "channel_title": "Pulse Fitness",
        "published_at": "2026-01-20T09:15:00Z",
        "views": 320100,
        "likes": 19800,
        "comment_count": 940,
        "duration": "22:10",
        "duration_seconds": 1330,
        "category": "Health & Fitness",
        "thumbnail_url": "https://images.unsplash.com/photo-1517838277536-f5f99be501cd?w=800&auto=format&fit=crop&q=60",
        "description": (
            "Burn calories and boost endurance with this 20-minute bodyweight HIIT workout! Follow along round by round. "
            "Remember to hydrate and take rest intervals if needed. Let me know your progress in the comments!"
        ),
        "tags": ["Workout", "Fitness", "HIIT", "Health", "Cardio", "Home Workout", "No Equipment"]
    }
}

# Detailed comments for each sample video with realistic sentiments, varied vocabulary, and emojis
SAMPLE_COMMENTS = {
    "tech_ai_crash_course": [
        {"author": "Alex Rivera", "text": "This is easily the best and most structured AI tutorial on YouTube! The architecture diagram cleared up so much confusion. 🔥🚀", "likes": 342, "published_at": "2026-03-15T15:10:00Z"},
        {"author": "Sophia Chen", "text": "Loved how you explained vector databases simply. Super clear explanation, thank you! ❤️👏", "likes": 189, "published_at": "2026-03-15T16:22:00Z"},
        {"author": "DevGuru99", "text": "Incredible work! Finally someone explaining agentic orchestration without unnecessary jargon. 💯🙌", "likes": 154, "published_at": "2026-03-15T17:05:00Z"},
        {"author": "Marcus Brody", "text": "Timestamp 18:24 saved my final year college project! You deserve a million subscribers. ⭐✨", "likes": 98, "published_at": "2026-03-15T18:12:00Z"},
        {"author": "Priya Sharma", "text": "Phenomenal video! The code quality is enterprise level. Keep making these high quality tutorials! 🚀❤️", "likes": 121, "published_at": "2026-03-15T19:00:00Z"},
        {"author": "Liam Vance", "text": "The audio was slightly quiet between 12:00 and 15:00, but otherwise brilliant content.", "likes": 42, "published_at": "2026-03-15T19:35:00Z"},
        {"author": "Elena Rostova", "text": "Could you please do a follow-up on fine-tuning models locally using Ollama? That would be awesome! 👍🤖", "likes": 88, "published_at": "2026-03-15T20:10:00Z"},
        {"author": "CyberKnight", "text": "I tried running the GitHub repo code and ran into a dependency mismatch with Pydantic v2. Please check requirements.txt.", "likes": 35, "published_at": "2026-03-15T21:04:00Z"},
        {"author": "Noah Kim", "text": "Clean code, zero fluff, straight to the point. Subscribed immediately! 🔥🔥", "likes": 77, "published_at": "2026-03-15T22:30:00Z"},
        {"author": "David Miller", "text": "I was struggling with streaming responses in FastAPI and your example fixed it in 2 minutes. Thank you so much! 🎉", "likes": 64, "published_at": "2026-03-16T01:15:00Z"},
        {"author": "Jessica Taylor", "text": "Very informative and well edited. Enjoyed watching this on my weekend study session.", "likes": 29, "published_at": "2026-03-16T03:20:00Z"},
        {"author": "Kevin Zhang", "text": "Terrible pacing in the first 5 minutes. Got bored and almost clicked off.", "likes": 14, "published_at": "2026-03-16T04:12:00Z"},
        {"author": "Rachel Green", "text": "Great breakdown! I learned more in 40 minutes than in my whole university semester! 💡🎓", "likes": 210, "published_at": "2026-03-16T05:40:00Z"},
        {"author": "Lucas Silva", "text": "The explanation of LangChain vs custom agent state loops was gold. Absolutely top tier! 🥇", "likes": 95, "published_at": "2026-03-16T06:33:00Z"},
        {"author": "Emily Watson", "text": "Why do people still recommend ChromaDB when Qdrant and Milvus are faster? A bit disappointed by that choice.", "likes": 23, "published_at": "2026-03-16T07:18:00Z"},
        {"author": "Carlos Gomez", "text": "Brilliant project design. Already started building my own version for legal document retrieval! 📂🤖", "likes": 48, "published_at": "2026-03-16T08:00:00Z"},
        {"author": "Fatima Zahra", "text": "Such clear pronunciation and calm delivery. It makes complex concepts very easy to follow. 🙏", "likes": 56, "published_at": "2026-03-16T08:50:00Z"},
        {"author": "Daniel Lee", "text": "Does this work on Windows? Or do we need WSL2 for Docker?", "likes": 17, "published_at": "2026-03-16T09:22:00Z"},
        {"author": "Samir Patel", "text": "Yes Daniel, Docker Desktop on Windows with WSL2 backend runs this smoothly.", "likes": 12, "published_at": "2026-03-16T10:05:00Z"},
        {"author": "Chloe Bennett", "text": "One of the best programming tutorials ever uploaded to YouTube! ❤️❤️❤️", "likes": 83, "published_at": "2026-03-16T11:45:00Z"},
        {"author": "Benjamin Scott", "text": "Honestly didn't like the color theme of the code editor. Hurt my eyes a bit.", "likes": 8, "published_at": "2026-03-16T12:30:00Z"},
        {"author": "Maya Lin", "text": "Bookmarked and shared with my engineering team. Fantastic resource! 🚀💡", "likes": 73, "published_at": "2026-03-16T13:14:00Z"},
        {"author": "Oliver Queen", "text": "Straightforward, practical, and no sponsored ads every 5 minutes. Respect! 🤝", "likes": 105, "published_at": "2026-03-16T14:20:00Z"},
        {"author": "Zara Ali", "text": "Can someone share the link to Discord community?", "likes": 5, "published_at": "2026-03-16T15:02:00Z"},
        {"author": "Victor Hugo", "text": "Masterpiece! The Streamlit UI integration part was super slick. 🎨✨", "likes": 61, "published_at": "2026-03-16T15:45:00Z"},
        {"author": "Emma Davis", "text": "I was confused about embeddings, but your visual analogy made it click instantly! 🧠💡", "likes": 92, "published_at": "2026-03-16T16:10:00Z"},
        {"author": "Hassan Raza", "text": "Not worth the hype. Basic concepts covered on official docs anyway.", "likes": 11, "published_at": "2026-03-16T17:00:00Z"},
        {"author": "Grace Hopper", "text": "Inspiring tutorial! Building AI apps is so accessible now. Thank you creator! 👏💖", "likes": 115, "published_at": "2026-03-16T17:40:00Z"},
        {"author": "Nate Robinson", "text": "Awesome video, watched the whole thing at 1.5x speed and took 4 pages of notes! 📝🔥", "likes": 68, "published_at": "2026-03-16T18:30:00Z"},
        {"author": "Aisha Khan", "text": "Extremely helpful and well structured. Will recommend to all my classmates. 👍🎓", "likes": 44, "published_at": "2026-03-16T19:15:00Z"},
        {"author": "Tom Hardy", "text": "Too many cuts and the background music was annoying.", "likes": 9, "published_at": "2026-03-16T20:00:00Z"},
        {"author": "Hannah Abbott", "text": "Loved every second! Can't wait for Part 2! 🎉🚀", "likes": 53, "published_at": "2026-03-16T20:45:00Z"},
        {"author": "Leo Messi", "text": "Goated tutorial! 🐐⚽", "likes": 81, "published_at": "2026-03-16T21:10:00Z"},
        {"author": "Sandra Bullock", "text": "Clear instructions, easy setup, worked on the first try. Perfect! 💯", "likes": 39, "published_at": "2026-03-16T22:00:00Z"},
        {"author": "Jack Sparrow", "text": "Ahoy! Finally someone who doesn't rush through the tricky code segments. 🏴‍☠️⚓", "likes": 31, "published_at": "2026-03-16T22:45:00Z"},
        {"author": "Mia Khalifa", "text": "What version of Python is recommended for this project?", "likes": 14, "published_at": "2026-03-16T23:20:00Z"},
        {"author": "Bruce Wayne", "text": "Dark mode everywhere. Approved. 🦇", "likes": 120, "published_at": "2026-03-17T00:05:00Z"},
        {"author": "Barry Allen", "text": "Watched this so fast! Super practical code. ⚡", "likes": 47, "published_at": "2026-03-17T01:00:00Z"},
        {"author": "Clark Kent", "text": "Superb presentation of complex concepts. A genuine service to the developer community! 🦸‍♂️", "likes": 63, "published_at": "2026-03-17T02:15:00Z"},
        {"author": "Diana Prince", "text": "Empowering and thoroughly researched. 5 stars out of 5! ⭐⭐⭐⭐⭐", "likes": 75, "published_at": "2026-03-17T03:30:00Z"}
    ],
    "tech_gadget_review": [
        {"author": "TechLover", "text": "Best review on the market! Honest, no hype, just real testing. Thank you! 🔥👍", "likes": 540, "published_at": "2026-02-10T18:30:00Z"},
        {"author": "Sarah Jenkins", "text": "The camera low-light test surprised me. The shadows look way more natural than last year! 📸✨", "likes": 230, "published_at": "2026-02-10T19:15:00Z"},
        {"author": "Dave K", "text": "Total waste of money. The battery only lasted 5 hours on my unit. Terrible optimization! 😡👎", "likes": 88, "published_at": "2026-02-10T20:00:00Z"},
        {"author": "Mark Spencer", "text": "I really appreciate you talking about the thermal throttling during gaming. Most other reviewers ignored it! 🎮🔥", "likes": 165, "published_at": "2026-02-10T21:10:00Z"},
        {"author": "Anna Bell", "text": "Is the titanium frame prone to scratches like earlier models?", "likes": 45, "published_at": "2026-02-10T22:05:00Z"},
        {"author": "Chris P", "text": "Not worth upgrading if you already have the 15 Pro. Barely any noticeable difference. 🤷‍♂️", "likes": 312, "published_at": "2026-02-10T23:00:00Z"},
        {"author": "Julia Roberts", "text": "Gorgeous b-roll shots and smooth narration. Always look forward to your uploads! ❤️🎥", "likes": 95, "published_at": "2026-02-11T00:30:00Z"},
        {"author": "Robbie M", "text": "The price tag is ridiculous for what you get. Company is getting greedy.", "likes": 140, "published_at": "2026-02-11T02:00:00Z"},
        {"author": "Lucas Grey", "text": "Clean review! Direct comparison at 12:40 made my purchasing decision so easy. Bought one today! 📱🚀", "likes": 82, "published_at": "2026-02-11T03:15:00Z"},
        {"author": "Olivia Wilde", "text": "The audio recording quality on the mic test was surprisingly crisp. 🎙️👌", "likes": 50, "published_at": "2026-02-11T04:20:00Z"}
    ],
    "fitness_workout_guide": [
        {"author": "Jenny Fit", "text": "My legs are shaking but I feel amazing! Best 20 minutes of my day! 💦💪🔥", "likes": 310, "published_at": "2026-01-20T10:00:00Z"},
        {"author": "Carlos M", "text": "Day 14 of doing this daily. Down 4 pounds and my energy is through the roof! Thank you! 🏃‍♂️⚡", "likes": 180, "published_at": "2026-01-20T11:20:00Z"},
        {"author": "Linda Carter", "text": "Burpees at the end were brutal haha! But so worth it! 😭👏💯", "likes": 95, "published_at": "2026-01-20T12:00:00Z"},
        {"author": "Tom Hardy", "text": "Great low-impact modifications shown on screen. Very helpful for bad knees! 👍❤️", "likes": 125, "published_at": "2026-01-20T13:45:00Z"},
        {"author": "Megan Fox", "text": "Music volume was a bit too loud compared to your voice cues.", "likes": 32, "published_at": "2026-01-20T14:30:00Z"},
        {"author": "David B", "text": "Excellent coaching and encouragement throughout. Kept me pushing! 🏆", "likes": 74, "published_at": "2026-01-20T15:10:00Z"}
    ]
}


def get_benchmark_dataset(num_samples: int = 500) -> pd.DataFrame:
    """
    Generates a realistic synthetic benchmark dataset of YouTube video performance
    for training the Random Forest Regressors predicting Likes and Comments.
    Features:
      - views: Total view count
      - title_length: Character length of video title
      - desc_length: Character length of description
      - tag_count: Number of tags attached to video
      - publish_hour: Hour of day when video was published (0-23)
      - duration_sec: Video duration in seconds
      - sentiment_score: Average sentiment polarity (-1.0 to +1.0)
    Targets:
      - likes: Number of likes
      - comments: Number of comments
    """
    np.random.seed(42)

    # Views range from 5,000 to 2,500,000 (log-normal distribution)
    views = np.random.lognormal(mean=11.5, sigma=1.2, size=num_samples).astype(int)
    views = np.clip(views, 5000, 5000000)

    title_length = np.random.randint(20, 95, size=num_samples)
    desc_length = np.random.randint(150, 2500, size=num_samples)
    tag_count = np.random.randint(3, 25, size=num_samples)
    publish_hour = np.random.randint(0, 24, size=num_samples)
    duration_sec = np.random.randint(180, 3600, size=num_samples)
    sentiment_score = np.random.normal(loc=0.25, scale=0.18, size=num_samples)
    sentiment_score = np.clip(sentiment_score, -0.6, 0.9)

    # YouTube statistics show average like-to-view ratio is 2% - 7%, comments 0.1% - 0.5%
    # Influenced positively by higher sentiment, optimal title length (40-60 chars), tag count, and duration
    title_boost = 1.0 + 0.15 * np.sin((title_length - 20) / 75.0 * np.pi)
    sentiment_boost = 1.0 + 0.35 * sentiment_score
    tag_boost = 1.0 + 0.1 * (tag_count / 25.0)

    # Base like rate ~ 3.5% with random noise
    base_like_rate = np.random.normal(loc=0.038, scale=0.009, size=num_samples)
    base_like_rate = np.clip(base_like_rate, 0.012, 0.085)
    likes = (views * base_like_rate * title_boost * sentiment_boost * tag_boost).astype(int)
    likes = np.maximum(likes, 10)

    # Base comment rate ~ 0.3% with noise
    base_comment_rate = np.random.normal(loc=0.0035, scale=0.001, size=num_samples)
    base_comment_rate = np.clip(base_comment_rate, 0.0008, 0.012)
    # Higher sentiment controversy or high passion triggers more comments
    controversy_factor = 1.0 + 0.4 * np.abs(sentiment_score - 0.2)
    comments = (views * base_comment_rate * controversy_factor * tag_boost).astype(int)
    comments = np.maximum(comments, 2)

    df = pd.DataFrame({
        "views": views,
        "title_length": title_length,
        "desc_length": desc_length,
        "tag_count": tag_count,
        "publish_hour": publish_hour,
        "duration_sec": duration_sec,
        "sentiment_score": np.round(sentiment_score, 3),
        "likes": likes,
        "comments": comments
    })

    return df
