# CleanSpeak: YouTube Video Analytics and Audience Insights Application

**CleanSpeak** is an interactive, data-driven YouTube video analytics and audience insights application built with Python and Streamlit. It allows content creators, researchers, and digital marketers to understand video performance, audience reception, and emotional responses through Natural Language Processing (NLP) and Machine Learning (ML).

---

## 📌 Key Features

1. **YouTube Data Extraction**
   - Seamlessly extracts video metadata (title, channel, published date, tags, category, duration) and engagement metrics (views, likes, comments) using the **YouTube Data API v3**.
   - Supports all standard YouTube URL formats, Short links, and direct video IDs.
   - Built-in **Demo Sample Mode** allowing instant testing and demonstration without needing an API key.

2. **Video Performance & Benchmarking**
   - Computes key engagement metrics including:
     - **Engagement Rate**: `((Likes + Comments) / Views) * 100`
     - **Like-to-View Ratio** and **Comment-to-View Ratio**
   - Gauges performance against YouTube industry benchmarks (Normal, High Growth, Exceptional).

3. **Audience Sentiment Analysis (TextBlob)**
   - Performs NLP sentiment classification on audience comments using **TextBlob**.
   - Computes Polarity score (`-1.0` to `+1.0`) and Subjectivity (`0.0` to `1.0`).
   - Categorizes comments into **Positive**, **Neutral**, and **Negative**.
   - Donut charts, polarity distribution histograms, and an interactive comment explorer with category filters and text search.

4. **Word Cloud & Keyword Insights**
   - Generates high-resolution **Word Cloud** visualizations from viewer comments.
   - Filters domain-specific stopwords and fillers to surface meaningful themes.
   - Plots the top 15 most frequent keywords with interactive Plotly bar charts.

5. **Audience Emotional Expression (Emoji Analysis)**
   - Extracts all Unicode emojis used across comments.
   - Quantifies emoji frequencies and translates codes into human-readable names.
   - Computes average sentiment polarity associated with each emoji to reveal audience tone.

6. **Machine Learning Engagement Predictor (Random Forest Regression)**
   - Implements **Random Forest Regressors** (`scikit-learn`) to model the non-linear relationship between video characteristics and engagement outcomes.
   - Features analyzed: Views, Title Length, Description Length, Tag Count, Publish Hour, Duration, and Comment Sentiment Polarity.
   - Predicts **Potential Likes** and **Potential Comments**.
   - Provides Model Evaluation Metrics ($R^2$ Score, MAE, RMSE) and a **Feature Importance** chart.
   - Includes an interactive **"What-If" Video Simulator** for creators planning upcoming uploads.

7. **Data Explorer & Export**
   - Filter and inspect all processed comments and sentiment scores in a structured table.
   - One-click export to **CSV** and **JSON** formats.

---

## 📁 Project Structure

```
cleanspeak/
│
├── app.py                   # Main Streamlit dashboard application
├── youtube_extractor.py     # YouTube Data API v3 integration & URL parsing
├── sentiment_analyzer.py    # TextBlob sentiment analysis & engagement calculations
├── text_emoji_analyzer.py   # WordCloud, keyword extraction & emoji analytics
├── ml_predictor.py          # Random Forest Regression models & feature importance
├── sample_data.py           # Preloaded realistic sample videos & benchmark dataset
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation and execution instructions
```

---

## 🚀 Installation & Execution

### 1. Prerequisites
- Python 3.9+ or Python 3.11+
- Git (optional)

### 2. Navigate to the project directory
```bash
cd "C:\Users\chand\.gemini\antigravity\scratch\cleanspeak"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
python -m streamlit run app.py
```
*(Or `streamlit run app.py` if Streamlit is in your system PATH).*

Once launched, your default web browser will automatically open to `http://localhost:8501`.

---

## 🔑 How to Get a Free YouTube Data API v3 Key (Optional)

CleanSpeak provides a full **Demo Mode** with pre-packaged realistic datasets, but if you want to analyze live YouTube videos:

1. Visit the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g., `CleanSpeak-Analytics`).
3. In the sidebar, navigate to **APIs & Services** > **Library**.
4. Search for **YouTube Data API v3** and click **Enable**.
5. Go to **APIs & Services** > **Credentials**.
6. Click **Create Credentials** > **API Key**.
7. Copy your generated API key and paste it into the CleanSpeak sidebar.
*(Google provides 10,000 free quota units per day).*

---

## 📊 Technologies Used

- **Frontend / Dashboard**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Interactive Visualizations**: [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/)
- **Natural Language Processing**: [TextBlob](https://textblob.readthedocs.io/), [WordCloud](https://github.com/amueller/word_cloud)
- **Emoji Analytics**: [emoji](https://pypi.org/project/emoji/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (`RandomForestRegressor`)
- **API Integration**: [google-api-python-client](https://github.com/googleapis/google-api-python-client)
