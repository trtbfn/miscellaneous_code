import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import re
from pathlib import Path

def preprocess_text(text):
    """A simple text preprocessing function."""
    if pd.isna(text) or text == "": return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Keep Cyrillic, Latin, and basic punctuation
    text = re.sub(r'[^а-яёa-z\s.,!?-]', '', text)
    return text

def load_and_prepare_data(platform="youtube", min_length=15, sample_size=None):
    """Load, preprocess, and prepare comment data."""
    print(f"\n--- Loading and Preparing Data for: {platform.upper()} ---")
    file_path = f"data/formatted_all_{platform}_comments_with_language_ru.csv"
    
    try:
        df = pd.read_csv(file_path)
    except (pd.errors.ParserError, ValueError):
        df = pd.read_csv(file_path, engine='python')

    df['Processed_Comment'] = df['Comment'].apply(preprocess_text)
    df = df[df['Processed_Comment'].str.len() >= min_length].copy()
    df['Published_At'] = pd.to_datetime(df['Published At'])
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Loaded and prepared {len(df)} comments for {platform.upper()}.")
    return df.sort_values('Published_At')

def run_bertopic_analysis(platform, base_output_dir):
    """Run the full BERTopic analysis for a given platform."""
    print(f"\n{'='*20} Starting BERTopic Analysis for {platform.upper()} {'='*20}")
    
    # 1. Load Data
    # For a quicker demonstration, let's use a sample. Remove sample_size for full run.
    df = load_and_prepare_data(platform=platform, sample_size=40000)
    docs = df['Processed_Comment'].tolist()
    timestamps = df['Published_At'].tolist()

    # 2. Setup BERTopic Model
    # Using a multilingual model for embeddings
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Remove Russian stopwords
    nltk.download('stopwords')
    russian_stopwords = list(stopwords.words('russian'))
    vectorizer_model = CountVectorizer(stop_words=russian_stopwords)
    
    # Configure the model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",
        verbose=True
    )

    # 3. Train the Model and Fit Transform
    print("\n--- Training BERTopic Model ---")
    topics, _ = topic_model.fit_transform(docs)

    # 4. Generate Topics Over Time
    print("\n--- Generating Topics Over Time ---")
    try:
        # This function requires timestamps to work
        topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
        
        # Create visualization
        fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
        
        # Save visualization
        platform_dir = base_output_dir / platform
        platform_dir.mkdir(exist_ok=True, parents=True)
        plot_path = platform_dir / f"{platform}_topics_over_time.html"
        fig.write_html(plot_path)
        print(f"Saved topics-over-time plot to: {plot_path}")

    except Exception as e:
        print(f"Could not generate topics over time. Error: {e}")

    # 5. Save the Model and Topic Info
    model_path = base_output_dir / platform / f"{platform}_bertopic_model"
    topic_model.save(model_path)
    print(f"Saved BERTopic model to: {model_path}")
    
    topic_info = topic_model.get_topic_info()
    topic_info_path = base_output_dir / platform / f"{platform}_topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False)
    print(f"Saved topic info to: {topic_info_path}")

    print(f"\n✅ Analysis for {platform.upper()} COMPLETE.")

def main():
    """Main execution function."""
    print("--- Starting BERTopic Analysis Pipeline ---")
    base_output_dir = Path("analysis_thailand/bertopic_results")
    base_output_dir.mkdir(exist_ok=True, parents=True)
    
    platforms = ["youtube", "telegram"]
    for platform in platforms:
        run_bertopic_analysis(platform, base_output_dir)

if __name__ == "__main__":
    main() 