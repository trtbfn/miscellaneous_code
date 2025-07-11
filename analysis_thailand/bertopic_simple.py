import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import json
from pathlib import Path

# BERTopic imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

print("Starting BERTopic Analysis with ru-en-RoSBERTa")
print("="*60)

# Create analysis directory
analysis_dir = Path("analysis_thailand")
analysis_dir.mkdir(exist_ok=True)

def preprocess_text(text):
    """Simple text preprocessing for Russian comments"""
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text).lower()
    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Keep only letters and basic punctuation
    text = re.sub(r'[^а-яёa-z\s\.,!?;:-]', '', text)
    
    return text

def load_data(max_samples=10000):
    """Load and prepare comment data"""
    print(f"Loading data (max {max_samples} samples)...")
    
    # Load the language-classified files
    try:
        youtube_df = pd.read_csv("data/formatted_all_youtube_comments_with_language.csv")
        telegram_df = pd.read_csv("data/formatted_telegram_comments_with_language.csv")
        print(f"Loaded {len(youtube_df)} YouTube and {len(telegram_df)} Telegram comments")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Add platform info
    youtube_df['Platform'] = 'YouTube'
    telegram_df['Platform'] = 'Telegram'
    
    # Combine and filter for Russian comments
    combined_df = pd.concat([youtube_df, telegram_df], ignore_index=True)
    russian_df = combined_df[combined_df['Language'] == 'ru'].copy()
    print(f"Found {len(russian_df)} Russian comments")
    
    # Preprocess comments
    russian_df['Processed_Comment'] = russian_df['Comment'].apply(preprocess_text)
    
    # Filter by length (at least 10 characters)
    russian_df = russian_df[russian_df['Processed_Comment'].str.len() >= 10]
    print(f"After filtering: {len(russian_df)} comments")
    
    # Convert timestamps
    russian_df['Published_At'] = pd.to_datetime(russian_df['Published At'])
    
    # Sample if too many comments
    if len(russian_df) > max_samples:
        russian_df = russian_df.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} comments for analysis")
    
    return russian_df

def create_topic_model():
    """Create and configure BERTopic model"""
    print("Setting up BERTopic model...")
    
    # Initialize embedding model
    print("Loading ru-en-RoSBERTa model...")
    try:
        embedding_model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        # Fallback to a simpler model
        print("Using fallback model: all-MiniLM-L6-v2")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Configure dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # Configure clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Configure vectorizer for Russian text
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        min_df=2,
        max_df=0.9
    )
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True
    )
    
    return topic_model

def analyze_topics(topic_model, df):
    """Perform topic analysis"""
    print("Performing topic analysis...")
    
    # Prepare texts with clustering prefix
    texts = ["clustering: " + text for text in df['Processed_Comment'].tolist()]
    
    # Fit the model
    print("Fitting BERTopic model...")
    try:
        topics, probabilities = topic_model.fit_transform(texts)
        print("Model fitted successfully!")
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None, None
    
    # Add results to dataframe
    df['Topic'] = topics
    df['Topic_Probability'] = probabilities
    
    print(f"Found {len(set(topics))} topics")
    print(f"Outliers (topic -1): {sum(1 for t in topics if t == -1)}")
    
    return df, topic_model

def create_temporal_analysis(topic_model, df):
    """Create topics over time analysis"""
    print("Creating temporal analysis...")
    
    try:
        # Prepare data for temporal analysis
        texts = ["clustering: " + text for text in df['Processed_Comment'].tolist()]
        timestamps = df['Published_At'].tolist()
        
        # Calculate topics over time
        topics_over_time = topic_model.topics_over_time(
            texts, 
            timestamps, 
            nr_bins=15,
            global_tuning=True,
            evolution_tuning=True
        )
        
        print("Temporal analysis completed!")
        return topics_over_time
    
    except Exception as e:
        print(f"Error in temporal analysis: {e}")
        return None

def save_results(df, topic_model, topics_over_time=None):
    """Save analysis results"""
    print("Saving results...")
    
    # Save main data
    df.to_csv(analysis_dir / "bertopic_results.csv", index=False)
    
    # Save topic information
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(analysis_dir / "topic_info.csv", index=False)
    
    # Save topic words
    topics_dict = {}
    for topic_id in topic_info['Topic']:
        if topic_id != -1:
            topics_dict[topic_id] = topic_model.get_topic(topic_id)
    
    with open(analysis_dir / "topic_words.json", 'w', encoding='utf-8') as f:
        json.dump(topics_dict, f, ensure_ascii=False, indent=2)
    
    # Save topics over time if available
    if topics_over_time is not None:
        topics_over_time.to_csv(analysis_dir / "topics_over_time.csv", index=False)
    
    # Save model
    try:
        topic_model.save(str(analysis_dir / "bertopic_model"), serialization="pickle")
        print("Model saved successfully!")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    print(f"Results saved to {analysis_dir}")

def create_visualizations(df, topic_model, topics_over_time=None):
    """Create basic visualizations"""
    print("Creating visualizations...")
    
    # Topic distribution
    plt.figure(figsize=(15, 8))
    topic_counts = df['Topic'].value_counts().head(20)
    topic_counts.plot(kind='bar')
    plt.title('Top 20 Topics by Number of Comments')
    plt.xlabel('Topic ID')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(analysis_dir / "topic_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Platform comparison if available
    if 'Platform' in df.columns:
        plt.figure(figsize=(12, 6))
        platform_counts = df.groupby('Platform')['Topic'].nunique()
        platform_counts.plot(kind='bar')
        plt.title('Number of Topics by Platform')
        plt.ylabel('Number of Topics')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(analysis_dir / "platform_topics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Interactive visualizations
    try:
        # Topic similarity heatmap
        fig = topic_model.visualize_heatmap(top_n_topics=20)
        fig.write_html(str(analysis_dir / "topic_similarity.html"))
        
        # Intertopic distance map
        fig = topic_model.visualize_topics()
        fig.write_html(str(analysis_dir / "intertopic_distance.html"))
        
        # Topics over time if available
        if topics_over_time is not None:
            fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
            fig.write_html(str(analysis_dir / "topics_over_time.html"))
        
        print("Interactive visualizations created!")
    except Exception as e:
        print(f"Error creating interactive visualizations: {e}")

def generate_report(df, topic_model):
    """Generate analysis report"""
    print("Generating report...")
    
    report = []
    report.append("BERTopic Analysis Report")
    report.append("="*50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model: ai-forever/ru-en-RoSBERTa")
    report.append("")
    
    # Data summary
    report.append("DATA SUMMARY")
    report.append(f"Total comments: {len(df):,}")
    report.append(f"Language: Russian")
    report.append(f"Date range: {df['Published_At'].min()} to {df['Published_At'].max()}")
    
    if 'Platform' in df.columns:
        platform_counts = df['Platform'].value_counts()
        report.append("Platform distribution:")
        for platform, count in platform_counts.items():
            report.append(f"  {platform}: {count:,}")
    
    report.append("")
    
    # Topic summary
    topic_info = topic_model.get_topic_info()
    report.append("TOPIC ANALYSIS")
    report.append(f"Total topics: {len(topic_info) - 1}")  # Exclude outliers
    report.append(f"Outliers: {len(df[df['Topic'] == -1]):,}")
    report.append("")
    
    # Top topics
    report.append("TOP 10 TOPICS:")
    top_topics = topic_info[topic_info['Topic'] != -1].head(10)
    for _, row in top_topics.iterrows():
        topic_id = row['Topic']
        topic_size = row['Count']
        topic_words = topic_model.get_topic(topic_id)
        top_words = [word for word, _ in topic_words[:5]]
        report.append(f"Topic {topic_id}: {topic_size:,} comments")
        report.append(f"  Keywords: {', '.join(top_words)}")
        report.append("")
    
    # Save report
    with open(analysis_dir / "analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("Report generated!")

def main():
    """Main analysis function"""
    
    # Load data
    df = load_data(max_samples=15000)  # Limit for faster processing
    if df is None:
        return
    
    # Create topic model
    topic_model = create_topic_model()
    
    # Perform analysis
    df, topic_model = analyze_topics(topic_model, df)
    if df is None:
        return
    
    # Temporal analysis
    topics_over_time = create_temporal_analysis(topic_model, df)
    
    # Save results
    save_results(df, topic_model, topics_over_time)
    
    # Create visualizations
    create_visualizations(df, topic_model, topics_over_time)
    
    # Generate report
    generate_report(df, topic_model)
    
    print("\nAnalysis completed successfully!")
    print(f"Check the '{analysis_dir}' directory for results")

if __name__ == "__main__":
    main() 