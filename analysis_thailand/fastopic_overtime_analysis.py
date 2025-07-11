#!/usr/bin/env python3
"""
Topic Modeling over Time with FASTopic and Evaluation Metrics.

This script performs the following steps for YouTube and Telegram datasets:
1. Loads and preprocesses Russian-language comments for each platform.
2. Performs topic modeling using FASTopic with enhanced Russian preprocessing.
3. Calculates Topic Coherence and Diversity to evaluate model quality.
4. Calculates and visualizes topic activity over time (by quarter).
5. Saves all artifacts, including the model, top words, metrics, and visualizations,
   into a dedicated output directory for each platform.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import pickle
import time
import warnings
import itertools

# FASTopic and related imports
from fastopic import FASTopic
from topmost.preprocess import Preprocess
from sklearn.feature_extraction.text import CountVectorizer
from razdel import tokenize as razdel_tokenize

# Use Agg backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
print("All libraries loaded.")


# --- Text Preprocessing and Data Loading ---

def russian_tokenizer(text):
    """Custom tokenizer for Russian text using razdel."""
    return [token.text for token in razdel_tokenize(text)]

def load_and_prepare_data(platform="youtube", min_length=15, sample_size=None, chunk_size=50000):
    """Load and prepare comment data using chunking to handle large files."""
    start_time = time.time()
    print(f"\n--- Loading and Preparing Data for: {platform.upper()} ---")
    
    file_map = {
        "youtube": "data/formatted_all_youtube_comments_with_language_ru.csv",
        "telegram": "data/formatted_telegram_comments_with_language_ru.csv"
    }
    
    if platform not in file_map:
        raise ValueError(f"Platform must be one of {list(file_map.keys())}")
        
    processed_chunks = []
    total_raw_comments = 0
    
    # Use chunking to process large files more efficiently
    with pd.read_csv(file_map[platform], chunksize=chunk_size, lineterminator='\n') as reader:
        for i, chunk in enumerate(reader):
            total_raw_comments += len(chunk)
            print(f"Processing chunk {i+1}...")
            
            chunk = chunk[chunk['Language'] == 'ru'].copy()
            if chunk.empty:
                continue

            # Vectorized text preprocessing for performance
            comments = chunk['Comment'].fillna('').astype(str).str.lower()
            comments = comments.str.replace(r'http\S+|www\S+|@\w+', '', regex=True)
            comments = comments.str.replace(r'[^а-яёa-z\s]', '', regex=True)
            comments = comments.str.replace(r'\s+', ' ', regex=True).str.strip()
            chunk['Processed_Comment'] = comments
            
            chunk = chunk[chunk['Processed_Comment'].str.len() >= min_length]
            
            if not chunk.empty:
                processed_chunks.append(chunk)

    print(f"Loaded {total_raw_comments} raw comments in total.")

    if not processed_chunks:
        print("No data found after processing all chunks.")
        return pd.DataFrame()
        
    df = pd.concat(processed_chunks, ignore_index=True)
    print(f"Filtered to {len(df)} comments after processing all chunks.")
    
    df['Published_At'] = pd.to_datetime(df['Published At'])
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {len(df)} comments.")
        
    elapsed_time = time.time() - start_time
    print(f"--- Data Loading and Preparation finished in {elapsed_time:.2f} seconds ---")
    return df.sort_values('Published_At')


# --- Metrics Calculation ---

def calculate_topic_diversity(top_words):
    """Calculate topic diversity (fraction of unique words across all topics)."""
    if not top_words: return 0.0
    all_words = [word for sublist in top_words for word in sublist]
    if not all_words: return 0.0
    return len(set(all_words)) / len(all_words)

def calculate_topic_coherence(top_words, docs, coherence_type='c_v'):
    """
    Calculate average topic coherence for all topics.
    NOTE: This is a simplified coherence calculation. For more robust results,
    consider using gensim's CoherenceModel if the corpus is large.
    """
    print(f"Calculating Topic Coherence ({coherence_type})...")
    vectorizer = CountVectorizer(tokenizer=russian_tokenizer)
    try:
        corpus_bow = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        word2id = {w: i for i, w in enumerate(vocab)}
    except ValueError:
        print("Warning: Vocabulary is empty. Skipping coherence calculation.")
        return 0.0

    topic_coherences = []
    for topic_idx, words in enumerate(top_words):
        topic_words = [word for word in words if word in word2id]
        if len(topic_words) < 2:
            continue

        word_indices = [word2id[w] for w in topic_words]
        
        # Co-occurrence counting
        co_occurrence_count = 0
        for doc_bow in corpus_bow:
            doc_words = [idx for idx, count in zip(doc_bow.indices, doc_bow.data) if count > 0]
            if all(w_idx in doc_words for w_idx in word_indices):
                co_occurrence_count += 1
        
        # Simplified NPMI-like coherence (using co-occurrence)
        # This is a basic form and not a full NPMI or C_v
        epsilon = 1e-10
        doc_freq = corpus_bow[:, word_indices].sum(axis=0) / corpus_bow.shape[0]
        
        if co_occurrence_count > 0 and np.all(doc_freq > 0):
            prob_words = np.prod(doc_freq)
            prob_co_occurrence = co_occurrence_count / corpus_bow.shape[0]
            coherence = np.log((prob_co_occurrence + epsilon) / (prob_words + epsilon))
            topic_coherences.append(coherence)
        else:
            topic_coherences.append(0.0)

    avg_coherence = np.mean(topic_coherences) if topic_coherences else 0.0
    print(f"Average Topic Coherence: {avg_coherence:.4f}")
    return avg_coherence


# --- Main Analysis Function ---

def run_analysis_for_platform(platform: str, base_output_dir: Path, num_topics=50):
    """Run the full FASTopic analysis pipeline for one platform."""
    
    total_start_time = time.time()
    print(f"\n{'='*20} Starting Analysis for {platform.upper()} {'='*20}")
    
    # 1. Setup directories
    platform_dir = base_output_dir / platform
    platform_dir.mkdir(exist_ok=True, parents=True)
    
    # 2. Load Data
    try:
        df = load_and_prepare_data(platform=platform, sample_size=None)
        if df.empty:
            print(f"No data to process for {platform}. Skipping.")
            return
    except Exception as e:
        print(f"Could not load data for {platform}: {e}")
        return

    docs = df['Processed_Comment'].tolist()

    # 3. FASTopic Modeling
    stage_start_time = time.time()
    print("\n--- Running FASTopic ---")
    
    # Preprocessing for FASTopic with Russian support
    preprocess = Preprocess(
        stopwords=None, 
        vocab_size=32000,
        tokenizer=russian_tokenizer
    )
    
    # Create and fit the model
    model = FASTopic(num_topics, preprocess=preprocess)
    print(f"Fitting FASTopic model with {num_topics} topics...")
    top_words, doc_topic_dist = model.fit_transform(docs)
    print(f"--- FASTopic fitting finished in {time.time() - stage_start_time:.2f} seconds ---")

    # Save top words
    top_words_df = pd.DataFrame(top_words)
    top_words_path = platform_dir / f"{platform}_top_words.csv"
    top_words_df.to_csv(top_words_path, index=False)
    print(f"Top words saved to: {top_words_path}")

    # Save the model
    model_path = platform_dir / f"{platform}_fastopic_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")

    # 4. Calculate and Save Metrics
    print("\n--- Calculating Evaluation Metrics ---")
    topic_diversity = calculate_topic_diversity(top_words)
    topic_coherence = calculate_topic_coherence(top_words, docs)
    
    metrics_df = pd.DataFrame([
        {'Metric': 'Topic Diversity', 'Score': topic_diversity},
        {'Metric': 'Topic Coherence', 'Score': topic_coherence}
    ])
    metrics_path = base_output_dir / f"{platform}_fastopic_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    # 5. Topic Activity Over Time
    stage_start_time = time.time()
    print("\n--- Analyzing and Visualizing Topics Over Time ---")
    
    # Prepare time slices for overtime analysis (by quarter)
    df['Time_Slice'] = df['Published_At'].dt.to_period('Q').astype(str)
    time_slices_per_doc = df['Time_Slice'].tolist()
    unique_time_slices = sorted(df['Time_Slice'].unique())
    
    try:
        # Calculate topic activity
        activity = model.topic_activity_over_time(time_slices_per_doc)
        
        # Save activity data
        activity_df = pd.DataFrame(activity, columns=unique_time_slices)
        activity_path = platform_dir / f"{platform}_topic_activity.csv"
        activity_df.to_csv(activity_path)
        print(f"Topic activity data saved to: {activity_path}")

        # Visualize topic activity
        fig = model.visualize_topic_activity(top_n=10, topic_activity=activity, time_slices=unique_time_slices)
        fig_path = platform_dir / f"{platform}_topic_activity.html"
        fig.write_html(fig_path)
        print(f"Topic activity visualization saved to: {fig_path}")

    except Exception as e:
        print(f"Could not analyze topics over time: {e}")

    print(f"--- Topic activity analysis finished in {time.time() - stage_start_time:.2f} seconds ---")
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\n✅ Analysis for {platform.upper()} COMPLETE in {total_elapsed_time:.2f} seconds.")
    print(f"Results saved in: {platform_dir}")


def main():
    """Main execution function."""
    print("Starting FASTopic Topic Modeling Script")
    base_output_dir = Path("analysis_thailand/fastopic_overtime")
    base_output_dir.mkdir(exist_ok=True, parents=True)
    print(f"All results will be saved in: {base_output_dir}")
    
    platforms = ["youtube", "telegram"]
    
    for platform in platforms:
        try:
            run_analysis_for_platform(platform, base_output_dir, num_topics=50)
        except Exception as e:
            print(f"\n{'!'*10} A critical error occurred during the analysis for {platform.upper()} {'!'*10}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nProceeding to the next platform if available...")

    print("\n\nAll platform analyses are complete.")

if __name__ == "__main__":
    main() 