#!/usr/bin/env python3
"""
Comprehensive Dynamic Topic Modeling with BERTopic and DTM Metrics

This script performs the following steps for YouTube and Telegram datasets:
1. Loads and preprocesses Russian-language comments for each platform.
2. Performs dynamic topic modeling using BERTopic with a Russian-English model.
3. Integrates the SimplifiedDTMEvaluator to calculate a suite of metrics
   based on the "Evaluating Dynamic Topic Models" paper (TTC, TTS, TTQ, DTQ, etc.).
4. Combines the BERTopic results (topics, words, frequencies over time) with the
   calculated DTM metrics into a single, structured CSV file for each dataset.
5. Saves all artifacts, including the final CSV, evaluation reports, and
   visualizations, into a dedicated output directory for each platform.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import warnings
import itertools
import time

# BERTopic and related imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

warnings.filterwarnings('ignore')
print("All libraries loaded.")


# --- DTM Evaluator Class (Adapted from dtm_metrics_simple.py) ---

class SimplifiedDTMEvaluator:
    """
    Simplified implementation of Dynamic Topic Quality metrics, adapted to take
    DataFrames as input instead of file paths.
    """
    
    def __init__(self, df_bertopic, df_topics_over_time, results_dir, top_n_topics=30, n_words=10):
        """
        Initialize DTM evaluator.
        
        Args:
            df_bertopic: DataFrame with BERTopic results (documents, topics, timestamps).
            df_topics_over_time: DataFrame with topics over time from BERTopic.
            results_dir: Path to save evaluation results.
            top_n_topics: Number of top topics to evaluate.
            n_words: Number of words per topic for evaluation.
        """
        self.top_n_topics = top_n_topics
        self.n_words = n_words
        
        # Set up directories
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"=== Simplified DTM Evaluation Initialized ===")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Load BERTopic results from DataFrames
        self.df = df_bertopic
        self.topics_over_time = df_topics_over_time
        
        print(f"Loaded {len(self.df)} documents for evaluation.")
        
        # Process data
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for DTM evaluation."""
        print("\n=== Preparing Data for DTM Evaluation ===")
        
        valid_topics = self.df[self.df['Topic'] != -1].copy()
        if valid_topics.empty:
            raise ValueError("No valid topics found in the input data (all are outliers).")
            
        topic_counts = valid_topics['Topic'].value_counts()
        self.selected_topics = topic_counts.head(self.top_n_topics).index.tolist()
        
        print(f"Selected top {len(self.selected_topics)} topics: {self.selected_topics}")
        
        self.filtered_df = valid_topics[valid_topics['Topic'].isin(self.selected_topics)].copy()
        self.filtered_df['Published_At'] = pd.to_datetime(self.filtered_df['Published_At'])
        self.filtered_df = self.filtered_df.sort_values('Published_At')
        
        # Convert timestamp column to datetime objects to handle quarters
        self.topics_over_time['Timestamp'] = pd.to_datetime(self.topics_over_time['Timestamp'])
        
        unique_timestamps = sorted(self.topics_over_time['Timestamp'].unique())
        self.num_time_slices = len(unique_timestamps)
        self.timestamps = [pd.Timestamp(ts) for ts in unique_timestamps]
        self.timestamp_map = {ts: i for i, ts in enumerate(self.timestamps)}
        
        print(f"Number of time slices: {self.num_time_slices}")
        
        self.prepare_topic_words_by_time()
        self.prepare_corpus()
        
        print("Data preparation completed.")
    
    def prepare_topic_words_by_time(self):
        """Prepare topic words organized by time and topic."""
        print("Preparing topic words by time...")
        self.topic_words_by_time = {}
        
        for t_idx, timestamp in enumerate(self.timestamps):
            self.topic_words_by_time[t_idx] = {}
            for topic_id in self.selected_topics:
                mask = (self.topics_over_time['Timestamp'] == timestamp) & \
                       (self.topics_over_time['Topic'] == topic_id)
                topic_data = self.topics_over_time[mask]
                
                if not topic_data.empty:
                    words = topic_data['Words'].iloc[0]
                    if isinstance(words, str):
                        word_list = [word.strip() for word in words.split(',') if word.strip()]
                        self.topic_words_by_time[t_idx][topic_id] = word_list[:self.n_words]
                    else:
                        self.topic_words_by_time[t_idx][topic_id] = []
                else:
                    self.topic_words_by_time[t_idx][topic_id] = []
        print(f"Prepared topic words for {len(self.timestamps)} time slices.")

    def prepare_corpus(self):
        """Prepare corpus for coherence calculations."""
        print("Preparing reference corpus...")
        self.corpus_texts = self.filtered_df['Processed_Comment'].tolist()
        self.tokenized_texts = [text.split() for text in self.corpus_texts]
        
        vectorizer = CountVectorizer(max_features=5000)
        vectorizer.fit_transform(self.corpus_texts)
        self.vocabulary = vectorizer.get_feature_names_out().tolist()
        
        print(f"Corpus size: {len(self.corpus_texts)} documents, Vocabulary size: {len(self.vocabulary)}")

    def _calculate_pmi_coherence(self, word_pairs):
        """Helper to calculate PMI-based coherence for a list of word pairs."""
        coherence_sum = 0.0
        pair_count = 0
        epsilon = 1e-10
        
        doc_freq = {word: set() for word in self.vocabulary}
        for i, text in enumerate(self.tokenized_texts):
            for word in set(text):
                if word in doc_freq:
                    doc_freq[word].add(i)

        for word_i, word_j in word_pairs:
            if word_i in self.vocabulary and word_j in self.vocabulary:
                docs_i = doc_freq[word_i]
                docs_j = doc_freq[word_j]
                count_ij = len(docs_i.intersection(docs_j))
                
                if count_ij > 0:
                    prob_i = len(docs_i) / len(self.tokenized_texts)
                    prob_j = len(docs_j) / len(self.tokenized_texts)
                    prob_ij = count_ij / len(self.tokenized_texts)
                    
                    coherence = np.log((prob_ij + epsilon) / (prob_i * prob_j + epsilon))
                    coherence_sum += coherence
                    pair_count += 1
                    
        return coherence_sum / pair_count if pair_count > 0 else 0.0

    def calculate_temporal_topic_coherence(self, topic_k, time_t):
        """Calculate Temporal Topic Coherence (TTC)."""
        if time_t + 1 >= self.num_time_slices: return 0.0
        
        words_t = self.topic_words_by_time[time_t].get(topic_k, [])
        words_t_plus = self.topic_words_by_time[time_t + 1].get(topic_k, [])
        
        if not words_t or not words_t_plus: return 0.0
        
        word_pairs = list(itertools.product(words_t, words_t_plus))
        return self._calculate_pmi_coherence(word_pairs)

    def calculate_temporal_topic_smoothness(self, topic_k, time_t):
        """Calculate Temporal Topic Smoothness (TTS)."""
        if time_t + 1 >= self.num_time_slices: return 0.0
        
        words_i = set(self.topic_words_by_time[time_t].get(topic_k, []))
        words_j = set(self.topic_words_by_time[time_t + 1].get(topic_k, []))
        
        if not words_i or not words_j: return 0.0
        
        intersection = len(words_i & words_j)
        union = len(words_i | words_j)
        return intersection / union if union > 0 else 0.0

    def calculate_topic_coherence(self, topic_words):
        """Calculate standard topic coherence (TC)."""
        if not topic_words or len(topic_words) < 2: return 0.0
        word_pairs = list(itertools.combinations(topic_words, 2))
        return self._calculate_pmi_coherence(word_pairs)

    def calculate_topic_diversity(self, all_topic_words):
        """Calculate topic diversity (TD)."""
        if not all_topic_words: return 0.0
        all_words = [word for sublist in all_topic_words for word in sublist]
        if not all_words: return 0.0
        return len(set(all_words)) / len(all_words)

    def compute_all_metrics(self):
        """Compute all DTM metrics."""
        print("\n=== Computing All DTM Metrics ===")
        results = {
            'ttc_per_topic_per_time': {}, 'tts_per_topic_per_time': {},
            'ttq_per_topic': {}, 'tc_per_time': {}, 'td_per_time': {},
            'tq_per_time': {}, 'overall_metrics': {}
        }

        # 1. TTC and TTS for each topic across time
        for topic_k in self.selected_topics:
            results['ttc_per_topic_per_time'][topic_k] = {}
            results['tts_per_topic_per_time'][topic_k] = {}
            ttc_scores, tts_scores = [], []
            for time_t in range(self.num_time_slices - 1):
                ttc = self.calculate_temporal_topic_coherence(topic_k, time_t)
                tts = self.calculate_temporal_topic_smoothness(topic_k, time_t)
                results['ttc_per_topic_per_time'][topic_k][time_t] = ttc
                results['tts_per_topic_per_time'][topic_k][time_t] = tts
                ttc_scores.append(ttc)
                tts_scores.append(tts)
            ttq_scores = [c * s for c, s in zip(ttc_scores, tts_scores)]
            results['ttq_per_topic'][topic_k] = np.mean(ttq_scores) if ttq_scores else 0.0
        
        # 2. TC, TD, TQ for each time slice
        for time_t in range(self.num_time_slices):
            all_topic_words_t, tc_scores_t = [], []
            for topic_k in self.selected_topics:
                topic_words = self.topic_words_by_time[time_t].get(topic_k, [])
                if topic_words:
                    all_topic_words_t.append(topic_words)
                    tc_scores_t.append(self.calculate_topic_coherence(topic_words))
            results['tc_per_time'][time_t] = np.mean(tc_scores_t) if tc_scores_t else 0.0
            results['td_per_time'][time_t] = self.calculate_topic_diversity(all_topic_words_t)
            results['tq_per_time'][time_t] = results['tc_per_time'][time_t] * results['td_per_time'][time_t]
            
        # 3. Overall metrics
        all_ttc = [s for k in self.selected_topics for s in results['ttc_per_topic_per_time'][k].values()]
        all_tts = [s for k in self.selected_topics for s in results['tts_per_topic_per_time'][k].values()]
        results['overall_metrics']['TTC'] = np.mean(all_ttc) if all_ttc else 0.0
        results['overall_metrics']['TTS'] = np.mean(all_tts) if all_tts else 0.0
        results['overall_metrics']['TTQ'] = np.mean(list(results['ttq_per_topic'].values()))
        results['overall_metrics']['TC'] = np.mean(list(results['tc_per_time'].values()))
        results['overall_metrics']['TD'] = np.mean(list(results['td_per_time'].values()))
        results['overall_metrics']['TQ'] = np.mean(list(results['tq_per_time'].values()))
        results['overall_metrics']['DTQ'] = 0.5 * (results['overall_metrics']['TQ'] + results['overall_metrics']['TTQ'])
        
        print("DTM metrics computation completed.")
        return results

    def save_results(self, results):
        """Save all evaluation results to files."""
        print(f"\n=== Saving Evaluation Results to {self.results_dir} ===")
        
        # Detailed JSON
        with open(self.results_dir / 'dtm_detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # Summary CSV
        summary_df = pd.DataFrame([{'Metric': k, 'Score': v} for k, v in results['overall_metrics'].items()])
        summary_df.to_csv(self.results_dir / 'dtm_metrics_summary.csv', index=False)
        print("Results saved.")

    def run_evaluation(self):
        """Run complete DTM evaluation."""
        start_time = time.time()
        print("🚀 Starting Simplified DTM Evaluation")
        try:
            results = self.compute_all_metrics()
            self.save_results(results)
            elapsed_time = time.time() - start_time
            print(f"\n🎯 DTM Evaluation Complete in {elapsed_time:.2f} seconds!")
            print(f"  🏆 DTQ (Dynamic Topic Quality): {results['overall_metrics']['DTQ']:.4f}")
            return results
        except Exception as e:
            print(f"❌ Error in DTM evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None


# --- Main Analysis Functions ---

def preprocess_text(text):
    """Simple text preprocessing for Russian comments."""
    if pd.isna(text) or text == "": return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^а-яёa-z\s]', '', text) # Keep only letters and spaces
    return text

def load_and_prepare_data(platform="youtube", min_length=15, sample_size=20000):
    """Load and prepare comment data for a specific platform."""
    start_time = time.time()
    print(f"\n--- Loading and Preparing Data for: {platform.upper()} ---")
    
    file_map = {
        "youtube": "data/formatted_all_youtube_comments_with_language_ru.csv",
        "telegram": "data/formatted_telegram_comments_with_language_ru.csv"
    }
    
    if platform not in file_map:
        raise ValueError(f"Platform must be one of {list(file_map.keys())}")
        
    df = pd.read_csv(file_map[platform])
    print(f"Loaded {len(df)} raw comments.")
    
    df = df[df['Language'] == 'ru'].copy()
    print(f"Filtered to {len(df)} Russian comments.")
    
    df['Processed_Comment'] = df['Comment'].apply(preprocess_text)
    df = df[df['Processed_Comment'].str.len() >= min_length]
    print(f"Found {len(df)} comments after length filtering (>={min_length}).")
    
    df['Published_At'] = pd.to_datetime(df['Published At'])
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {len(df)} comments.")
        
    elapsed_time = time.time() - start_time
    print(f"--- Data Loading and Preparation finished in {elapsed_time:.2f} seconds ---")
    return df.sort_values('Published_At')

def run_analysis_for_platform(platform: str, base_output_dir: Path):
    """Run the full DTM analysis and evaluation pipeline for one platform."""
    
    total_start_time = time.time()
    print(f"\n{'='*20} Starting Analysis for {platform.upper()} {'='*20}")
    
    # 1. Setup directories
    platform_dir = base_output_dir / platform
    platform_dir.mkdir(exist_ok=True, parents=True)
    
    # 2. Load Data
    try:
        df = load_and_prepare_data(platform=platform, sample_size=None) # Use all data
    except Exception as e:
        print(f"Could not load data for {platform}: {e}")
        return

    # 3. BERTopic Modeling
    stage_start_time = time.time()
    print("\n--- Running BERTopic ---")
    embedding_model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
    vectorizer_model = CountVectorizer(stop_words=None, ngram_range=(1, 2))
    hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        hdbscan_model=hdbscan_model,
        verbose=True
    )
    
    texts = df['Processed_Comment'].tolist()
    timestamps = df['Published_At'].tolist()
    
    # Add "clustering: " prefix for potentially better performance
    prefixed_texts = ["clustering: " + text for text in texts]
    
    topics, probs = topic_model.fit_transform(prefixed_texts)
    df['Topic'] = topics
    df['Probability'] = probs
    print(f"--- BERTopic fitting finished in {time.time() - stage_start_time:.2f} seconds ---")
    
    print("\n--- Analyzing Topics Over Time (Grouped by Quarter) ---")
    stage_start_time = time.time()
    # Convert timestamps to quarters for dynamic topic modeling
    timestamps_for_dtm = df['Published_At'].dt.to_period('Q').astype(str).tolist()
    topics_over_time = topic_model.topics_over_time(prefixed_texts, timestamps_for_dtm)
    print(f"--- Topics Over Time analysis finished in {time.time() - stage_start_time:.2f} seconds ---")
    
    # Save model and basic results
    topic_model.save(platform_dir / f"{platform}_bertopic_model", serialization="pickle")
    df.to_csv(platform_dir / f"{platform}_bertopic_docs.csv", index=False)
    topics_over_time.to_csv(platform_dir / f"{platform}_topics_over_time_raw.csv", index=False)

    # 4. DTM Evaluation
    print("\n--- Running DTM Evaluation Metrics ---")
    evaluator = SimplifiedDTMEvaluator(
        df_bertopic=df,
        df_topics_over_time=topics_over_time,
        results_dir=platform_dir / "evaluation_metrics",
        top_n_topics=30
    )
    metrics_results = evaluator.run_evaluation()
    
    if not metrics_results:
        print("Skipping final report generation due to evaluation error.")
        return

    # 5. Combine and Format Final Report
    stage_start_time = time.time()
    print("\n--- Generating Final Combined Metrics Report ---")
    
    # Start with the topics_over_time dataframe
    final_df = topics_over_time.copy()
    final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'])
    
    # Get metrics from the evaluator's results
    dtm_res = metrics_results
    
    # Map time slice index to timestamp for merging
    ts_map = evaluator.timestamp_map
    
    # Add per-time metrics (TC, TD, TQ)
    final_df['TC_timeslice'] = final_df['Timestamp'].map(ts_map).map(dtm_res['tc_per_time'])
    final_df['TD_timeslice'] = final_df['Timestamp'].map(ts_map).map(dtm_res['td_per_time'])
    final_df['TQ_timeslice'] = final_df['Timestamp'].map(ts_map).map(dtm_res['tq_per_time'])
    
    # Add per-topic-average metric (TTQ)
    final_df['TTQ_topic_avg'] = final_df['Topic'].map(dtm_res['ttq_per_topic'])
    
    # Add per-topic-per-time metrics (TTC, TTS)
    def get_per_topic_time_metric(row, metric_dict):
        ts_idx = ts_map.get(row['Timestamp'])
        topic_id = row['Topic']
        if ts_idx is not None and topic_id in metric_dict and ts_idx in metric_dict[topic_id]:
            return metric_dict[topic_id][ts_idx]
        return None
        
    final_df['TTC_t'] = final_df.apply(lambda row: get_per_topic_time_metric(row, dtm_res['ttc_per_topic_per_time']), axis=1)
    final_df['TTS_t'] = final_df.apply(lambda row: get_per_topic_time_metric(row, dtm_res['tts_per_topic_per_time']), axis=1)

    # Add overall metrics (repeated for each row)
    for metric, score in dtm_res['overall_metrics'].items():
        final_df[f"Overall_{metric}"] = score
        
    # Reorder and save final CSV
    column_order = [
        'Timestamp', 'Topic', 'Words', 'Frequency',
        'TTC_t', 'TTS_t', 'TTQ_topic_avg',
        'TC_timeslice', 'TD_timeslice', 'TQ_timeslice',
    ] + [f"Overall_{m}" for m in dtm_res['overall_metrics']]
    
    final_df = final_df.rename(columns={'Topic': 'Topic_ID'})
    
    # Handle potential missing columns if some metrics fail
    final_columns = [col for col in column_order if col in final_df.columns]
    
    final_df = final_df[final_columns]
    
    output_path = base_output_dir / f"{platform}_dynamic_topic_metrics.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"--- Final Report generation finished in {time.time() - stage_start_time:.2f} seconds ---")
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\n✅ Analysis for {platform.upper()} COMPLETE in {total_elapsed_time:.2f} seconds.")
    print(f"Final combined metrics report saved to: {output_path}")

def main():
    """Main execution function."""
    print("Starting Comprehensive Dynamic Topic Modeling Script")
    base_output_dir = Path("analysis_thailand/bert_overtime")
    base_output_dir.mkdir(exist_ok=True, parents=True)
    print(f"All results will be saved in: {base_output_dir}")
    
    platforms = ["youtube", "telegram"]
    
    for platform in platforms:
        try:
            run_analysis_for_platform(platform, base_output_dir)
        except Exception as e:
            print(f"\n{'!'*10} A critical error occurred during the analysis for {platform.upper()} {'!'*10}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nProceeding to the next platform if available...")

    print("\n\nAll platform analyses are complete.")

if __name__ == "__main__":
    main() 