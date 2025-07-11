#!/usr/bin/env python3
"""
Comprehensive pipeline for Dynamic Topic Modeling on Russian comments.
This script performs the following steps:
1.  Loads and prepares Russian comments from YouTube and Telegram.
2.  Trains a C-FTM (Correlated and a bit of a mouthful - Fast Topic Model) dynamic topic model.
3.  Evaluates the results using metrics from the 'Evaluating Dynamic Topic Models' paper.
4.  Saves all results, including visualizations and a detailed report.
"""
import pandas as pd
import numpy as np
import os
import re
import json
from datetime import datetime
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import topmost
from dtm_metrics_simple import SimplifiedDTMEvaluator
import itertools
import shutil
from topmost import Preprocess

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- NLTK for text processing ---
import nltk
from nltk.corpus import stopwords
try:
    stopwords.words('russian')
except OSError:
    print("Downloading Russian stopwords for NLTK...")
    nltk.download('stopwords')

# --- Machine Learning and Topic Modeling ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import topmost
from dtm_metrics_simple import SimplifiedDTMEvaluator

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# Get the directory of the current script to build absolute paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# File paths
YOUTUBE_COMMENTS_PATH = SCRIPT_DIR / '../data/formatted_all_youtube_comments_with_language_ru.csv'
TELEGRAM_COMMENTS_PATH = SCRIPT_DIR / '../data/formatted_telegram_comments_with_language_ru.csv'
DATA_DIR = SCRIPT_DIR / "russian_dtm_data"
RESULTS_DIR = SCRIPT_DIR / "russian_dtm_results"

# Model parameters
NUM_TOPICS = 20
N_WORDS_PER_TOPIC = 10
VOCAB_SIZE = 10000
EPOCHS = 50  # Reduced for faster execution, recommend 200+ for better results
BATCH_SIZE = 32 # Reduced to prevent memory issues with a large dataset

# Create directories
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==============================================================================
# --- Part 1: Data Preparation ---
# ==============================================================================

def preprocess_text(text, stop_words):
    """Clean and preprocess a single text document."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)  # Keep only Russian letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

def prepare_data():
    """Load, merge, clean, and format the data for the DTM model using a fully manual process."""
    print("\n--- 1. Preparing Data (Manual Process) ---")
    
    # --- Step 1: Initial data loading and cleaning ---
    try:
        df_youtube = pd.read_csv(YOUTUBE_COMMENTS_PATH)
        df_telegram = pd.read_csv(TELEGRAM_COMMENTS_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the input CSV files exist.")
        return None
    
    df = pd.concat([df_youtube, df_telegram], ignore_index=True)
    stop_words = stopwords.words('russian')
    df['Processed_Comment'] = df['Comment'].apply(lambda x: preprocess_text(x, stop_words))
    
    df['Published_At'] = pd.to_datetime(df['Published At'], errors='coerce')
    df.dropna(subset=['Published_At', 'Processed_Comment'], inplace=True)
    df = df[df['Processed_Comment'] != ""]
    df = df.sort_values('Published_At')
    
    df['Time_Slice_Str'] = df['Published_At'].dt.to_period('M').astype(str)
    slice_counts = df['Time_Slice_Str'].value_counts()
    valid_slices = slice_counts[slice_counts >= 10].index
    df_filtered = df[df['Time_Slice_Str'].isin(valid_slices)].copy()
    
    unique_times = sorted(df_filtered['Time_Slice_Str'].unique())
    time_map = {ts: i for i, ts in enumerate(unique_times)}
    df_filtered['Time_Slice'] = df_filtered['Time_Slice_Str'].map(time_map)
    
    # --- Step 2: Create Vocabulary and BoW from filtered data ---
    vectorizer = CountVectorizer(max_features=VOCAB_SIZE, min_df=5, max_df=0.9)
    vectorizer.fit(df_filtered['Processed_Comment'])
    vocab = vectorizer.get_feature_names_out()

    # Create final train/test split from the *filtered* and *vectorizable* data
    # This ensures no empty documents after vectorization
    df_final = df_filtered[df_filtered['Processed_Comment'].apply(lambda x: len(x.split()) > 0)]
    train_df, test_df = train_test_split(df_final, test_size=0.1, random_state=42, stratify=df_final['Time_Slice'])

    train_bow = vectorizer.transform(train_df['Processed_Comment'])
    test_bow = vectorizer.transform(test_df['Processed_Comment'])

    # --- Step 3: Save all required files ---
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir()

    # Save BoW and vocabulary
    scipy.sparse.save_npz(DATA_DIR / "train_bow.npz", train_bow)
    scipy.sparse.save_npz(DATA_DIR / "test_bow.npz", test_bow)
    with open(DATA_DIR / "vocab.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))

    # Save dummy embeddings
    dummy_embeddings = scipy.sparse.csr_matrix((len(vocab), 300), dtype=np.float32)
    scipy.sparse.save_npz(DATA_DIR / "word_embeddings.npz", dummy_embeddings)

    # Save text and time files
    train_df['Processed_Comment'].to_csv(DATA_DIR / "train_texts.txt", index=False, header=False)
    test_df['Processed_Comment'].to_csv(DATA_DIR / "test_texts.txt", index=False, header=False)
    train_df['Time_Slice'].to_csv(DATA_DIR / "train_times.txt", index=False, header=False)
    test_df['Time_Slice'].to_csv(DATA_DIR / "test_times.txt", index=False, header=False)

    # Save metadata for evaluation
    train_df.to_csv(DATA_DIR / "train_data.csv", index=False)
    with open(DATA_DIR / "time_map.json", 'w', encoding='utf-8') as f:
        json.dump(time_map, f)
        
    print(f"Data prepared and saved to {DATA_DIR}/")
    return str(DATA_DIR)


# ==============================================================================
# --- Part 2: DTM Model Training ---
# ==============================================================================

def train_dtm_model(data_dir):
    """Train the DTM model on the prepared data."""
    print("\n--- 2. Training DTM Model (using DETM) ---")
    
    # Load data using topmost's DynamicDataset
    dataset = topmost.DynamicDataset(data_dir, batch_size=BATCH_SIZE, device=DEVICE)
    
    # Initialize DETM model (does not require pre-trained embeddings)
    model = topmost.models.DETM(
        num_topics=NUM_TOPICS,
        vocab_size=dataset.vocab_size,
        num_times=dataset.num_times,
        train_size=dataset.train_size,
        train_time_wordfreq=dataset.train_time_wordfreq,
        device=DEVICE
    )
    model.to(DEVICE)
    
    # Initialize trainer
    trainer = topmost.trainers.DynamicTrainer(model, dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Train the model
    top_words, train_theta = trainer.train()
    
    # Save model results
    np.save(RESULTS_DIR / "dtm_top_words.npy", top_words)
    np.save(RESULTS_DIR / "dtm_train_theta.npy", train_theta)
    
    print(f"Model training complete. Results saved to {RESULTS_DIR}/")
    return top_words, train_theta


# ==============================================================================
# --- Part 3: Evaluation with Integrated Metrics ---
# ==============================================================================

def calculate_topic_coherence(topic_words, tokenized_texts, vocabulary):
    """Calculate standard topic coherence (NPMI-based) for a set of words."""
    if not topic_words:
        return 0.0
    
    coherence_sum = 0.0
    pair_count = 0
    epsilon = 1e-10
    
    doc_freq = {word: 0 for word in vocabulary}
    for text in tokenized_texts:
        unique_words_in_doc = set(text)
        for word in unique_words_in_doc:
            if word in doc_freq:
                doc_freq[word] += 1
                
    total_docs = len(tokenized_texts)

    for i, word_i in enumerate(topic_words):
        for j, word_j in enumerate(topic_words):
            if i < j and word_i in vocabulary and word_j in vocabulary:
                count_i = doc_freq.get(word_i, 0)
                count_j = doc_freq.get(word_j, 0)
                
                count_ij = sum(1 for text in tokenized_texts if word_i in text and word_j in text)
                
                if count_i > 0 and count_j > 0:
                    prob_i = count_i / total_docs
                    prob_j = count_j / total_docs
                    prob_ij = count_ij / total_docs
                    
                    if prob_ij > 0:
                        pmi = np.log(prob_ij / (prob_i * prob_j + epsilon))
                        # Normalize PMI to range [-1, 1]
                        npmi = pmi / (-np.log(prob_ij + epsilon))
                        coherence_sum += npmi
                        pair_count += 1
    
    return coherence_sum / pair_count if pair_count > 0 else 0.0


def calculate_temporal_topic_coherence(topic_k, time_t, topic_words_by_time, tokenized_texts, vocabulary, num_time_slices):
    """Calculate Temporal Topic Coherence (TTC) for topic k at time t."""
    if time_t + 1 >= num_time_slices:
        return 0.0
    
    words_t = topic_words_by_time[time_t].get(topic_k, [])
    words_t_plus_1 = topic_words_by_time[time_t + 1].get(topic_k, [])
    
    if not words_t or not words_t_plus_1:
        return 0.0
    
    # Use topic coherence logic between word sets from consecutive timestamps
    combined_words = [words_t, words_t_plus_1]
    return calculate_topic_coherence(list(itertools.chain.from_iterable(combined_words)), tokenized_texts, vocabulary)


def calculate_temporal_topic_smoothness(topic_k, time_t, topic_words_by_time, num_time_slices):
    """Calculate Temporal Topic Smoothness (TTS) for topic k at time t."""
    if time_t + 1 >= num_time_slices:
        return 0.0
    
    words_t = set(topic_words_by_time[time_t].get(topic_k, []))
    words_t_plus_1 = set(topic_words_by_time[time_t + 1].get(topic_k, []))
    
    if not words_t or not words_t_plus_1:
        return 0.0
    
    intersection = len(words_t & words_t_plus_1)
    union = len(words_t | words_t_plus_1)
    
    return intersection / union if union > 0 else 0.0


def calculate_topic_diversity(all_topic_words):
    """Calculate topic diversity for a time slice."""
    if not all_topic_words:
        return 0.0
    
    all_words = list(itertools.chain.from_iterable(all_topic_words))
    if not all_words:
        return 0.0
    
    unique_words = len(set(all_words))
    return unique_words / len(all_words)


def create_visualizations(results, vis_dir):
    """Create comprehensive visualizations for DTM metrics."""
    print("\n--- Creating Visualizations ---")
    vis_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    metrics = ['TTC', 'TTS', 'TTQ', 'TC', 'TD', 'TQ', 'DTQ']
    values = [results['overall_metrics'][m] for m in metrics]
    
    bars = axes[0, 0].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'pink', 'yellow', 'red'])
    axes[0, 0].set_title('Overall DTM Metrics', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom')

    topics = list(results['ttq_per_topic'].keys())
    ttq_values = list(results['ttq_per_topic'].values())
    axes[0, 1].bar(topics, ttq_values, color='lightblue')
    axes[0, 1].set_title('TTQ per Topic', fontsize=14, fontweight='bold')
    
    time_slices = list(results['tq_per_time'].keys())
    tq_values = list(results['tq_per_time'].values())
    axes[0, 2].plot(time_slices, tq_values, 'o-', color='orange')
    axes[0, 2].set_title('TQ over Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'dtm_comprehensive_analysis.png', dpi=300)
    plt.close()
    print(f"Visualizations saved to {vis_dir}")


def save_results(results, metadata, res_dir):
    """Save all results to files."""
    print("\n--- Saving Results ---")
    res_dir.mkdir(exist_ok=True)
    
    detailed_results = {'evaluation_metadata': metadata, 'dtm_results': results}
    with open(res_dir / 'dtm_detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)

    summary_df = pd.DataFrame([{'Metric': k, 'Score': v} for k, v in results['overall_metrics'].items()])
    summary_df.to_csv(res_dir / 'dtm_metrics_summary.csv', index=False)
    
    print(f"Results saved to {res_dir}")
    return detailed_results


def generate_report(detailed_results, report_path):
    """Generate a markdown report of the evaluation."""
    print("\n--- Generating Report ---")
    results = detailed_results['dtm_results']
    metadata = detailed_results['evaluation_metadata']
    report = f"# Dynamic Topic Model Evaluation Report\n\n"
    report += f"## Analysis Overview\n- **Topics**: {metadata['num_topics']}\n\n"
    report += "## Overall Metrics\n"
    for metric, score in results['overall_metrics'].items():
        report += f"- **{metric}**: {score:.4f}\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {report_path}")


def evaluate_dtm_results():
    """Evaluate the DTM results using the entire training dataset."""
    print("\n--- 3. Evaluating DTM Results on Full Training Data ---")

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    top_words_raw = np.load(RESULTS_DIR / "dtm_top_words.npy", allow_pickle=True)
    with open(DATA_DIR / "time_map.json", 'r', encoding='utf-8') as f:
        time_map = json.load(f)
    
    # Prepare data for evaluation
    timestamps = sorted(time_map.keys())
    num_time_slices = len(timestamps)
    
    topic_words_by_time = {}
    for t_idx in range(num_time_slices):
        topic_words_by_time[t_idx] = {}
        for k_idx in range(NUM_TOPICS):
            topic_words_by_time[t_idx][k_idx] = top_words_raw[t_idx][k_idx][:N_WORDS_PER_TOPIC]

    # Use entire training set as reference corpus
    corpus_texts = train_df['Processed_Comment'].tolist()
    tokenized_texts = [text.split() for text in corpus_texts]
    with open(DATA_DIR / "vocab.txt", 'r', encoding='utf-8') as f:
        vocabulary = [line.strip() for line in f]
    
    # Compute metrics
    results = {'ttc_per_topic': {}, 'tts_per_topic': {}, 'ttq_per_topic': {}, 'tc_per_time': {}, 'td_per_time': {}, 'tq_per_time': {}}
    
    for k in range(NUM_TOPICS):
        ttc_scores = [calculate_temporal_topic_coherence(k, t, topic_words_by_time, tokenized_texts, vocabulary, num_time_slices) for t in range(num_time_slices)]
        tts_scores = [calculate_temporal_topic_smoothness(k, t, topic_words_by_time, num_time_slices) for t in range(num_time_slices)]
        results['ttc_per_topic'][k] = ttc_scores
        results['tts_per_topic'][k] = tts_scores
        results['ttq_per_topic'][k] = np.mean([ttc * tts for ttc, tts in zip(ttc_scores, tts_scores) if ttc is not None and tts is not None])

    for t in range(num_time_slices):
        topic_words_at_t = [topic_words_by_time[t][k] for k in range(NUM_TOPICS)]
        results['tc_per_time'][t] = np.mean([calculate_topic_coherence(words, tokenized_texts, vocabulary) for words in topic_words_at_t])
        results['td_per_time'][t] = calculate_topic_diversity(topic_words_at_t)
        results['tq_per_time'][t] = results['tc_per_time'][t] * results['td_per_time'][t]

    # Compute overall metrics
    overall_ttq = np.mean(list(results['ttq_per_topic'].values()))
    overall_tq = np.mean(list(results['tq_per_time'].values()))
    results['overall_metrics'] = {
        'TTC': np.mean([item for sublist in results['ttc_per_topic'].values() for item in sublist]),
        'TTS': np.mean([item for sublist in results['tts_per_topic'].values() for item in sublist]),
        'TTQ': overall_ttq,
        'TC': np.mean(list(results['tc_per_time'].values())),
        'TD': np.mean(list(results['td_per_time'].values())),
        'TQ': overall_tq,
        'DTQ': 0.5 * (overall_tq + overall_ttq)
    }

    # Save and report
    eval_dir = RESULTS_DIR / "evaluation_on_full_data"
    metadata = {'num_topics': NUM_TOPICS, 'n_words': N_WORDS_PER_TOPIC}
    detailed_results = save_results(results, metadata, eval_dir)
    create_visualizations(results, eval_dir)
    generate_report(detailed_results, eval_dir / "dtm_evaluation_report.md")

    print("\n✅ Evaluation complete!")
    print(f"Check the {eval_dir} directory for detailed analysis.")


# ==============================================================================
# --- Main Execution ---
# ==============================================================================

def main():
    """Main execution function to run the entire pipeline."""
    print("🚀 Starting Russian DTM Pipeline...")
    print("="*60)

    # Step 1: Prepare data
    data_dir = prepare_data()
    if not data_dir:
        return

    # Step 2: Train model
    try:
        train_dtm_model(data_dir)
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Evaluate results
    try:
        evaluate_dtm_results()
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\n🎉 Pipeline finished successfully!")

if __name__ == "__main__":
    main() 