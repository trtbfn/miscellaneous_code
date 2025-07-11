#!/usr/bin/env python3
"""
Simplified Dynamic Topic Quality Metrics Implementation
Based on "Evaluating Dynamic Topic Models" paper sections 4.1-4.4

Implements:
4.1 Temporal Topic Coherence (TTC)
4.2 Temporal Topic Smoothness (TTS)  
4.3 Temporal Topic Quality (TTQ)
4.4 Dynamic Topic Quality (DTQ)

Plus our existing TopMost evaluation for comparison.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import itertools
warnings.filterwarnings('ignore')

# Try to import TopMost for comparison
try:
    import topmost
    from topmost import eva
    TOPMOST_AVAILABLE = True
    print("TopMost library loaded successfully")
except ImportError:
    TOPMOST_AVAILABLE = False
    print("TopMost not available - using DTM metrics only")


class SimplifiedDTMEvaluator:
    """
    Simplified implementation of Dynamic Topic Quality metrics
    """
    
    def __init__(self, bertopic_results_path, topics_over_time_path, top_n_topics=30, n_words=10):
        """
        Initialize DTM evaluator
        
        Args:
            bertopic_results_path: Path to BERTopic results CSV
            topics_over_time_path: Path to topics over time CSV
            top_n_topics: Number of top topics to evaluate
            n_words: Number of words per topic for evaluation
        """
        self.top_n_topics = top_n_topics
        self.n_words = n_words
        
        # Set up directories
        self.results_dir = Path("dtm_evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"=== Simplified DTM Evaluation ===")
        print(f"Top N Topics: {top_n_topics}")
        print(f"Words per Topic: {n_words}")
        
        # Load BERTopic results
        print("\nLoading BERTopic results...")
        self.df = pd.read_csv(bertopic_results_path)
        self.topics_over_time = pd.read_csv(topics_over_time_path)
        
        # Load topic information
        with open("topic_words.json", 'r', encoding='utf-8') as f:
            self.topic_words = json.load(f)
        
        print(f"Loaded {len(self.df)} documents with {len(self.topic_words)} topics")
        
        # Process data
        self.prepare_data()
    
    def prepare_data(self):
        """
        Prepare data for DTM evaluation
        """
        print("\n=== Preparing Data ===")
        
        # Filter out outliers and get top N topics
        valid_topics = self.df[self.df['Topic'] != -1].copy()
        topic_counts = valid_topics['Topic'].value_counts()
        self.selected_topics = topic_counts.head(self.top_n_topics).index.tolist()
        
        print(f"Selected top {len(self.selected_topics)} topics: {self.selected_topics}")
        
        # Filter data
        self.filtered_df = valid_topics[valid_topics['Topic'].isin(self.selected_topics)].copy()
        
        # Process timestamps
        self.filtered_df['Published_At'] = pd.to_datetime(self.filtered_df['Published_At'])
        self.filtered_df = self.filtered_df.sort_values('Published_At')
        
        # Get unique timestamps and create time mapping
        unique_timestamps = sorted(self.topics_over_time['Timestamp'].unique())
        self.num_time_slices = len(unique_timestamps)
        self.timestamps = unique_timestamps
        
        print(f"Number of time slices: {self.num_time_slices}")
        
        # Map documents to time slices
        self.filtered_df['Time_Slice'] = self.filtered_df['Published_At'].apply(
            lambda x: self._find_closest_time_slice(x, unique_timestamps)
        )
        
        # Prepare topic words by time
        self.prepare_topic_words_by_time()
        
        # Prepare corpus for coherence calculation
        self.prepare_corpus()
        
        print("Data preparation completed")
    
    def _find_closest_time_slice(self, doc_time, timestamps):
        """Find the closest time slice for a document"""
        doc_time = pd.to_datetime(doc_time)
        timestamps = [pd.to_datetime(ts) for ts in timestamps]
        
        closest_idx = min(range(len(timestamps)), 
                         key=lambda i: abs((timestamps[i] - doc_time).total_seconds()))
        return closest_idx
    
    def prepare_topic_words_by_time(self):
        """
        Prepare topic words organized by time and topic
        """
        print("Preparing topic words by time...")
        
        # Initialize structure: [time_slice][topic_id] = [words]
        self.topic_words_by_time = {}
        
        for t, timestamp in enumerate(self.timestamps):
            self.topic_words_by_time[t] = {}
            
            for topic_id in self.selected_topics:
                # Get words for this topic at this time
                mask = (self.topics_over_time['Timestamp'] == timestamp) & \
                       (self.topics_over_time['Topic'] == topic_id)
                topic_data = self.topics_over_time[mask]
                
                if len(topic_data) > 0:
                    words = topic_data['Words'].iloc[0]
                    if isinstance(words, str):
                        # Parse words and clean them
                        word_list = []
                        for word in words.split(','):
                            clean_word = word.strip()
                            # Remove clustering prefix if present
                            if clean_word.startswith('clustering '):
                                clean_word = clean_word[11:]
                            if clean_word:
                                word_list.append(clean_word)
                        
                        # Take top N words
                        self.topic_words_by_time[t][topic_id] = word_list[:self.n_words]
                    else:
                        self.topic_words_by_time[t][topic_id] = []
                else:
                    self.topic_words_by_time[t][topic_id] = []
        
        print(f"Prepared topic words for {len(self.timestamps)} time slices")
    
    def prepare_corpus(self):
        """
        Prepare corpus for coherence calculations
        """
        print("Preparing reference corpus...")
        
        # Create list of all documents for coherence calculation
        self.corpus_texts = self.filtered_df['Processed_Comment'].tolist()
        
        # Tokenize documents
        self.tokenized_texts = [text.split() for text in self.corpus_texts]
        
        # Create vocabulary from all documents
        vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            max_features=5000,
            min_df=2,
            max_df=0.9
        )
        
        X = vectorizer.fit_transform(self.corpus_texts)
        self.vocabulary = vectorizer.get_feature_names_out().tolist()
        
        print(f"Corpus size: {len(self.corpus_texts)} documents")
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def calculate_temporal_topic_coherence(self, topic_k, time_t, window_size=1):
        """
        Calculate Temporal Topic Coherence (TTC) for topic k at time t
        Based on paper section 4.1
        
        TTC measures coherence between consecutive time stamps of one topic
        """
        if time_t + window_size >= self.num_time_slices:
            return 0.0
        
        # Get words for topic k at time t and t+window_size
        words_t = self.topic_words_by_time[time_t].get(topic_k, [])
        words_t_plus = self.topic_words_by_time[time_t + window_size].get(topic_k, [])
        
        if not words_t or not words_t_plus:
            return 0.0
        
        # Calculate coherence between word pairs across time
        coherence_sum = 0.0
        pair_count = 0
        epsilon = 1e-10
        
        for word_i in words_t:
            for word_j in words_t_plus:
                if word_i in self.vocabulary and word_j in self.vocabulary:
                    # Count co-occurrences in reference corpus
                    count_i = sum(1 for text in self.tokenized_texts if word_i in text)
                    count_j = sum(1 for text in self.tokenized_texts if word_j in text)
                    count_ij = sum(1 for text in self.tokenized_texts if word_i in text and word_j in text)
                    
                    if count_i > 0 and count_j > 0:
                        # PMI-based coherence
                        prob_i = count_i / len(self.tokenized_texts)
                        prob_j = count_j / len(self.tokenized_texts)
                        prob_ij = count_ij / len(self.tokenized_texts)
                        
                        if prob_ij > 0:
                            coherence = np.log((prob_ij + epsilon) / (prob_i * prob_j + epsilon))
                            coherence_sum += coherence
                            pair_count += 1
        
        return coherence_sum / pair_count if pair_count > 0 else 0.0
    
    def calculate_temporal_topic_smoothness(self, topic_k, time_t, window_size=1):
        """
        Calculate Temporal Topic Smoothness (TTS) for topic k at time t
        Based on paper section 4.2
        
        TTS measures smoothness of topic evolution (opposite of diversity)
        """
        if time_t + window_size >= self.num_time_slices:
            return 0.0
        
        # Get words for consecutive time periods
        words_sequence = []
        for t in range(time_t, min(time_t + window_size + 1, self.num_time_slices)):
            words = self.topic_words_by_time[t].get(topic_k, [])
            if words:
                words_sequence.append(words)
        
        if len(words_sequence) < 2:
            return 0.0
        
        # Calculate smoothness as inverse of diversity
        # Smoothness is high when words overlap between consecutive periods
        total_smoothness = 0.0
        comparison_count = 0
        
        for i in range(len(words_sequence) - 1):
            words_i = set(words_sequence[i])
            words_j = set(words_sequence[i + 1])
            
            if words_i and words_j:
                # Jaccard similarity as smoothness measure
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                smoothness = intersection / union if union > 0 else 0.0
                total_smoothness += smoothness
                comparison_count += 1
        
        return total_smoothness / comparison_count if comparison_count > 0 else 0.0
    
    def calculate_topic_coherence(self, topic_words):
        """
        Calculate standard topic coherence for a set of words
        """
        if not topic_words:
            return 0.0
        
        coherence_sum = 0.0
        pair_count = 0
        epsilon = 1e-10
        
        for i, word_i in enumerate(topic_words):
            for j, word_j in enumerate(topic_words):
                if i < j and word_i in self.vocabulary and word_j in self.vocabulary:
                    # Count co-occurrences
                    count_i = sum(1 for text in self.tokenized_texts if word_i in text)
                    count_j = sum(1 for text in self.tokenized_texts if word_j in text)
                    count_ij = sum(1 for text in self.tokenized_texts if word_i in text and word_j in text)
                    
                    if count_i > 0 and count_j > 0:
                        prob_i = count_i / len(self.tokenized_texts)
                        prob_j = count_j / len(self.tokenized_texts)
                        prob_ij = count_ij / len(self.tokenized_texts)
                        
                        if prob_ij > 0:
                            coherence = np.log((prob_ij + epsilon) / (prob_i * prob_j + epsilon))
                            coherence_sum += coherence
                            pair_count += 1
        
        return coherence_sum / pair_count if pair_count > 0 else 0.0
    
    def calculate_topic_diversity(self, all_topic_words):
        """
        Calculate topic diversity for a time slice
        """
        if not all_topic_words:
            return 0.0
        
        # Flatten all words
        all_words = []
        for topic_words in all_topic_words:
            all_words.extend(topic_words)
        
        if not all_words:
            return 0.0
        
        # Calculate diversity as ratio of unique words to total words
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        return unique_words / total_words if total_words > 0 else 0.0
    
    def compute_all_metrics(self):
        """
        Compute all DTM metrics: TTC, TTS, TTQ, TC, TD, TQ, DTQ
        """
        print("\n=== Computing DTM Metrics ===")
        
        results = {
            'topics': [],
            'time_slices': [],
            'ttc_per_topic': {},  # TTC for each topic across time
            'tts_per_topic': {},  # TTS for each topic across time
            'ttq_per_topic': {},  # TTQ for each topic
            'tc_per_time': {},    # TC for each time slice
            'td_per_time': {},    # TD for each time slice
            'tq_per_time': {},    # TQ for each time slice
            'overall_metrics': {}
        }
        
        # 1. Compute TTC and TTS for each topic across time
        print("Computing Temporal Topic Coherence (TTC) and Smoothness (TTS)...")
        
        for topic_k in self.selected_topics:
            ttc_scores = []
            tts_scores = []
            
            for time_t in range(self.num_time_slices - 1):  # -1 because we look ahead
                ttc = self.calculate_temporal_topic_coherence(topic_k, time_t)
                tts = self.calculate_temporal_topic_smoothness(topic_k, time_t)
                
                ttc_scores.append(ttc)
                tts_scores.append(tts)
            
            results['ttc_per_topic'][topic_k] = ttc_scores
            results['tts_per_topic'][topic_k] = tts_scores
            
            # Calculate TTQ for this topic (average of TTC * TTS)
            ttq_scores = [ttc * tts for ttc, tts in zip(ttc_scores, tts_scores)]
            results['ttq_per_topic'][topic_k] = np.mean(ttq_scores) if ttq_scores else 0.0
        
        # 2. Compute TC and TD for each time slice
        print("Computing Topic Coherence (TC) and Diversity (TD) per time slice...")
        
        for time_t in range(self.num_time_slices):
            # Get all topic words for this time slice
            all_topic_words = []
            tc_scores = []
            
            for topic_k in self.selected_topics:
                topic_words = self.topic_words_by_time[time_t].get(topic_k, [])
                if topic_words:
                    all_topic_words.append(topic_words)
                    tc = self.calculate_topic_coherence(topic_words)
                    tc_scores.append(tc)
            
            # Average TC for this time slice
            results['tc_per_time'][time_t] = np.mean(tc_scores) if tc_scores else 0.0
            
            # TD for this time slice
            results['td_per_time'][time_t] = self.calculate_topic_diversity(all_topic_words)
            
            # TQ for this time slice (TC * TD)
            results['tq_per_time'][time_t] = results['tc_per_time'][time_t] * results['td_per_time'][time_t]
        
        # 3. Compute overall metrics
        print("Computing overall metrics...")
        
        # Overall TTC, TTS, TTQ
        all_ttc = []
        all_tts = []
        all_ttq = []
        
        for topic_k in self.selected_topics:
            if results['ttc_per_topic'][topic_k]:
                all_ttc.extend(results['ttc_per_topic'][topic_k])
            if results['tts_per_topic'][topic_k]:
                all_tts.extend(results['tts_per_topic'][topic_k])
            all_ttq.append(results['ttq_per_topic'][topic_k])
        
        results['overall_metrics']['TTC'] = np.mean(all_ttc) if all_ttc else 0.0
        results['overall_metrics']['TTS'] = np.mean(all_tts) if all_tts else 0.0
        results['overall_metrics']['TTQ'] = np.mean(all_ttq) if all_ttq else 0.0
        
        # Overall TC, TD, TQ
        all_tc = list(results['tc_per_time'].values())
        all_td = list(results['td_per_time'].values())
        all_tq = list(results['tq_per_time'].values())
        
        results['overall_metrics']['TC'] = np.mean(all_tc) if all_tc else 0.0
        results['overall_metrics']['TD'] = np.mean(all_td) if all_td else 0.0
        results['overall_metrics']['TQ'] = np.mean(all_tq) if all_tq else 0.0
        
        # Dynamic Topic Quality (DTQ) - Section 4.4
        # DTQ = 1/2 * [average(TQ_t) + average(TTQ_k)]
        results['overall_metrics']['DTQ'] = 0.5 * (
            results['overall_metrics']['TQ'] + results['overall_metrics']['TTQ']
        )
        
        print("DTM metrics computation completed")
        
        return results
    
    def create_visualizations(self, results):
        """
        Create comprehensive visualizations for DTM metrics
        """
        print("\n=== Creating Visualizations ===")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Overall Metrics Bar Chart
        metrics = ['TTC', 'TTS', 'TTQ', 'TC', 'TD', 'TQ', 'DTQ']
        values = [results['overall_metrics'][m] for m in metrics]
        
        bars = axes[0, 0].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 
                                                     'orange', 'pink', 'yellow', 'red'])
        axes[0, 0].set_title('Overall DTM Metrics', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. TTQ per Topic
        topics = list(results['ttq_per_topic'].keys())
        ttq_values = list(results['ttq_per_topic'].values())
        
        axes[0, 1].bar(range(len(topics)), ttq_values, color='lightblue')
        axes[0, 1].set_title('TTQ (Temporal Topic Quality) per Topic', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Topic ID')
        axes[0, 1].set_ylabel('TTQ Score')
        axes[0, 1].set_xticks(range(0, len(topics), max(1, len(topics)//10)))
        axes[0, 1].set_xticklabels([topics[i] for i in range(0, len(topics), max(1, len(topics)//10))])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. TQ per Time Slice
        time_slices = list(results['tq_per_time'].keys())
        tq_values = list(results['tq_per_time'].values())
        
        axes[0, 2].plot(time_slices, tq_values, 'o-', linewidth=2, markersize=6, color='orange')
        axes[0, 2].set_title('TQ (Topic Quality) over Time', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Time Slice')
        axes[0, 2].set_ylabel('TQ Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. TC vs TD Scatter
        tc_values = list(results['tc_per_time'].values())
        td_values = list(results['td_per_time'].values())
        
        scatter = axes[1, 0].scatter(tc_values, td_values, c=tq_values, 
                                    cmap='viridis', s=60, alpha=0.7)
        axes[1, 0].set_title('TC vs TD (colored by TQ)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('TC (Topic Coherence)')
        axes[1, 0].set_ylabel('TD (Topic Diversity)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='TQ Score')
        
        # 5. TTC and TTS Evolution
        if results['ttc_per_topic'] and results['tts_per_topic']:
            # Average TTC and TTS across topics
            avg_ttc_per_time = []
            avg_tts_per_time = []
            
            for t in range(self.num_time_slices - 1):
                ttc_at_t = [results['ttc_per_topic'][topic][t] 
                           for topic in results['ttc_per_topic'] 
                           if t < len(results['ttc_per_topic'][topic])]
                tts_at_t = [results['tts_per_topic'][topic][t] 
                           for topic in results['tts_per_topic'] 
                           if t < len(results['tts_per_topic'][topic])]
                
                avg_ttc_per_time.append(np.mean(ttc_at_t) if ttc_at_t else 0.0)
                avg_tts_per_time.append(np.mean(tts_at_t) if tts_at_t else 0.0)
            
            axes[1, 1].plot(range(len(avg_ttc_per_time)), avg_ttc_per_time, 
                           'o-', label='TTC', linewidth=2, markersize=6)
            axes[1, 1].plot(range(len(avg_tts_per_time)), avg_tts_per_time, 
                           's-', label='TTS', linewidth=2, markersize=6)
            axes[1, 1].set_title('TTC & TTS Evolution', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Time Period')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. DTQ Highlights
        dtq_score = results['overall_metrics']['DTQ']
        axes[1, 2].bar(['DTQ'], [dtq_score], width=0.5, color='red', alpha=0.7)
        axes[1, 2].set_title(f'Dynamic Topic Quality (DTQ)\n{dtq_score:.4f}', 
                            fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('DTQ Score')
        axes[1, 2].set_ylim(0, max(1.0, dtq_score * 1.2))
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add quality assessment text
        if dtq_score > 0.7:
            quality = "Excellent"
            color = "green"
        elif dtq_score > 0.5:
            quality = "Good"
            color = "orange"
        elif dtq_score > 0.3:
            quality = "Moderate"
            color = "blue"
        else:
            quality = "Poor"
            color = "red"
        
        axes[1, 2].text(0, dtq_score + 0.05, quality, ha='center', va='bottom',
                       fontsize=12, fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'dtm_comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved")
    
    def save_results(self, results):
        """
        Save all results to files
        """
        print("\n=== Saving Results ===")
        
        # Save detailed results as JSON
        detailed_results = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'top_n_topics': self.top_n_topics,
                'n_words': self.n_words,
                'num_time_slices': self.num_time_slices,
                'selected_topics': self.selected_topics
            },
            'dtm_results': results
        }
        
        with open(self.results_dir / 'dtm_detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
        
        # Save summary CSV
        summary_data = [{
            'Metric': metric,
            'Score': score,
            'Description': self._get_metric_description(metric)
        } for metric, score in results['overall_metrics'].items()]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.results_dir / 'dtm_metrics_summary.csv', index=False)
        
        # Save per-topic TTQ results
        ttq_df = pd.DataFrame([
            {'Topic_ID': topic_id, 'TTQ_Score': score}
            for topic_id, score in results['ttq_per_topic'].items()
        ])
        ttq_df.to_csv(self.results_dir / 'ttq_per_topic.csv', index=False)
        
        # Save per-time TQ results
        tq_df = pd.DataFrame([
            {
                'Time_Slice': time_slice,
                'TC_Score': results['tc_per_time'][time_slice],
                'TD_Score': results['td_per_time'][time_slice], 
                'TQ_Score': score
            }
            for time_slice, score in results['tq_per_time'].items()
        ])
        tq_df.to_csv(self.results_dir / 'tq_per_time.csv', index=False)
        
        print(f"Results saved to {self.results_dir}/")
        
        return detailed_results
    
    def _get_metric_description(self, metric):
        """Get description for each metric"""
        descriptions = {
            'TTC': 'Temporal Topic Coherence - semantic consistency over time',
            'TTS': 'Temporal Topic Smoothness - smooth evolution over time',
            'TTQ': 'Temporal Topic Quality - TTC × TTS',
            'TC': 'Topic Coherence - semantic coherence per time slice',
            'TD': 'Topic Diversity - topic diversity per time slice',
            'TQ': 'Topic Quality - TC × TD',
            'DTQ': 'Dynamic Topic Quality - overall DTM quality measure'
        }
        return descriptions.get(metric, 'Unknown metric')
    
    def generate_report(self, detailed_results):
        """
        Generate comprehensive DTM evaluation report
        """
        print("\n=== Generating Report ===")
        
        results = detailed_results['dtm_results']
        metadata = detailed_results['evaluation_metadata']
        
        report = f"""
# Dynamic Topic Model Evaluation Report

## Analysis Overview
- **Analysis Date**: {metadata['timestamp']}
- **Top N Topics**: {metadata['top_n_topics']}
- **Words per Topic**: {metadata['n_words']}
- **Time Slices**: {metadata['num_time_slices']}
- **Selected Topics**: {', '.join(map(str, metadata['selected_topics']))}

## Dynamic Topic Quality Metrics

### From "Evaluating Dynamic Topic Models" Paper (Sections 4.1-4.4)

#### 4.1 Temporal Topic Coherence (TTC): {results['overall_metrics']['TTC']:.4f}
Measures how semantically consistent topics are across consecutive time stamps.
- **Interpretation**: {"High coherence - topics maintain semantic consistency over time" if results['overall_metrics']['TTC'] > 0.1 else "Medium coherence - some semantic drift over time" if results['overall_metrics']['TTC'] > 0.05 else "Low coherence - significant semantic changes over time"}

#### 4.2 Temporal Topic Smoothness (TTS): {results['overall_metrics']['TTS']:.4f}
Measures how smoothly topics evolve over time (opposite of diversity).
- **Interpretation**: {"High smoothness - gradual topic evolution" if results['overall_metrics']['TTS'] > 0.5 else "Medium smoothness - moderate changes" if results['overall_metrics']['TTS'] > 0.3 else "Low smoothness - abrupt topic changes"}

#### 4.3 Temporal Topic Quality (TTQ): {results['overall_metrics']['TTQ']:.4f}
Combination of TTC and TTS (TTQ = TTC × TTS).
- **Interpretation**: {"Excellent temporal quality" if results['overall_metrics']['TTQ'] > 0.5 else "Good temporal quality" if results['overall_metrics']['TTQ'] > 0.3 else "Moderate temporal quality" if results['overall_metrics']['TTQ'] > 0.1 else "Poor temporal quality"}

### Topic Quality per Time Slice

#### Topic Coherence (TC): {results['overall_metrics']['TC']:.4f}
Standard topic coherence averaged across time slices.

#### Topic Diversity (TD): {results['overall_metrics']['TD']:.4f}
Topic diversity averaged across time slices.

#### Topic Quality (TQ): {results['overall_metrics']['TQ']:.4f}
Combination of TC and TD (TQ = TC × TD).

### 4.4 Dynamic Topic Quality (DTQ): {results['overall_metrics']['DTQ']:.4f}

**Overall DTM Quality Assessment**: {"🏆 Excellent" if results['overall_metrics']['DTQ'] > 0.7 else "✅ Good" if results['overall_metrics']['DTQ'] > 0.5 else "⚠️  Moderate" if results['overall_metrics']['DTQ'] > 0.3 else "❌ Poor"}

DTQ is computed as: DTQ = 1/2 × [average(TQ_t) + average(TTQ_k)]

This measure combines:
- **Temporal consistency** (how well topics maintain coherence over time)
- **Topic quality** (how coherent and diverse topics are within each time period)

## Top Performing Topics (by TTQ)

"""
        
        # Add top topics by TTQ
        sorted_topics = sorted(results['ttq_per_topic'].items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        
        for i, (topic_id, ttq_score) in enumerate(sorted_topics, 1):
            report += f"{i}. **Topic {topic_id}**: TTQ = {ttq_score:.4f}\n"
        
        report += f"""

## Time Slice Analysis

### Best Time Periods (by TQ)
"""
        
        # Add best time periods
        sorted_times = sorted(results['tq_per_time'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        
        for i, (time_slice, tq_score) in enumerate(sorted_times, 1):
            timestamp = self.timestamps[time_slice] if time_slice < len(self.timestamps) else f"Time {time_slice}"
            report += f"{i}. **{timestamp}**: TQ = {tq_score:.4f}\n"
        
        report += f"""

## Key Findings

1. **Temporal Consistency**: {"Topics show strong semantic consistency over time, indicating stable thematic evolution." if results['overall_metrics']['TTC'] > 0.1 else "Topics show moderate temporal consistency with some semantic drift." if results['overall_metrics']['TTC'] > 0.05 else "Topics show significant changes over time, indicating dynamic thematic evolution."}

2. **Evolution Pattern**: {"Topic evolution is smooth and gradual, suggesting natural thematic development." if results['overall_metrics']['TTS'] > 0.5 else "Topic evolution shows moderate changes over time." if results['overall_metrics']['TTS'] > 0.3 else "Topic evolution shows abrupt changes, indicating significant shifts in discussions."}

3. **Overall Quality**: The DTQ score of {results['overall_metrics']['DTQ']:.4f} indicates {"excellent dynamic topic modeling performance" if results['overall_metrics']['DTQ'] > 0.7 else "good dynamic topic modeling performance" if results['overall_metrics']['DTQ'] > 0.5 else "moderate dynamic topic modeling performance" if results['overall_metrics']['DTQ'] > 0.3 else "poor dynamic topic modeling performance"}.

## Recommendations

"""
        
        if results['overall_metrics']['DTQ'] > 0.7:
            report += "✅ **Excellent Performance**: The model shows outstanding dynamic topic quality. Continue using current parameters.\n"
        elif results['overall_metrics']['DTQ'] > 0.5:
            report += "✅ **Good Performance**: The model performs well. Consider fine-tuning parameters for potential improvements.\n"
        elif results['overall_metrics']['DTQ'] > 0.3:
            report += "⚠️ **Moderate Performance**: Consider adjusting model parameters, preprocessing steps, or number of topics.\n"
        else:
            report += "❌ **Poor Performance**: Significant improvements needed. Review data preprocessing, model parameters, and topic number selection.\n"
        
        if results['overall_metrics']['TTC'] < 0.05:
            report += "- **Low TTC**: Topics show significant semantic drift. Consider adjusting temporal modeling parameters.\n"
        
        if results['overall_metrics']['TTS'] < 0.3:
            report += "- **Low TTS**: Abrupt topic changes detected. Investigate periods with major shifts in discussions.\n"
        
        report += f"""

## Files Generated

- `dtm_detailed_results.json`: Complete evaluation results
- `dtm_metrics_summary.csv`: Summary of all DTM metrics  
- `ttq_per_topic.csv`: TTQ scores for each topic
- `tq_per_time.csv`: TQ scores for each time slice
- `dtm_comprehensive_analysis.png`: Comprehensive visualizations
- `dtm_evaluation_report.md`: This report

## References

- Karakkaparambil James, Charu, et al. "Evaluating Dynamic Topic Models." *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. 2024.
- BERTopic implementation with ru-en-RoSBERTa embeddings

---
*Report generated by Simplified DTM Evaluation System*
"""
        
        # Save report
        with open(self.results_dir / 'dtm_evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("Evaluation report generated!")
        
        return report
    
    def run_evaluation(self):
        """
        Run complete DTM evaluation
        """
        print("🚀 Starting Simplified DTM Evaluation")
        print("Implementing metrics from 'Evaluating Dynamic Topic Models' paper")
        print("="*60)
        
        try:
            # Compute all DTM metrics
            results = self.compute_all_metrics()
            
            # Create visualizations
            self.create_visualizations(results)
            
            # Save results
            detailed_results = self.save_results(results)
            
            # Generate report
            report = self.generate_report(detailed_results)
            
            print("\n🎯 DTM Evaluation Complete!")
            print(f"📁 Results saved to: {self.results_dir}/")
            print("\n📊 Final DTM Scores:")
            print(f"  TTC (Temporal Topic Coherence): {results['overall_metrics']['TTC']:.4f}")
            print(f"  TTS (Temporal Topic Smoothness): {results['overall_metrics']['TTS']:.4f}")
            print(f"  TTQ (Temporal Topic Quality): {results['overall_metrics']['TTQ']:.4f}")
            print(f"  TC (Topic Coherence): {results['overall_metrics']['TC']:.4f}")
            print(f"  TD (Topic Diversity): {results['overall_metrics']['TD']:.4f}")
            print(f"  TQ (Topic Quality): {results['overall_metrics']['TQ']:.4f}")
            print(f"  🏆 DTQ (Dynamic Topic Quality): {results['overall_metrics']['DTQ']:.4f}")
            
            # Quality assessment
            dtq_score = results['overall_metrics']['DTQ']
            if dtq_score > 0.7:
                print("\n🏆 Assessment: Excellent Dynamic Topic Model Quality!")
            elif dtq_score > 0.5:
                print("\n✅ Assessment: Good Dynamic Topic Model Quality")
            elif dtq_score > 0.3:
                print("\n⚠️  Assessment: Moderate Dynamic Topic Model Quality")
            else:
                print("\n❌ Assessment: Poor Dynamic Topic Model Quality")
            
            return detailed_results
            
        except Exception as e:
            print(f"❌ Error in DTM evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """
    Main execution function
    """
    print("🔬 Simplified Dynamic Topic Model Evaluation")
    print("Implementing all DTM metrics from paper sections 4.1-4.4")
    print("TTC, TTS, TTQ, DTQ + Topic Quality metrics")
    print("="*60)
    
    # File paths
    bertopic_results_path = "bertopic_results.csv"
    topics_over_time_path = "topics_over_time.csv"
    
    # Check if files exist
    required_files = [bertopic_results_path, topics_over_time_path]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ ERROR: {file_path} not found!")
            return
    
    # Initialize evaluator
    evaluator = SimplifiedDTMEvaluator(
        bertopic_results_path=bertopic_results_path,
        topics_over_time_path=topics_over_time_path,
        top_n_topics=30,
        n_words=10
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    if results:
        print("\n✅ Success! Check the dtm_evaluation_results/ directory for:")
        print("  📈 DTM analysis visualization")
        print("  📊 Detailed CSV results")
        print("  📑 Comprehensive evaluation report")
        print("  📋 All DTM metrics (TTC, TTS, TTQ, DTQ)")
    else:
        print("\n❌ Evaluation failed. Check error messages above.")


if __name__ == "__main__":
    main() 