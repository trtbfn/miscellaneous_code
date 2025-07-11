import pandas as pd
import os

def show_language_stats(file_path, file_type):
    """
    Show language distribution statistics for a file
    """
    if not os.path.exists(file_path):
        print(f"{file_type} file not found: {file_path}")
        return
    
    print(f"\n{'='*50}")
    print(f"{file_type} LANGUAGE DISTRIBUTION")
    print(f"{'='*50}")
    
    # Read the file
    df = pd.read_csv(file_path)
    
    print(f"Total comments: {len(df):,}")
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    # Language distribution
    print(f"\nLanguage Distribution:")
    lang_counts = df['Language'].value_counts()
    
    # Show top 20 languages
    print(f"\nTop 20 languages:")
    for i, (lang, count) in enumerate(lang_counts.head(20).items()):
        percentage = (count / len(df)) * 100
        print(f"  {i+1:2d}. {lang:>6s}: {count:>8,} comments ({percentage:>6.2f}%)")
    
    # Show summary statistics
    print(f"\nSummary:")
    print(f"  Total unique languages detected: {len(lang_counts)}")
    print(f"  Most common language: {lang_counts.index[0]} ({lang_counts.iloc[0]:,} comments, {(lang_counts.iloc[0]/len(df)*100):.2f}%)")
    
    # Show language score statistics
    if 'Language_Score' in df.columns:
        print(f"\nLanguage Detection Confidence:")
        print(f"  Mean confidence score: {df['Language_Score'].mean():.3f}")
        print(f"  Median confidence score: {df['Language_Score'].median():.3f}")
        print(f"  Comments with high confidence (>0.9): {len(df[df['Language_Score'] > 0.9]):,} ({(len(df[df['Language_Score'] > 0.9])/len(df)*100):.2f}%)")
        print(f"  Comments with low confidence (<0.5): {len(df[df['Language_Score'] < 0.5]):,} ({(len(df[df['Language_Score'] < 0.5])/len(df)*100):.2f}%)")
    
    return df

def main():
    # File paths
    youtube_file = "data/formatted_all_youtube_comments_with_language.csv"
    telegram_file = "data/formatted_telegram_comments_with_language.csv"
    
    print("LANGUAGE CLASSIFICATION RESULTS")
    print("="*50)
    
    # Show stats for both files
    youtube_df = show_language_stats(youtube_file, "YOUTUBE COMMENTS")
    telegram_df = show_language_stats(telegram_file, "TELEGRAM COMMENTS")
    
    # Combined statistics
    if youtube_df is not None and telegram_df is not None:
        print(f"\n{'='*50}")
        print("COMBINED STATISTICS")
        print(f"{'='*50}")
        
        total_comments = len(youtube_df) + len(telegram_df)
        print(f"Total comments across both platforms: {total_comments:,}")
        
        # Combine language counts
        youtube_langs = youtube_df['Language'].value_counts()
        telegram_langs = telegram_df['Language'].value_counts()
        
        # Create combined dataframe
        combined_langs = pd.concat([youtube_langs, telegram_langs], axis=1).fillna(0)
        combined_langs.columns = ['YouTube', 'Telegram']
        combined_langs['Total'] = combined_langs.sum(axis=1)
        combined_langs = combined_langs.sort_values('Total', ascending=False)
        
        print(f"\nTop 10 languages across both platforms:")
        for i, (lang, row) in enumerate(combined_langs.head(10).iterrows()):
            percentage = (row['Total'] / total_comments) * 100
            print(f"  {i+1:2d}. {lang:>6s}: {row['Total']:>8,} total ({row['YouTube']:>6,} YouTube, {row['Telegram']:>6,} Telegram) - {percentage:>6.2f}%")

if __name__ == "__main__":
    main() 