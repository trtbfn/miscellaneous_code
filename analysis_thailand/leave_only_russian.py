import pandas as pd
import os

# Define relative paths to the data files from the script's location
input_files = [
    '../data/formatted_all_youtube_comments_with_language.csv',
    '../data/formatted_telegram_comments_with_language.csv'
]

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

for file_path in input_files:
    # Construct the absolute path to the input file
    absolute_file_path = os.path.join(script_dir, file_path)

    if not os.path.exists(absolute_file_path):
        print(f"File not found: {absolute_file_path}")
        continue

    print(f"Processing file: {absolute_file_path}")
    
    try:
        # Read the data
        df = pd.read_csv(absolute_file_path)

        # Check if 'Language' column exists
        if 'Language' not in df.columns:
            print(f"Error: 'Language' column not found in {absolute_file_path}.")
            print(f"Available columns are: {df.columns.tolist()}")
            continue

        # Filter for Russian language ('ru')
        russian_df = df[df['Language'] == 'ru'].copy()

        # Create the new file name
        directory, filename = os.path.split(absolute_file_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_ru{ext}"
        new_file_path = os.path.join(directory, new_filename)

        # Save the new dataframe to a csv file
        russian_df.to_csv(new_file_path, index=False, encoding='utf-8')

        print(f"Saved Russian comments to: {new_file_path}")

    except Exception as e:
        print(f"An error occurred while processing {absolute_file_path}: {e}")

print("Script finished.")
