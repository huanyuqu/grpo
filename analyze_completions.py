import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob
from scipy.stats import skew
from collections import defaultdict
import random


def analyze_completion_lengths(directory: str, batch_size: int = 72):
    """
    Analyzes the distribution of completion lengths from prompt_completions_rankX.csv files.

    Args:
        directory (str): The directory containing the CSV files (e.g., "Qwen2-0.5B-GRPO").
        batch_size (int): This parameter reflects the user's conceptual batch size (72 completions
                          across 3 files). The function collects all available completions from
                          the specified files and analyzes their overall distribution.
    """
    all_completion_lengths = []
    csv_files_pattern = os.path.join(directory, "prompt_completions_rank*.csv")
    
    found_files = glob.glob(csv_files_pattern)
    if not found_files:
        print(f"No CSV files matching '{csv_files_pattern}' found in '{directory}'.")
        return

    print(f"Found {len(found_files)} CSV files: {sorted(found_files)}")

    for filepath in sorted(found_files): # Sort to ensure consistent order
        try:
            df = pd.read_csv(filepath)
            if 'Completion Tokens' in df.columns:
                all_completion_lengths.extend(df['Completion Tokens'].tolist())
            else:
                print(f"Warning: 'Completion Tokens' column not found in {filepath}. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"Warning: {filepath} is empty. Skipping.")
        except FileNotFoundError: # Should not happen with glob, but good for robustness
            print(f"Error: {filepath} not found. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while reading {filepath}: {e}. Skipping.")

    if not all_completion_lengths:
        print("No completion lengths found across all files for analysis.")
        return

    print(f"\n--- Analysis for {len(all_completion_lengths)} Completions (conceptual batch size: {batch_size}) ---")
    
    # Convert to pandas Series for easier statistical analysis
    lengths_series = pd.Series(all_completion_lengths)

    print("\n--- Completion Length Distribution Statistics ---")
    print(lengths_series.describe())
    
    # Plotting the histogram to show distribution and long-tail
    plt.figure(figsize=(12, 7))
    
    # Use 'auto' for bins to let matplotlib decide a good number of bins
    plt.hist(lengths_series, bins='auto', edgecolor='black', alpha=0.7)
    
    plt.title(f"Distribution of Completion Lengths (Total: {len(all_completion_lengths)} completions)")
    plt.xlabel("Completion Length (tokens)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    
    # Add lines for mean and median to better understand distribution
    mean_len = lengths_series.mean()
    median_len = lengths_series.median()
    max_len = lengths_series.max()

    plt.axvline(x=median_len, color='r', linestyle='--', label=f'Median: {median_len:.2f}')
    plt.axvline(x=mean_len, color='g', linestyle=':', label=f'Mean: {mean_len:.2f}')
    plt.legend()

    # Add text for max length, especially if it's an outlier, to highlight the long tail
    # Heuristic: if max length is significantly larger than median and a certain absolute value
    if max_len > (median_len * 1.5) and max_len > 50: 
        plt.text(max_len, plt.ylim()[1] * 0.9, f'Max: {max_len}', 
                 color='purple', ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        print(f"\nNote: The distribution shows a long-tail characteristic. "
              f"Max length ({max_len}) is significantly higher than median ({median_len:.2f}) and mean ({mean_len:.2f}).")
    else:
        print("\nNo prominent long-tail characteristic detected based on simple heuristic for visual emphasis.")

    plot_filename = os.path.join(directory, "completion_length_distribution_histogram.png")
    plt.savefig(plot_filename)
    print(f"Completion length distribution histogram saved to {plot_filename}")
    plt.close()

    # Box plot to clearly show outliers (long tail)
    plt.figure(figsize=(10, 5))
    plt.boxplot(lengths_series, vert=False)
    plt.title(f"Box Plot of Completion Lengths (Total: {len(all_completion_lengths)} completions)")
    plt.xlabel("Completion Length (tokens)")
    plt.grid(axis='x', alpha=0.75)
    box_plot_filename = os.path.join(directory, "completion_length_distribution_boxplot.png")
    plt.savefig(box_plot_filename)
    print(f"Completion length box plot saved to {box_plot_filename}")
    plt.close()

def analyze_skewness_per_batch(directory, timestamp_suffix=None, batch_size=72, num_devices=3):
    """
    Calculates the skewness of completion lengths for each batch of data
    and plots a histogram of these skewness values.

    A "batch" is defined as `batch_size` completions, distributed across
    `num_devices` CSV files (e.g., 72 completions across 3 files means 24 per file).
    
    If `timestamp_suffix` is provided, only files matching that specific timestamp
    prefix (YYYYMMDD_HHMM) will be processed. Otherwise, all detected timestamps
    will be processed.

    Args:
        directory (str): The path to the directory containing the CSV files.
        timestamp_suffix (str, optional): The specific timestamp suffix prefix (e.g., "20250605_1148")
                                          to filter the CSV files. If None, all timestamps are processed.
        batch_size (int): The total number of completions in a logical batch.
        num_devices (int): The number of devices (and thus CSV files) that
                           contribute to a single batch.
    """
    print(f"\n--- Analyzing Skewness of Completion Lengths per Batch ---")
    if timestamp_suffix:
        print(f"--- Filtering by timestamp prefix: {timestamp_suffix} ---")
    
    if batch_size % num_devices != 0:
        print(f"Warning: batch_size ({batch_size}) is not evenly divisible by num_devices ({num_devices}). "
              f"This might lead to uneven distribution of completions per file.")
    
    completions_per_file = batch_size // num_devices
    
    # Group files by timestamp (representing a unique batch)
    file_groups = {}
    for filename in os.listdir(directory):
        if filename.startswith("prompt_completions_") and filename.endswith(".csv"):
            # Example filename: prompt_completions_20250605_114856_rank0.csv
            # Match YYYYMMDD_HHMM, ignoring the seconds part
            match = re.match(r"prompt_completions_(\d{8}_\d{4})\d{2}_rank(\d+)\.csv", filename)
            if match:
                timestamp_prefix = match.group(1) # This will be like "20250605_1148"
                
                # Only process files if no timestamp_suffix is specified, or if it matches the current file's timestamp prefix
                if timestamp_suffix is None or timestamp_prefix == timestamp_suffix:
                    rank = int(match.group(2))
                    # Use timestamp_prefix (YYYYMMDD_HHMM) as the key for grouping,
                    # as different ranks starting at the same minute might have different seconds.
                    if timestamp_prefix not in file_groups:
                        file_groups[timestamp_prefix] = {}
                    file_groups[timestamp_prefix][rank] = os.path.join(directory, filename)
            else:
                print(f"Skipping file with unexpected name format: {filename}")

    all_batch_skewness = []

    # Process each group (which represents a single logical timestamp, potentially containing multiple batches)
    # Sort by timestamp to ensure consistent order
    for timestamp, files_by_rank in sorted(file_groups.items()):
        # Check if we have files for all expected ranks for this timestamp
        # This ensures we only process complete sets of files for a given timestamp
        if len(files_by_rank) < num_devices:
            print(f"Skipping timestamp {timestamp}: Found {len(files_by_rank)} files, expected {num_devices}. Incomplete set of files for this timestamp.")
            continue
        
        # Store completion token series for each rank's file
        completion_series_by_rank = {}
        
        # Sort files by rank to ensure consistent processing order (e.g., rank0, rank1, rank2)
        sorted_ranks = sorted(files_by_rank.keys())
        
        # First, load all relevant data from each file for the current timestamp
        for rank in sorted_ranks:
            file_path = files_by_rank[rank]
            try:
                df = pd.read_csv(file_path)
                if 'Completion Tokens' in df.columns:
                    # Convert to numeric, coercing errors to NaN, then drop NaNs
                    lengths = pd.to_numeric(df['Completion Tokens'], errors='coerce').dropna()
                    completion_series_by_rank[rank] = lengths
                else:
                    print(f"Warning: 'Completion Tokens' column not found in {file_path}. Skipping data from this file for timestamp {timestamp}.")
            except Exception as e:
                print(f"Error reading {file_path}: {e}. Skipping data from this file for timestamp {timestamp}.")
        
        # Ensure we have data for all expected ranks after loading
        if len(completion_series_by_rank) < num_devices:
            print(f"Skipping timestamp {timestamp}: Not all expected ranks ({num_devices}) had valid 'Completion Tokens' data. Found {len(completion_series_by_rank)}.")
            continue

        # Determine the maximum number of full 'completions_per_file' chunks available across all files
        # This ensures we don't try to read beyond the shortest file's data
        min_rows_across_files = min(len(s) for s in completion_series_by_rank.values())
        num_full_chunks_per_file = min_rows_across_files // completions_per_file

        if num_full_chunks_per_file == 0:
            print(f"Warning: Not enough completions in files for timestamp {timestamp} to form even one chunk of {completions_per_file} per file. Skipping skewness calculation for this timestamp.")
            continue

        # Iterate through the data, forming batches by taking 'completions_per_file' from each rank's file
        for i in range(num_full_chunks_per_file):
            current_batch_lengths = []
            
            # Collect completions for the current batch from each rank's series
            for rank in sorted_ranks:
                start_idx = i * completions_per_file
                end_idx = start_idx + completions_per_file
                
                # Extract the chunk for the current batch from this rank's series
                chunk = completion_series_by_rank[rank].iloc[start_idx:end_idx].tolist()
                current_batch_lengths.extend(chunk)
            
            # Verify the collected batch size
            if len(current_batch_lengths) != batch_size:
                # This should ideally not happen if num_full_chunks_per_file is calculated correctly
                # and all files have at least min_rows_across_files.
                print(f"Internal Warning: Collected batch {i+1} for timestamp {timestamp} has {len(current_batch_lengths)} completions, expected {batch_size}. Skipping.")
                continue

            # Calculate skewness only if we have exactly the expected number of completions for a full batch
            # Skewness is undefined for constant data (e.g., all lengths are the same)
            if len(set(current_batch_lengths)) > 1: 
                batch_skew = skew(current_batch_lengths)
                all_batch_skewness.append(batch_skew)
            else:
                print(f"Warning: All completion lengths in batch {i+1} for timestamp {timestamp} are identical. Skewness is undefined. Skipping this batch.")

    if not all_batch_skewness:
        print(f"No complete batches found or no skewness could be calculated for any batch in '{directory}'"
              f"{f' for timestamp prefix {timestamp_suffix}' if timestamp_suffix else ''}.")
        return
    # Plotting the histogram of skewness values
    # Increase figure size slightly for better readability and set a higher DPI for saving
    plt.figure(figsize=(12, 8)) 
    # To make the histogram bins denser, increase the number of bins.
    # 'auto' lets matplotlib decide, but a fixed number like 50 or 100 provides more granularity.
    plt.hist(all_batch_skewness, bins=50, edgecolor='black', alpha=0.7) 
    
    # Calculate mean and median skewness
    mean_skewness = pd.Series(all_batch_skewness).mean()
    median_skewness = pd.Series(all_batch_skewness).median()

    # Add lines for mean and median to better understand distribution
    # Increase font size for legend labels
    plt.axvline(x=mean_skewness, color='g', linestyle=':', label=f'Mean: {mean_skewness:.2f}')
    plt.axvline(x=median_skewness, color='r', linestyle='--', label=f'Median: {median_skewness:.2f}')
    plt.legend(fontsize=16) # Increase legend font size

    # Use the provided timestamp_suffix for the plot title if filtered, otherwise indicate all batches
    plot_title_suffix = f" for Timestamp Prefix: {timestamp_suffix}" if timestamp_suffix else ""
    # Increase font size for title and labels
    plt.title(f"Distribution of Skewness of Completion Lengths per Batch (Total: {len(all_batch_skewness)} batches)", fontsize=18)
    plt.xlabel("Skewness Value", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.grid(axis='y', alpha=0.75)
    
    # Improve tick label font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Use the provided timestamp_suffix for the filename if filtered
    plot_filename_suffix = f"_{timestamp_suffix}" if timestamp_suffix else ""
    plot_filename = os.path.join(directory, f"batch_skewness_distribution_histogram{plot_filename_suffix}.png")
    # Save the plot with a higher DPI for better clarity
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight') # Added dpi=300 and bbox_inches='tight'
    print(f"Batch skewness distribution histogram saved to {plot_filename}")
    plt.close()


import random
from collections import defaultdict
from typing import List, Tuple


def analyze_random_batch_completion_lengths(directory: str, 
                                            timestamp_suffix: str,
                                            batch_size: int = 72, 
                                            skewness_ranges: List[Tuple[float, float]] = None
                                            ):
    """
    Analyzes the distribution of completion lengths for conceptual batch segments,
    selecting one batch from specific skewness ranges for visualization with violin plots.

    Args:
        directory (str): The directory containing the CSV files (e.g., "Qwen2-0.5B-GRPO").
        timestamp_suffix (str): A specific timestamp prefix (e.g., "20250605_1148")
                                to filter the CSV files.
        batch_size (int): The total number of completions that constitute a conceptual batch.
        skewness_ranges (List[Tuple[float, float]], optional): A list of tuples defining the skewness intervals to categorize batches. For example, [(0.0, 1.0), (1.0, 2.0)]. The last interval's upper bound is inclusive, others are exclusive (e.g., [lower, upper) for all but the last, which is [lower, upper]). Defaults to [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)].
    """
    if skewness_ranges is None:
        skewness_ranges = [
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
            (3.0, 4.0),
            (4.0, 5.0)
        ]

    # Stores data for the single specified timestamp, grouped by rank: [(rank, completion_lengths_list), ...]
    rank_data_for_current_timestamp = [] 
    
    # Glob pattern to match files for the specific timestamp prefix
    csv_files_pattern = os.path.join(directory, f"prompt_completions_{timestamp_suffix}*_rank*.csv")

    found_files = glob.glob(csv_files_pattern)
    if not found_files:
        print(f"No CSV files matching '{csv_files_pattern}' found in '{directory}'.")
        return

    print(f"Found {len(found_files)} CSV files for timestamp '{timestamp_suffix}': {sorted(found_files)}")

    filename_pattern = re.compile(r"prompt_completions_(\d{8}_\d{6})_rank(\d+)\.csv$")

    for filepath in sorted(found_files):
        match = filename_pattern.search(filepath)
        if not match:
            print(f"Warning: Filename '{os.path.basename(filepath)}' does not match expected pattern. Skipping.")
            continue
        
        rank = match.group(2)  # This is the rank (e.g., '0', '1', '2')

        try:
            df = pd.read_csv(filepath)
            if 'Completion Tokens' in df.columns:
                completion_lengths = df['Completion Tokens'].tolist()
                if completion_lengths:
                    rank_data_for_current_timestamp.append((rank, completion_lengths))
                else:
                    print(f"Warning: 'Completion Tokens' column in {filepath} is empty. Skipping.")
            else:
                print(f"Warning: 'Completion Tokens' column not found in {filepath}. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"Warning: {filepath} is empty. Skipping.")
        except FileNotFoundError:  # Should not happen with glob, but good for robustness
            print(f"Error: {filepath} not found. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while reading {filepath}: {e}. Skipping.")

    if not rank_data_for_current_timestamp:
        print(f"No completion lengths found for timestamp '{timestamp_suffix}' across all files for analysis.")
        return

    num_files_in_batch = len(rank_data_for_current_timestamp)
    print(f"Collected data from {num_files_in_batch} rank files for timestamp '{timestamp_suffix}'.")

    # Calculate how many completions to sample from each file for this batch segment
    completions_per_file = batch_size // num_files_in_batch
    
    if completions_per_file == 0:
        print(f"Warning: Batch size ({batch_size}) is too small relative to number of files ({num_files_in_batch}) "
              f"to sample any completions per file. Cannot form conceptual batches. Skipping analysis.")
        return

    # Find the minimum length among all files for this timestamp
    min_file_length = float('inf')
    for _, lengths in rank_data_for_current_timestamp:
        if not lengths: # Handle empty lists within rank_data_for_current_timestamp
            min_file_length = 0
            break
        min_file_length = min(min_file_length, len(lengths))
    
    if min_file_length == 0:
        print(f"Warning: At least one file for timestamp prefix '{timestamp_suffix}' is empty or has no completions. Skipping analysis.")
        return

    # Determine how many full conceptual batch segments can be formed from this timestamp group
    num_total_conceptual_batches = min_file_length // completions_per_file
    
    if num_total_conceptual_batches == 0:
        print(f"Warning: Files for timestamp prefix '{timestamp_suffix}' are too short ({min_file_length} tokens min) "
              f"to form a full segment of {completions_per_file} completions per file. Skipping analysis.")
        return

    print(f"Found {num_total_conceptual_batches} batches of size {batch_size}.")

    # Store all conceptual batches and their skewness
    # List of (skewness, segment_index, completion_lengths_list)
    all_conceptual_batches_with_skewness = [] 

    for segment_index in range(num_total_conceptual_batches):
        start_index = segment_index * completions_per_file
        end_index = start_index + completions_per_file

        current_conceptual_batch_lengths = []
        for rank, file_lengths in rank_data_for_current_timestamp:
            # Extract the contiguous block of completions from each rank's data
            current_conceptual_batch_lengths.extend(file_lengths[start_index:end_index])
        
        # Ensure the collected batch has the expected size before calculating skewness
        if len(current_conceptual_batch_lengths) == batch_size:
            # Calculate skewness only if there's variation in lengths (skewness is undefined for constant data)
            if len(set(current_conceptual_batch_lengths)) > 1: 
                batch_skew = skew(current_conceptual_batch_lengths)
                all_conceptual_batches_with_skewness.append((batch_skew, segment_index, current_conceptual_batch_lengths))
            else:
                print(f"Warning: All completion lengths in conceptual batch {segment_index+1} for timestamp {timestamp_suffix} are identical. Skewness is undefined. Skipping this batch for skewness categorization.")
        else:
            print(f"Warning: Conceptual batch {segment_index+1} for timestamp {timestamp_suffix} has {len(current_conceptual_batch_lengths)} completions, expected {batch_size}. Skipping for skewness analysis.")

    if not all_conceptual_batches_with_skewness:
        print(f"No conceptual batches found with calculable skewness for timestamp prefix '{timestamp_suffix}'.")
        return

    # Store batches categorized by their skewness range
    # Key: tuple (lower_bound, upper_bound), Value: list of (skewness, segment_index, lengths)
    categorized_batches = defaultdict(list) 

    for batch_skew, segment_index, lengths in all_conceptual_batches_with_skewness:
        assigned = False
        for i, (lower, upper) in enumerate(skewness_ranges):
            if i == len(skewness_ranges) - 1: # Last range, upper bound is inclusive
                if lower <= batch_skew <= upper:
                    categorized_batches[(lower, upper)].append((batch_skew, segment_index, lengths))
                    assigned = True
                    break
            else: # Other ranges, upper bound is exclusive
                if lower <= batch_skew < upper:
                    categorized_batches[(lower, upper)].append((batch_skew, segment_index, lengths))
                    assigned = True
                    break
        if not assigned:
            # Batches with skewness outside the specified ranges are not categorized for this plot.
            pass 

    # Select one random batch from each category
    selected_batches_for_plotting = [] # List of (label, completion_lengths_list)
    
    # Iterate through the provided skewness_ranges to ensure categories are processed in order
    for lower, upper in skewness_ranges:
        category_key = (lower, upper)
        batches_in_category = categorized_batches.get(category_key)
        
        if batches_in_category:
            # Randomly select one batch from this category
            selected_batch_info = random.choice(batches_in_category)
            selected_skew, selected_segment_index, selected_lengths = selected_batch_info
            
            # Format label for the plot
            # Use .0f for integer bounds as per prompt, .2f for actual skewness
            label = f"Batch {selected_segment_index}: {selected_skew:.2f} ([{lower:.0f}, {upper:.0f}])"
            selected_batches_for_plotting.append((label, selected_lengths))
            print(f"Selected batch {selected_segment_index} (skew={selected_skew:.2f}) for range [{lower:.0f}, {upper:.0f}].")
        else:
            print(f"No batches found for skewness range [{lower:.0f}, {upper:.0f}].")

    if not selected_batches_for_plotting:
        print("No batches selected for plotting after skewness categorization.")
        return

    # Prepare data for violin plot
    plot_data = [lengths for label, lengths in selected_batches_for_plotting]
    x_labels = [label for label, lengths in selected_batches_for_plotting]

    # Plotting the violin plot
    plt.figure(figsize=(16, 9)) # Increased figure size for better visualization, especially with long labels
    
    # Create the violin plot
    violin_parts = plt.violinplot(plot_data, showmeans=True, showmedians=True, showextrema=True)

    # Customize the appearance of the violin plot elements
    # Define a list of distinct colors for different violins
    colors = ['#D43F3A', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)]) # Assign a different color to each violin body, cycling through the list
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customize mean, median, and extrema lines
    violin_parts['cmeans'].set_edgecolor('green')
    violin_parts['cmeans'].set_linewidth(2)
    violin_parts['cmedians'].set_edgecolor('blue')
    violin_parts['cmedians'].set_linewidth(2)
    violin_parts['cmins'].set_edgecolor('gray')
    violin_parts['cmaxes'].set_edgecolor('gray')
    violin_parts['cbars'].set_edgecolor('gray')

    # Set x-axis tick labels
    plt.xticks(range(1, len(plot_data) + 1), x_labels, fontsize=14)
    
    plt.title(f"Completion Length Distribution for Selected Batches by Skewness", fontsize=22)
    plt.xlabel("Batch Index: Skewness (Range)", fontsize=18)
    plt.ylabel("Completion Length (Tokens)", fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.yticks(fontsize=16)

    # Add a legend for mean/median
    plt.plot([], [], color='green', linestyle='-', linewidth=2, label='Mean')
    plt.plot([], [], color='blue', linestyle='-', linewidth=2, label='Median')
    # Place legend outside the plot area to prevent overlap
    plt.legend(fontsize=18, loc='upper right')

    # Sanitize timestamp_suffix for filename as it's always present
    sanitized_timestamp_suffix = timestamp_suffix.replace(':', '_').replace('.', '_')
    plot_filename_suffix = f"_{sanitized_timestamp_suffix}"
    plot_filename = os.path.join(directory, f"skewness_categorized_batch_segments_completion_lengths_violin{plot_filename_suffix}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Skewness-categorized batch segments completion lengths violin plot saved to {plot_filename}")
    plt.close()


if __name__ == "__main__":
    # The directory where the CSV files are located
    target_directory = "Qwen2-0.5B-GRPO" 
    # The batch size as described by the user (72 completions across 3 files)
    # The function will collect all available completions from the specified files
    # and analyze their overall distribution, treating the collected set as the "batch" for analysis.
    # analyze_completion_lengths(target_directory, batch_size=72)
    analyze_skewness_per_batch(target_directory, "20250605_1148")
    analyze_random_batch_completion_lengths(target_directory, "20250605_1148")
