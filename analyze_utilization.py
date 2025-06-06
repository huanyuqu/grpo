import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_gpu_utilization(filename: str, num_gpus: int = 3):
    """
    Analyzes SM (Streaming Multiprocessor) and memory utilization from a GPU dmon log CSV file.
    It plots individual GPU utilization trends and an overall average trend over time.
    Separate plots are generated for SM utilization and Memory utilization.

    Args:
        filename (str): The path to the GPU dmon log CSV file (e.g., gpu_dmon_log.csv).
        num_gpus (int): The expected number of GPUs. This is used for initial validation
                        and informational purposes, but the actual number of GPUs plotted
                        is derived from the 'idx' column in the data. Defaults to 3.
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    if num_gpus <= 0:
        print("Error: num_gpus must be a positive integer.")
        return

    # Set global font size for all text elements in plots to make them larger
    plt.rcParams.update({'font.size': 12}) # Base font size for general text, including tick labels

    try:
        # Step 1: Read the first line to get the actual column headers.
        # The header line starts with '#', so we need to strip it.
        with open(filename, 'r') as f:
            header_line = f.readline().strip()
            # Clean the header by removing '#' and splitting by comma
            raw_columns = header_line.lstrip('#').split(',')
            # Further strip whitespace from each column name
            cleaned_columns = [col.strip() for col in raw_columns]

        # Step 2: Read the rest of the CSV data, skipping the first two lines (header and units line).
        # Assign the cleaned column names.
        df = pd.read_csv(filename, skiprows=2, header=None, names=cleaned_columns)

        # Convert 'Time' to datetime objects. Ensure no extra whitespace.
        # 'errors='coerce'' will turn non-parsable values into NaT (Not a Time).
        df['Time'] = pd.to_datetime(df['Time'].astype(str).str.strip(), format='%H:%M:%S', errors='coerce')

        # Drop rows where 'Time' conversion resulted in NaT
        df.dropna(subset=['Time'], inplace=True)

        # Convert 'sm' and 'mem' utilization columns to numeric.
        # 'errors='coerce'' will turn non-numeric values (like '-') into NaN.
        df['sm'] = pd.to_numeric(df['sm'], errors='coerce')
        df['mem'] = pd.to_numeric(df['mem'], errors='coerce')

        # Drop rows where 'sm' or 'mem' conversion resulted in NaN
        df.dropna(subset=['sm', 'mem'], inplace=True)

        df['gpu'] = pd.to_numeric(df['gpu'], errors='coerce')
        df.dropna(subset=['gpu'], inplace=True)
        df['gpu'] = df['gpu'].astype(int) # Convert to integer GPU IDs

        if df.empty:
            print(f"No valid utilization data found in '{filename}' after cleaning.")
            return

        print(f"\n--- Analyzing GPU Utilization from {filename} ---")

        # Calculate elapsed time in seconds, starting from 0 for the first data point
        min_time = df['Time'].min()
        df['Elapsed_Seconds'] = (df['Time'] - min_time).dt.total_seconds()

        # Get unique GPU IDs present in the data
        unique_gpus = sorted(df['gpu'].unique())
        print(f"Found {len(unique_gpus)} GPUs in the log")

        # Calculate overall average utilization per timestamp across all GPUs
        # This groups by 'Time' and averages 'sm' and 'mem' for all GPUs at that specific timestamp.
        avg_util_df = df.groupby('Time').agg(
            avg_sm=('sm', 'mean'),
            avg_mem=('mem', 'mean')
        ).reset_index()

        if avg_util_df.empty:
            print(f"No aggregated utilization data found in '{filename}' after processing.")
            return
        
        # Calculate elapsed time for the aggregated data as well, using the same min_time
        avg_util_df['Elapsed_Seconds'] = (avg_util_df['Time'] - min_time).dt.total_seconds()

        # Define a color palette for individual GPUs to ensure higher distinctiveness
        # 'tab10' provides 10 distinct colors, suitable for a moderate number of GPUs.
        # If more than 10 GPUs are expected, consider a different colormap like 'viridis'
        # or 'plasma' and map GPU IDs to colors using a normalizer.
        gpu_colors = plt.cm.tab10.colors

        # --- Plotting SM Utilization (Individual GPUs + Overall Average) ---
        plt.figure(figsize=(12, 6))
        for i, gpu_id in enumerate(unique_gpus):
            gpu_df = df[df['gpu'] == gpu_id]
            # Assign a color from the palette, cycling if there are more GPUs than colors
            color = gpu_colors[i % len(gpu_colors)]
            plt.plot(gpu_df['Elapsed_Seconds'], gpu_df['sm'], label=f'GPU {gpu_id} SM', alpha=0.7, color=color)

        plt.plot(avg_util_df['Elapsed_Seconds'], avg_util_df['avg_sm'], label='Overall Average SM', color='black', linestyle='--', linewidth=2)
        # Increase font sizes for title, labels, and legend
        plt.title(f'SM Utilization Over Time (Individual GPUs and Overall Average)', fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=14) # Changed x-axis label to seconds
        plt.ylabel('SM Utilization (%)', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()

        sm_plot_filename = os.path.join(os.path.dirname(filename), "sm_utilization_plot.png")
        # Increase DPI for higher resolution
        plt.savefig(sm_plot_filename, dpi=300)
        print(f"SM utilization plot saved to {sm_plot_filename}")
        plt.close() # Close the figure to free memory

        # --- Plotting Memory Utilization (Individual GPUs + Overall Average) ---
        plt.figure(figsize=(12, 6))
        for i, gpu_id in enumerate(unique_gpus):
            gpu_df = df[df['gpu'] == gpu_id]
            # Assign a color from the palette, cycling if there are more GPUs than colors
            color = gpu_colors[i % len(gpu_colors)]
            plt.plot(gpu_df['Elapsed_Seconds'], gpu_df['mem'], label=f'GPU {gpu_id} Memory', alpha=0.7, color=color)

        plt.plot(avg_util_df['Elapsed_Seconds'], avg_util_df['avg_mem'], label='Overall Average Memory', color='black', linestyle='--', linewidth=2)
        # Increase font sizes for title, labels, and legend
        plt.title(f'Memory Utilization Over Time (Individual GPUs and Overall Average)', fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=14) # Changed x-axis label to seconds
        plt.ylabel('Memory Utilization (%)', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()

        mem_plot_filename = os.path.join(os.path.dirname(filename), "mem_utilization_plot.png")
        # Increase DPI for higher resolution
        plt.savefig(mem_plot_filename, dpi=300)
        print(f"Memory utilization plot saved to {mem_plot_filename}")
        plt.close() # Close the figure to free memory

    except Exception as e:
        print(f"An error occurred while analyzing '{filename}': {e}")

if __name__ == "__main__":
    # Example usage:
    # Assuming 'gpu_dmon_log.csv' is in the same directory as this script.
    # If the file is in a different location, provide the full path.
    # The example context shows 3 GPUs, so we'll use num_gpus=3.
    # gpu_dmon_log_20250603_1742
    # gpu_dmon_log_20250603_2026
    analyze_gpu_utilization("gpu_dmon_log_20250603_2026.csv", num_gpus=3)
