import matplotlib.pyplot as plt
import re
import os
import ast

def analyze_reward(filename: str):
    """
    Reads a file, extracts reward values from dictionary logs, and plots them.

    Args:
        filename (str): The path to the file containing reward logs.
    """
    rewards = []
    # Regex to find a dictionary string within the line
    # This regex captures the content between the first '{' and the last '}' on the line.
    # Assumes the relevant dictionary is the primary or only dictionary structure per line.
    dict_pattern = re.compile(r'(\{.*\})')

    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    try:
        with open(filename, 'r') as f:
            for line in f:
                match = dict_pattern.search(line)
                if match:
                    dict_str = match.group(1)
                    try:
                        # Safely evaluate the dictionary string found by the regex
                        log_data = ast.literal_eval(dict_str)

                        # Check if the evaluated result is a dictionary and contains the 'reward' key
                        if isinstance(log_data, dict) and 'reward' in log_data:
                            reward_value = log_data['reward']
                            # Ensure the reward value is a number before appending
                            if isinstance(reward_value, (int, float)):
                                rewards.append(float(reward_value))
                            else:
                                # Optional: Log a warning if 'reward' key exists but value is not numeric
                                # print(f"Warning: 'reward' value is not a number ({type(reward_value)}) in line: {line.strip()}")
                                pass # Silently skip non-numeric reward values
                        # Optional: Log if a dictionary was found but it didn't contain the 'reward' key
                        # else:
                        #     print(f"Info: Dictionary found but no 'reward' key in line: {line.strip()}")

                    except (ValueError, SyntaxError):
                        # Catch errors during ast.literal_eval (e.g., if the matched string isn't a valid literal)
                        # print(f"Warning: Could not parse dictionary from string: '{dict_str}' in line: {line.strip()}")
                        pass # Silently skip lines where dictionary parsing fails

                    except Exception as e:
                        # Catch any other unexpected errors during processing a line
                        print(f"An unexpected error occurred processing line: {line.strip()} - {e}")


    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if not rewards:
        print(f"No lines containing a valid dictionary with a numeric 'reward' key found in '{filename}'.")
        return

    # Generate x-axis values, multiplying by 10 as reward is output every 10 steps.
    # The first reward corresponds to step 10, the second to step 20, and so on.
    x_steps = [(i + 1) * 10 for i in range(len(rewards))]

    # Plot the rewards
    plt.figure(figsize=(10, 6))
    plt.plot(x_steps, rewards) # Use the scaled x_steps for the x-axis
    plt.title("Reward over Steps")
    plt.xlabel("Step Number") # The label remains "Step Number", but values are now scaled
    plt.ylabel("Reward Value")
    plt.grid(True)

    # Save the plot to a file instead of showing it
    plot_filename = "reward_plot.png"
    plt.savefig(plot_filename)
    print(f"Reward plot saved to {plot_filename}")
    plt.close() # Close the figure to free memory

def analyze_completion_length_over_steps(filename):
    """
    Analyzes the 'completions/mean_length' metric from the log file
    and plots its trend over training steps.

    Args:
        filename (str): The path to the log file (e.g., 'output.out').
    """
    mean_lengths = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Regex to find dictionary-like strings
                match = re.search(r"\{.*\}", line)
                if match:
                    dict_str = match.group(0)
                    try:
                        # Safely evaluate the string as a Python literal (dictionary)
                        log_data = ast.literal_eval(dict_str)
                        # Check if the evaluated result is a dictionary and contains the 'completions/mean_length' key
                        if isinstance(log_data, dict) and 'completions/mean_length' in log_data:
                            mean_length_value = log_data['completions/mean_length']
                            # Ensure the mean length value is a number before appending
                            if isinstance(mean_length_value, (int, float)):
                                mean_lengths.append(float(mean_length_value))
                            else:
                                # Silently skip non-numeric mean length values
                                pass 
                    except (ValueError, SyntaxError):
                        # Silently skip lines where dictionary parsing fails
                        pass 
                    except Exception as e:
                        # Catch any other unexpected errors during processing a line
                        print(f"An unexpected error occurred processing line: {line.strip()} - {e}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if not mean_lengths:
        print(f"No lines containing a valid dictionary with a numeric 'completions/mean_length' key found in '{filename}'.")
        return

    # Generate x-axis values, multiplying by 10 as metrics are output every 10 steps.
    # The first mean length corresponds to step 10, the second to step 20, and so on.
    x_steps = [(i + 1) * 10 for i in range(len(mean_lengths))]

    # Plot the mean completion lengths
    plt.figure(figsize=(10, 6))
    plt.plot(x_steps, mean_lengths) # Use the scaled x_steps for the x-axis
    plt.title("Mean Completion Length over Steps")
    plt.xlabel("Step Number") # The label remains "Step Number", but values are now scaled
    plt.ylabel("Mean Completion Length (tokens)")
    plt.grid(True)

    # Save the plot to a file instead of showing it
    plot_filename = "mean_completion_length_plot.png"
    plt.savefig(plot_filename)
    print(f"Mean completion length plot saved to {plot_filename}")
    plt.close() # Close the figure to free memory

def analyze_max_completion_length_over_steps(filename):
    """
    Analyzes a log file to extract and plot the 'completions/max_length' metric over training steps.

    Args:
        filename (str): The path to the log file to analyze.
    """
    max_lengths = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Regex to find dictionary-like strings
                match = re.search(r"\{.*\}", line)
                if match:
                    dict_str = match.group(0)
                    try:
                        # Safely evaluate the string as a Python literal (dictionary)
                        log_data = ast.literal_eval(dict_str)
                        # Check if the evaluated result is a dictionary and contains the 'completions/max_length' key
                        if isinstance(log_data, dict) and 'completions/max_length' in log_data:
                            max_length_value = log_data['completions/max_length']
                            # Ensure the max length value is a number before appending
                            if isinstance(max_length_value, (int, float)):
                                max_lengths.append(float(max_length_value))
                            else:
                                # Silently skip non-numeric max length values
                                pass
                    except (ValueError, SyntaxError):
                        # Silently skip lines where dictionary parsing fails
                        pass
                    except Exception as e:
                        # Catch any other unexpected errors during processing a line
                        print(f"An unexpected error occurred processing line: {line.strip()} - {e}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if not max_lengths:
        print(f"No lines containing a valid dictionary with a numeric 'completions/max_length' key found in '{filename}'.")
        return

    # Generate x-axis values, multiplying by 10 as metrics are output every 10 steps.
    # The first max length corresponds to step 10, the second to step 20, and so on.
    x_steps = [(i + 1) * 10 for i in range(len(max_lengths))]

    # Plot the max completion lengths
    plt.figure(figsize=(10, 6))
    plt.plot(x_steps, max_lengths) # Use the scaled x_steps for the x-axis
    plt.title("Max Completion Length over Steps")
    plt.xlabel("Step Number") # The label remains "Step Number", but values are now scaled
    plt.ylabel("Max Completion Length (tokens)")
    plt.grid(True)

    # Save the plot to a file instead of showing it
    plot_filename = "max_completion_length_plot.png"
    plt.savefig(plot_filename)
    print(f"Max completion length plot saved to {plot_filename}")
    plt.close() # Close the figure to free memory


if __name__ == "__main__":
    # Call the analysis function with the specified filename
    analyze_reward("output_20250605_1148.out")
    analyze_completion_length_over_steps("output_20250605_1148.out")
