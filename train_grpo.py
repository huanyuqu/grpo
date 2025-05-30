from datasets import load_dataset
from trl import GRPOConfig
from grpo_trainer import MyGRPOTrainer
import re
import csv
import os
import time
from transformers import AutoTokenizer
from accelerate import PartialState


def add_instruction_to_prompt(examples):
    # Define the instruction you want to add before each prompt
    # This instruction guides the model to generate text in the desired format
    instruction = "Please act as a math problem solver and solve the following problem. Put the final numerical answer after '#### '.\n"
    examples['prompt'] = [instruction + question for question in examples['question']]
    return examples

state = PartialState()

model_path = './models/Qwen2-0.5B-Instruct'
output_dir = 'Qwen2-0.5B-GRPO'

os.makedirs(output_dir, exist_ok=True)
rank = state.process_index   
base_csv_file_path = os.path.join(output_dir, "prompt_completions") 
csv_file_path = f"{base_csv_file_path}_rank{rank}.csv"
csv_file = open(csv_file_path, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
header = ['Prompt', 'Completion', 'Prompt Tokens', 'Completion Tokens', 'Predicted Answer (Numeric)', 'Reward', 'Ground Truth Text', 'Correct Answer (Numeric)']
csv_writer.writerow(header)
csv_file.flush()
if state.is_main_process:
    print(f"Logging prompt-completion pairs to {csv_file_path} (and potentially other rank files)")

profile_csv_file_path = f'{output_dir}/profile.csv'
profile_csv_file = open(profile_csv_file_path, 'w', newline='', encoding='utf-8')
profile_csv_writer = csv.writer(profile_csv_file)
profile_header = ['stage', 'step', 'device_id', 'start_time', 'end_time', 'stage_time']
profile_csv_writer.writerow(profile_header)
profile_csv_file.flush()
if state.is_main_process:
    print(f"Logging profile data to {profile_csv_file_path}")

dataset = load_dataset("./data/gsm8k", "main", split="train")
dataset = dataset.map(add_instruction_to_prompt, batched=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# GSM8k answers typically end with "#### <number>"
def parse_gsm8k_answer(text):
    match = re.search(r"#### (\d+)", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    else:
        # If the primary format "#### <number>" is not found, print a warning
        # print(f"Warning: Expected format '#### <number>' not found")
        # Fallback: try to find the last number in the text
        # This is less reliable but might catch some cases if the model doesn't follow the format
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                return int(numbers[-1])
            except ValueError:
                return None
        return None


# Define the reward function for GSM8k
# This function rewards completions that produce the correct final numerical answer
def reward_gsm8k(completions, prompts, **kwargs):
    rewards = []
    # The GRPO trainer passes the original batch data in kwargs
    ground_truth_answers = kwargs.get('answer')

    log_data = []

    if ground_truth_answers is None or len(ground_truth_answers) != len(completions):
        print("Warning: Ground truth answers not available or mismatching completions. Assigning 0 reward.")
        rewards = [0.0] * len(completions)

        for i in range(len(completions)):
             prompt = prompts[i] if prompts and i < len(prompts) else ""
             completion = completions[i]

             prompt_tokens = len(tokenizer.encode(prompt)) if tokenizer and prompt else 0
             completion_tokens = len(tokenizer.encode(completion)) if tokenizer and completion else 0

             log_data.append({
                 'prompt': prompt,
                 'completion': completion,
                 'prompt_tokens': prompt_tokens,
                 'completion_tokens': completion_tokens,
                 'predicted_answer': parse_gsm8k_answer(completion),
                 'reward': 0.0,
                 'ground_truth_text': None,
                 'correct_answer': None
             })
    else:
        for prompt, completion, gt_answer_text in zip(prompts, completions, ground_truth_answers):
            predicted_answer = parse_gsm8k_answer(completion)
            correct_answer = parse_gsm8k_answer(gt_answer_text)

            # Reward is 1.0 if the predicted answer matches the correct answer, 0.0 otherwise
            reward = 1.0 if predicted_answer is not None and correct_answer is not None and predicted_answer == correct_answer else 0.0
            rewards.append(reward)

            prompt_tokens = len(tokenizer.encode(prompt)) if tokenizer and prompt else 0
            completion_tokens = len(tokenizer.encode(completion)) if tokenizer and completion else 0

            log_data.append({
                'prompt': prompt,
                'completion': completion,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'predicted_answer': predicted_answer,
                'reward': reward,
                'ground_truth_text': gt_answer_text,
                'correct_answer': correct_answer
            })
    
    if log_data:
        try:
            with open(csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for item in log_data:
                    prompt_str = str(item['prompt']) if item['prompt'] is not None else ""
                    completion_str = str(item['completion']) if item['completion'] is not None else ""
                    prompt_tokens_str = str(item['prompt_tokens'])
                    completion_tokens_str = str(item['completion_tokens'])
                    predicted_answer_str = str(item['predicted_answer']) if item['predicted_answer'] is not None else ""
                    reward_str = str(item['reward'])
                    ground_truth_text_str = str(item['ground_truth_text']) if item['ground_truth_text'] is not None else ""
                    correct_answer_str = str(item['correct_answer']) if item['correct_answer'] is not None else ""

                    writer.writerow([prompt_str, completion_str, prompt_tokens_str, 
                                     completion_tokens_str, predicted_answer_str, 
                                     reward_str, ground_truth_text_str, 
                                     correct_answer_str])
                f.flush()
        except Exception as e:
            print(f"Error occurred while writing to CSV file {csv_file_path}: {e}")

    return rewards

training_args = GRPOConfig(output_dir=output_dir, logging_steps=10,
                           num_generations=2,
                           per_device_train_batch_size=1,
                           gradient_accumulation_steps=4,
                           fp16=True,
                           max_prompt_length=1024,
                           max_completion_length=1024,
                           num_train_epochs=5)

trainer = MyGRPOTrainer(
    model=model_path,
    reward_funcs=reward_gsm8k,
    args=training_args,
    train_dataset=dataset,
    profile_file=profile_csv_file_path
)

start = time.time()
trainer.train()
end = time.time()
total_time = end - start
print(f"RL time: [{start:.4f}, {end:.4f} {total_time:.4f}s")
