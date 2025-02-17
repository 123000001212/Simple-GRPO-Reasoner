# train.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re
import random

train_dataset = load_dataset("parquet", data_files={'train': './data/x_r1_750/train.parquet', 'test': './data/x_r1_750/test.parquet'}, split='train')
eval_dataset = load_dataset("parquet", data_files={'train': './data/x_r1_750/train.parquet', 'test': './data/x_r1_750/test.parquet'}, split='test')

def compute_reward(solution_str, ground_truth):
    overall_score = 0
    do_print = random.randint(1,64)==1
    if do_print: print(solution_str)

    # check format
    pattern = r"\\boxed{(.*?)}"
    match = re.search(pattern, solution_str)
    if match:
        overall_score += 0.1
    else: 
        if do_print: print("format error")
        return 0
    
    # check answer
    answer = match.group(1).strip()
    if answer == ground_truth:
        overall_score += 1.0
        if do_print: print("correct answer")
    else:
        if do_print: print(f"wrong answer, GT is {ground_truth}")
    return overall_score


def reward_func(completions, ground_truth, **kwargs):
    return [compute_reward(c,gt) for c,gt in zip(completions, ground_truth)]

training_args = GRPOConfig(output_dir="logs",
                            logging_steps=10,
                            logging_first_step = True,
                            max_completion_length = 1024,
                            num_generations= 8,
                            gradient_checkpointing = False,
                            temperature = 1.0,
                            num_train_epochs=3,
                            per_device_train_batch_size = 4,
                            gradient_accumulation_steps = 4,
                            lr_scheduler_type='constant',
                            save_strategy = 'steps',
                            save_steps = 100,
                            eval_strategy = 'steps',
                            eval_steps = 100,
                            eval_on_start= True,
                            bf16=True)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()