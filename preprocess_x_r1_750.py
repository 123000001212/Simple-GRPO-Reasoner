
"""
Preprocess the x_r1_750 dataset to parquet format
"""

import re
import os
import datasets
import argparse


def extract_solution(solution_str):
    solution = re.search(r'\\boxed{(.*?)}', solution_str)
    assert solution is not None
    final_solution = solution.group(1)
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/x_r1_750')

    args = parser.parse_args()

    data_source = 'xiaodongguaAIGC/X-R1-750'

    dataset = datasets.load_dataset(data_source, 'default')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    user_prefix = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and \\boxed{} tags, respectively, i.e., <think> reasoning process here </think> The answer is \\boxed{your answer}.\nUser: '
    assistant_prefix = 'Assistant: Let me solve this step by step. <think>'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('problem')

            question = user_prefix + question_raw + '\n' + assistant_prefix

            answer_raw = example.pop('solution')
            solution = extract_solution(answer_raw)
            data = {
                "prompt":  question,
                "ground_truth": solution,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

