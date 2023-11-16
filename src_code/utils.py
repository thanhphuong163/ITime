# This is my utilities for development
import time
import pandas as pd
import torch

# Decorators
def print_out(func):
    def _print_out(*args, **kwargs):
        params = [f"{arg[0]} = {arg[1]}" for arg in kwargs.items()]
        output = func(*args, **kwargs)
        if type(output) in [int, str]:
            output = [f"output = {output}"]
        elif type(output) == float:
            output = [f"output = {output:.4f}"]
        elif type(output) == dict:
            output = convert_str_fmt(output)
        else:
            pass
        params = "\n".join(params + output)
        print(f"{func.__name__}:\n{params}")
    return _print_out

def timing(func):
    def _timing(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"executing time = {(end - start)*1000.:.4} ms\n")
    return _timing

# Functions
def convert_str_fmt(item, align_right=True):
    if type(item) in [int, str]:
        if align_right:
            return f"{item:>10}"
        else:
            return f"{item:<10}"
    elif type(item) == float:
        if align_right:
            return f"{item:>10.4f}"
        else:
            return f"{item:<10.4f}"
    elif type(item) == dict:
        return [f"{arg[0]} = {convert_str_fmt(arg[1], align_right=False)}" for arg in item.items()]
    else:
        pass

def print_out_row(row):
    if type(row) in [list, tuple]:
        str_row = [convert_str_fmt(item) for item in row]
    elif type(row) == dict:
        str_row = [convert_str_fmt(item[1]) for item in row.items()]
    else:
        pass
    print("".join(str_row))

def print_metrics(metrics):
        print()
        for key, val in metrics.items():
            print(f"{key:40s}: {val:15.5f}")

def save_result(outputs, ground_truth, output_dir):
    score_keys = [key for key in outputs[0].keys() if 'score' in key]
    result = {key: torch.concat([output[key] for output in outputs]).cpu().numpy() for key in score_keys}
    # result['ground_truth'] = ground_truth.cpu().numpy()
    # output_dir = "../notebooks/predicted_scores.csv"
    print("Save predicted scores")
    output_path = f"{output_dir}/predicted_score.csv"
    result_df = pd.DataFrame(data=result)
    result_df.to_csv(output_path)