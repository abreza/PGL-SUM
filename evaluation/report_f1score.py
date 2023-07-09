import os
import json
import numpy as np
import argparse

def get_f_scores_from_file(file_path):
    with open(file_path, 'r') as f:
        f_scores = json.load(f)
        return np.max(f_scores), f_scores[-1]

def calculate_averages(save_dir, video_type):
    max_f_scores = []
    last_f_scores = []

    for split_index in range(5):
        split_dir = os.path.join(save_dir, f'{video_type}/results/split{split_index}')
        f_scores_file = os.path.join(split_dir, 'f_scores.txt')

        max_f_score, last_f_score = get_f_scores_from_file(f_scores_file)

        max_f_scores.append(max_f_score)
        last_f_scores.append(last_f_score)

    max_f_scores_avg = np.mean(max_f_scores)
    last_f_scores_avg = np.mean(last_f_scores)

    return max_f_scores_avg, last_f_scores_avg

def main(save_dir, video_type, output_file):
    max_avg, last_avg = calculate_averages(save_dir, video_type)

    result = {
        "average_max_f_score": max_avg,
        "average_last_f_score": last_avg
    }

    with open(output_file, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate average of max_f_score and last_f_score')
    parser.add_argument('save_dir', help='The save directory')
    parser.add_argument('video_type', help='The video type (e.g., SumMe, TVSum)')
    parser.add_argument('output_file', help='The output JSON file')

    args = parser.parse_args()

    main(args.save_dir, args.video_type, args.output_file)
