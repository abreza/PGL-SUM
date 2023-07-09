# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import h5py
import argparse
from tqdm import tqdm
from evaluation_metrics import evaluate_summary
from generate_summary import generate_summary

def main():
    # arguments to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,
                        default='../PGL-SUM/Summaries/PGL-SUM/exp1/SumMe/results/split0',
                        help="Path to the json files with the scores of the frames for each epoch")
    parser.add_argument("--dataset", type=str, default='SumMe', help="Dataset to be used")
    parser.add_argument("--eval", type=str, default="max", help="Eval method to be used for f_score reduction (max or avg)")

    args = parser.parse_args()
    path = args.path
    print(path)
    dataset = args.dataset
    eval_method = args.eval

    results = [f for f in os.listdir(path) if f.endswith(".json")]
    results.sort(key=lambda video: int(video[6:-5]))
    dataset_path = os.path.join('../PGL-SUM/data/datasets/', dataset, f'eccv16_dataset_{dataset.lower()}_google_pool5.h5')

    f_score_epochs = []
    for epoch in tqdm(results, desc="Processing epochs"):  # tqdm added for progress tracking
        all_scores, all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], [], []
        with open(os.path.join(path, epoch), 'r') as f:  # read the json file ...
            data = json.load(f)
            keys = list(data.keys())

            for video_name in keys:  # for each video inside that json file ...
                scores = np.asarray(data[video_name])  # read the importance scores from frames
                all_scores.append(scores)

        with h5py.File(dataset_path, 'r') as hdf:
            for video_name in keys:
                video_index = video_name[6:]
                dataset_group = hdf.get(f'video_{video_index}')

                all_user_summary.append(np.array(dataset_group.get('user_summary')))
                all_shot_bound.append(np.array(dataset_group.get('change_points')))
                all_nframes.append(np.array(dataset_group.get('n_frames')))
                all_positions.append(np.array(dataset_group.get('picks')))

        all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

        all_f_scores = []
        # compare the resulting summary with the ground truth one, for each video
        for video_index, summary in enumerate(all_summaries):
            user_summary = all_user_summary[video_index]
            f_score = evaluate_summary(summary, user_summary, eval_method)
            all_f_scores.append(f_score)

        f_score_epochs.append(np.mean(all_f_scores))

    # Save the importance scores in txt format.
    with open(os.path.join(path, 'f_scores.txt'), 'w') as outfile:
        json.dump(f_score_epochs, outfile)

if __name__ == "__main__":
    main()