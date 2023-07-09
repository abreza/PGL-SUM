save_dir="../PGL-SUM/Summaries/from_scratch"
pretrained_models=("pretrain-epoch-10.pkl" "pretrain-epoch-20.pkl" "pretrain-epoch-40.pkl")

python evaluation/report_f1score.py ${save_dir} SumMe averages.json
python evaluation/report_f1score.py ${save_dir} TVSum averages.json


for model in ${pretrained_models[@]}; do
    save_dir="../PGL-SUM/Summaries/from_pretrain_${model:15:-4}"
    python evaluation/report_f1score.py ${save_dir} SumMe averages.json
    python evaluation/report_f1score.py ${save_dir} TVSum averages.json
done