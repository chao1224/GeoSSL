source submit_utils.sh

cd ../../examples

export model_3d=painn
export dataset=md17
export task_list=(aspirin benzene2018 ethanol malonaldehyde naphthalene salicylic toluene uracil)

export lr_list=(5e-4)
export lr_scheduler_list=(CosineAnnealingLR)
export batch_size=1024
export lr_decay_step_size=200
export seed=42

export print_every_epoch=10
export epochs=1000

export MD17_train_batch_size=5
export time=3


for pretrain_dir in "${painn_pretrain_dir_list[@]}"; do
for task in "${task_list[@]}"; do
for lr in "${lr_list[@]}"; do
for lr_scheduler in "${lr_scheduler_list[@]}"; do

    export input_model_file=../output/"$pretrain_dir"/model.pth
    export output_model_dir=../output/"$pretrain_dir"/"$model_3d"/"$dataset"/"$task"_"$seed"/"$lr"_"$lr_scheduler"_"$MD17_train_batch_size"_"$epochs"
    export output_file="$output_model_dir"/result.out
    mkdir -p "$output_model_dir"

    if [[ ! -f "$output_model_dir"/model_final.pth ]]; then
        sbatch --gres=gpu:v100l:1 -c 8 --mem=32G -t "$time":59:00  --account=rrg-bengioy-ad --qos=high --job-name=md_"$model_3d"_"$time" \
        --output="$output_file" \
        ./run_finetune_md17.sh \
        --model_3d="$model_3d" --dataset="$dataset" --epochs="$epochs" \
        --task="$task" \
        --seed="$seed" \
        --batch_size="$batch_size" --MD17_train_batch_size="$MD17_train_batch_size" --print_every_epoch="$print_every_epoch" \
        --lr="$lr" --lr_scheduler="$lr_scheduler" --lr_decay_step_size="$lr_decay_step_size" \
        --no_eval_train --no_eval_test --num_workers=8 \
        --input_model_file="$input_model_file" --output_model_dir="$output_model_dir"
    fi

done
done
done
done