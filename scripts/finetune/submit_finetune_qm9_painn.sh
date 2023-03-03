source submit_utils.sh

cd ../../examples

export model_3d=painn
export dataset=qm9
export task_list=(mu alpha homo lumo gap r2 zpve u0 u298 h298 g298 cv)

export lr_list=(5e-4)
export lr_scheduler_list=(CosineAnnealingLR)
export split=customized_01
export seed=42

export epochs=1000


export time=11


for pretrain_dir in "${painn_pretrain_dir_list[@]}"; do
for task in "${task_list[@]}"; do
for lr in "${lr_list[@]}"; do
for lr_scheduler in "${lr_scheduler_list[@]}"; do

    export input_model_file=../output/"$pretrain_dir"/model.pth
    export output_model_dir=../output/"$pretrain_dir"/"$model_3d"/"$dataset"/"$task"_"$split"_"$seed"/"$lr"_"$lr_scheduler"_"$epochs"
    export output_file="$output_model_dir"/result.out

    mkdir -p "$output_model_dir"

    if [[ ! -f "$output_model_dir"/model_final.pth ]]; then
        sbatch --gres=gpu:v100l:1 -c 8 --mem=32G -t "$time":59:00  --account=rrg-bengioy-ad --qos=high --job-name=qm9_"$model_3d"_"$time" \
        --output="$output_file" \
        ./run_finetune_qm9.sh \
        --model_3d="$model_3d" --dataset="$dataset" --epochs="$epochs" \
        --task="$task" \
        --split="$split" --seed="$seed" \
        --batch_size=128 \
        --lr="$lr" --lr_scheduler="$lr_scheduler" --no_eval_train --print_every_epoch=1 --num_workers=8 \
        --input_model_file="$input_model_file" --output_model_dir="$output_model_dir"
    fi

done
done
done
done
