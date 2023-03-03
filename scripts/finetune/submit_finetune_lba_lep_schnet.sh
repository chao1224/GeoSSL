source submit_utils.sh

cd ../../examples

export model_3d=schnet
export dataset_list=(lba lep)

export lr_scheduler_list=(CosineAnnealingLR)
export split=atom3d_lba_split30
export seed_list=(12 22 32 42 52)


export epochs_list=(300)
export time=1



for dataset in "${dataset_list[@]}"; do
for pretrain_dir in "${schnet_pretrain_dir_list[@]}"; do
for lr_scheduler in "${lr_scheduler_list[@]}"; do
for epochs in "${epochs_list[@]}"; do
for seed in "${seed_list[@]}"; do

    export input_model_file=../output/"$pretrain_dir"/model.pth

    if [ $dataset == "lba" ]; then
        export batch_size=64
        export lr=1e-4
        export output_model_dir=../output/"$pretrain_dir"/"$model_3d"/"$dataset"/"$split"_"$seed"/"$lr"_"$lr_scheduler"_"$batch_size"_"$epochs"
    else
        export batch_size=16
        export lr=1e-4
        export output_model_dir=../output/"$pretrain_dir"/"$model_3d"/"$dataset"/"$seed"/"$lr"_"$lr_scheduler"_"$batch_size"_"$epochs"
    fi

    export output_file="$output_model_dir"/result.out
    mkdir -p "$output_model_dir"

    if [[ ! -f "$output_model_dir"/model_final.pth ]]; then
        sbatch --gres=gpu:v100l:1 -c 8 --mem=32G -t "$time":59:00  --account=rrg-bengioy-ad --qos=high --job-name="$dataset"_"$model_3d"_"$time" \
        --output="$output_file" \
        ./run_finetune_"$dataset".sh \
        --model_3d="$model_3d" --dataset="$dataset" --epochs="$epochs" \
        --split="$split" --seed="$seed" \
        --batch_size="$batch_size" \
        --lr="$lr" --lr_scheduler="$lr_scheduler" --no_eval_train --print_every_epoch=1 --num_workers=8 \
        --input_model_file="$input_model_file" --output_model_dir="$output_model_dir"
    fi

done
done
done
done
done
