cd ../../examples
export mode=pretrain_3DInfoGraph
export model_3d_list=(schnet painn)

export lr_list=(5e-4)
export lr_scheduler_list=(CosineAnnealingLR)
export batch_size=128
export num_workers=8



export input_data_dir=../data/Molecule3D
export dataset=Molecule3D_1000000
export epochs=100
export time=23


for model_3d in "${model_3d_list[@]}"; do
for lr in "${lr_list[@]}"; do
for lr_scheduler in "${lr_scheduler_list[@]}"; do

    export output_model_dir=../output/"$mode"/"$dataset"_"$model_3d"/"$lr"_"$lr_scheduler"_"$epochs"
    export output_file="$output_model_dir"/result.out
    echo "$output_model_dir"
    # mkdir -p "$output_model_dir"

    # if [[ ! -f "$output_file" ]]; then
        # sbatch --gres=gpu:v100l:1 -c 8 --mem=64G -t "$time":59:00  --account=rrg-bengioy-ad --qos=high --job-name=3DIG_"$model_3d"_"$time" \
        # --output="$output_file" \
        # ./run_"$mode".sh \
        # --model_3d="$model_3d" --input_data_dir="$input_data_dir" --dataset="$dataset" \
        # --epochs="$epochs" --lr="$lr" --lr_scheduler="$lr_scheduler" \
        # --num_workers="$num_workers" --batch_size="$batch_size" \
        # --output_model_dir="$output_model_dir"
    # fi

done
done
done
