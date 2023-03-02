cd ../../examples
export mode=pretrain_InfoNCE
export model_3d_list=(schnet painn)


export lr_list=(5e-4)
export lr_scheduler_list=(CosineAnnealingLR)
export batch_size=128
export num_workers=8


export input_data_dir=../data/Molecule3D
export dataset=Molecule3D_1000000
export epochs=100
export time=23



export GeoSSL_mu_list=(0)
export GeoSSL_sigma_list=(0.3 1)
export GeoSSL_atom_masking_ratio_list=(0 0.3)
export GeoSSL_option_list=(InfoNCE)


for model_3d in "${model_3d_list[@]}"; do
for lr in "${lr_list[@]}"; do
for lr_scheduler in "${lr_scheduler_list[@]}"; do
for GeoSSL_mu in "${GeoSSL_mu_list[@]}"; do
for GeoSSL_sigma in "${GeoSSL_sigma_list[@]}"; do
for GeoSSL_atom_masking_ratio in "${GeoSSL_atom_masking_ratio_list[@]}"; do
for EBM_option in "${EBM_option_list[@]}"; do

    export output_model_dir=../output/"$mode"/"$dataset"_"$model_3d"/"$lr"_"$lr_scheduler"_"$GeoSSL_mu"_"$GeoSSL_sigma"_"$GeoSSL_atom_masking_ratio"_"$EBM_option"_"$epochs"
    export output_file="$output_model_dir"/result.out
    echo "$output_model_dir"
    mkdir -p "$output_model_dir"

    if [[ ! -f "$output_model_dir"/model_final.pth ]]; then
        ls "$output_model_dir"
        echo "$output_file" undone
        sbatch --gres=gpu:v100l:1 -c 8 --mem=32G -t "$time":59:00  --account=rrg-bengioy-ad --qos=high --job-name=InfoNCE_"$model_3d"_"$time" \
        --output="$output_file" \
        ./run_pretrain_EBM.sh \
        --model_3d="$model_3d" --input_data_dir="$input_data_dir" --dataset="$dataset" \
        --epochs="$epochs" --lr="$lr" --lr_scheduler="$lr_scheduler" \
        --GeoSSL_mu="$GeoSSL_mu" --GeoSSL_sigma="$GeoSSL_sigma" --GeoSSL_atom_masking_ratio="$GeoSSL_atom_masking_ratio" \
        --EBM_option="$EBM_option" \
        --num_workers="$num_workers" --batch_size="$batch_size" \
        --output_model_dir="$output_model_dir"
    fi

done
done
done
done
done
done
done
