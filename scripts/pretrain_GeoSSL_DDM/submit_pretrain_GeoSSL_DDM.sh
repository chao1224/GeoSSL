cd ../../examples
export mode=pretrain_GeoSSL_DDM
export model_3d_list=(painn schnet)


export lr_list=(5e-4)
export lr_scheduler_list=(CosineAnnealingLR)
export batch_size=128
export num_workers=8



export input_data_dir=../data/Molecule3D
export dataset=Molecule3D_1000000
export epochs=100
export time=23



export GeoSSL_mu_list=(0 1)
export GeoSSL_sigma_list=(0.3 5)
export GeoSSL_atom_masking_ratio_list=(0 0.3)
export GeoSSL_option_list=(DDM)



export SM_sigma_begin_list=(10)
export SM_sigma_end_list=(0.01)
export SM_num_noise_level_list=(30 50)
export SM_noise_type_list=(symmetry)
export SM_anneal_power_list=(0.5 1 2)


export GeoSSL_sigma_list=(0.3)
export GeoSSL_atom_masking_ratio_list=(0 0.15 0.3)
export SM_num_noise_level_list=(30 50)
export SM_anneal_power_list=(0.05 0.1 0.2 0.5)


for model_3d in "${model_3d_list[@]}"; do
for lr in "${lr_list[@]}"; do
for lr_scheduler in "${lr_scheduler_list[@]}"; do
for GeoSSL_mu in "${GeoSSL_mu_list[@]}"; do
for GeoSSL_sigma in "${GeoSSL_sigma_list[@]}"; do
for GeoSSL_atom_masking_ratio in "${GeoSSL_atom_masking_ratio_list[@]}"; do
for GeoSSL_option in "${GeoSSL_option_list[@]}"; do

for SM_sigma_begin in "${SM_sigma_begin_list[@]}"; do
for SM_sigma_end in "${SM_sigma_end_list[@]}"; do
for SM_num_noise_level in "${SM_num_noise_level_list[@]}"; do
for SM_noise_type in "${SM_noise_type_list[@]}"; do
for SM_anneal_power in "${SM_anneal_power_list[@]}"; do

    export output_model_dir=../output/"$mode"/"$dataset"_"$model_3d"/"$lr"_"$lr_scheduler"_"$GeoSSL_mu"_"$GeoSSL_sigma"_"$GeoSSL_atom_masking_ratio"_"$GeoSSL_option"_"$SM_sigma_begin"_"$SM_sigma_end"_"$SM_num_noise_level"_"$SM_noise_type"_"$SM_anneal_power"_"$epochs"
    export output_file="$output_model_dir"/result.out
    echo "$output_model_dir"
    mkdir -p "$output_model_dir"

    if [[ ! -f "$output_model_dir"/model_final.pth ]]; then
        sbatch --gres=gpu:v100l:1 -c 8 --mem=32G -t "$time":59:00  --account=rrg-bengioy-ad --qos=high --job-name=NCE_"$model_3d"_"$time" \
        --output="$output_file" \
        ./run_pretrain_GeoSSL.sh \
        --model_3d="$model_3d" --input_data_dir="$input_data_dir" --dataset="$dataset" \
        --epochs="$epochs" --lr="$lr" --lr_scheduler="$lr_scheduler" \
        --GeoSSL_mu="$GeoSSL_mu" --GeoSSL_sigma="$GeoSSL_sigma" --GeoSSL_atom_masking_ratio="$GeoSSL_atom_masking_ratio" \
        --GeoSSL_option="$GeoSSL_option" \
        --SM_sigma_begin="$SM_sigma_begin" --SM_sigma_end="$SM_sigma_end" --SM_num_noise_level="$SM_num_noise_level" \
        --SM_noise_type="$SM_noise_type" --SM_anneal_power="$SM_anneal_power" \
        --num_workers="$num_workers" --batch_size="$batch_size" \
        --output_model_dir="$output_model_dir"
    fi

    echo
    echo
    echo

done
done
done
done
done

done
done
done
done
done
done
done
