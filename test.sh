#!/bin/bash

echo "Start testing..."
mkdir -p ./log_test

pretrained_models=()
for i in $(seq 10000 10000 200000); do
    pretrained_models+=("./checkpoints/transfered/superblue12_pretrain${i}/superblue12_model_iters_200.pth")
done

datasets=("superblue11_a" "superblue14" "superblue16_a" "superblue19")

for pretrained_model in "${pretrained_models[@]}"; do
    identifier=$(basename "$(dirname "${pretrained_model}")")

    for dataset in "${datasets[@]}"; do
        log_file="./log_test/${dataset}---test_${identifier}.log"
        echo "---------------------------------------------------"
        echo "$(date +"%Y-%m-%d %H:%M:%S") | Testing model: ${pretrained_model} on dataset: ${dataset}"
        
        python test.py \
            --pretrained "${pretrained_model}" \
            --dataroot ./transfer_data/ \
            --ann_file_test ./transfer_data/ \
            --dataset_type CongestionDataset \
            --mode transfer \
            --testdataset "${dataset}" > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ $(date +"%Y-%m-%d %H:%M:%S") | ${dataset} testing with ${identifier} completed."
    else
        echo "❌ $(date +"%Y-%m-%d %H:%M:%S") | ${dataset} testing with ${identifier} failed! Exiting..."
        exit 1
    fi
    done
done

echo "🎉 All testing jobs completed!"