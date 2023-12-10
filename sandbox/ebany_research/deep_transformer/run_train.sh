export CUDA_VISIBLE_DEVICES=1

# --model_name_or_path EleutherAI/pythia-70m \
python -m deep_transformer.run_clm_no_trainer \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
	--config_name EleutherAI/pythia-70m \
	--tokenizer_name EleutherAI/pythia-70m \
    --output_dir ./models/ \
	--num_train_epochs 4 \
	--gradient_accumulation_steps 4 \
	--per_device_train_batch_size 2 \
	--report_to wandb \
	--with_tracking