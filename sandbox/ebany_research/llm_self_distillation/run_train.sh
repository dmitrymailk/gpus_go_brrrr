export CUDA_VISIBLE_DEVICES=1

# --model_name_or_path EleutherAI/pythia-70m \
python -m llm_self_distillation.run_clm_no_trainer \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
	--config_name EleutherAI/pythia-70m \
	--tokenizer_name EleutherAI/pythia-70m \
    --output_dir ./models/ \
	--num_train_epochs 1 \
	--gradient_accumulation_steps 4 \
	--per_device_train_batch_size 16 \
	--report_to wandb \
	--with_tracking