export CUDA_VISIBLE_DEVICES=0
deepspeed --bind_cores_to_rank train_bert_ds.py --checkpoint_dir experiment_deepspeed $@
