export CUDA_VISIBLE_DEVICES=2
deepspeed --bind_cores_to_rank --include localhost:0 1_1_vanilla_deepspeed.py --checkpoint_dir experiment_deepspeed $@
