export CUDA_VISIBLE_DEVICES=1

cd .. && python -m ebany_research.llm_self_distillation.run_clm_no_trainer \
	  --cfg ebany_research/llm_self_distillation/configs/original-model-1.yaml