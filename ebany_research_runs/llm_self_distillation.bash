export CUDA_VISIBLE_DEVICES=3

cd .. && python -m ebany_research.llm_self_distillation.run_clm_no_trainer \
	--cfg ebany_research/llm_self_distillation/configs/be-your-own-teacher-pythia-410m-2.yaml
	# --cfg ebany_research/llm_self_distillation/configs/original-model-pythia-410m-1.yaml