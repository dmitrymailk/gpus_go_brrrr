export CUDA_VISIBLE_DEVICES=2

cd .. && python -m ebany_research.llm_self_distillation.run_clm_no_trainer \
	--cfg ebany_research/llm_self_distillation/configs/be-your-own-teacher-pythia-410m-8.yaml
	# --cfg ebany_research/llm_self_distillation/configs/simple-classifiers-pythia-410m-7.yaml
	#   --cfg ebany_research/llm_self_distillation/configs/simple-classifiers-pythia-410m-6.yaml
	#   --cfg ebany_research/llm_self_distillation/configs/simple-classifiers-pythia-410m-5.yaml
	# --cfg ebany_research/llm_self_distillation/configs/original-model-pythia-410m-4.yaml
	#   --cfg ebany_research/llm_self_distillation/configs/simple-classifiers-pythia-70m-3.yaml
	#   --cfg ebany_research/llm_self_distillation/configs/simple-classifiers.yaml
	#   --cfg ebany_research/llm_self_distillation/configs/original-model-1.yaml