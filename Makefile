.PHONY: help install verify eval ablation demo clean docker-verify docker-build

help:
	@echo "make install    Install dependencies"
	@echo "make verify     Recompute every number in the paper from released logs (no API key)"
	@echo "make eval       Full three-condition evaluation (needs OPENROUTER_API_KEY, hours)"
	@echo "make ablation   Five-condition component ablation on the 200-question subset"
	@echo "make demo       Launch the interactive Streamlit demo"
	@echo "make clean      Remove caches and scratch evaluation checkpoints"
	@echo ""
	@echo "make docker-verify   Verify the paper's numbers in a container (no local Python)"
	@echo "make docker-build    Build the full image for re-running the evaluation"

install:
	pip install -r requirements.txt

verify:
	python scripts/verify_paper_numbers.py

eval:
	./scripts/run_eval.sh

ablation:
	python ablation.py --n 200

demo:
	streamlit run app.py

docker-verify:
	docker build -t agentddx .
	docker run --rm agentddx

docker-build:
	docker build --target full -t agentddx:full .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f results/eval_llm_only.json results/eval_llm_pubmed.json \
	      results/eval_full_system.json results/eval_merged_log.json \
	      results/ablation_*.json
