# Everything assumes src/ is on the path.
export PYTHONPATH := src

.PHONY: help test smoke pipeline data dashboard leaderboard demos clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

test:        ## run the test suite (no pytest needed)
	python tests/test_smoke.py

smoke:       ## whole pipeline at CI scale, no network (~3 min)
	AIENH_TRACKER=local python -m aienh pipeline --scale smoke

pipeline:    ## the real demo run (~20-40 min on a laptop)
	python -m aienh pipeline --scale small

data:        ## show the preprocessing pipeline on a deliberately dirty corpus
	python -m aienh data --corpus dirty

dashboard:   ## regenerate artifacts/dashboard.html from the registry
	python -m aienh dashboard

leaderboard: ## print the registry as a table
	python -m aienh leaderboard

demos:       ## the two failure-mode demos
	python scripts/demo_template_mismatch.py
	python scripts/demo_data_bias.py

ci:          ## what a CI job should run
	python tests/test_smoke.py
	AIENH_TRACKER=local python -m aienh pipeline --scale smoke

clean:       ## remove run outputs
	rm -rf runs artifacts logs
