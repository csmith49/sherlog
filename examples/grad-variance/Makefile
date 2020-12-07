.phony: all
all: grad-variance-results.html

.precious: grad-variance-results.jsonl
grad-variance-results.jsonl: train.py
	@python3 train.py --log
	@python3 train.py --log
	@python3 train.py --log
	@python3 train.py --log
	@python3 train.py --log
	@python3 train.py --log
	@python3 train.py --log
	@python3 train.py --log
	@python3 train.py --log
	@python3 train.py --log

grad-variance-results.html: grad-variance-results.jsonl visualize.py
	@python3 visualize.py