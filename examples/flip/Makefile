.phony: all
all: flip-results.html

.precious: flip-results.jsonl
flip-results.jsonl: train.py
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

flip-results.html: flip-results.jsonl visualize.py
	@python3 visualize.py