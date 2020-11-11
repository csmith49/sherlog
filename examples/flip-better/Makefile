.phony: all
all: flip-better-results.html

.precious: flip-better-results.jsonl
flip-better-results.jsonl: train.py
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

flip-better-results.html: flip-better-results.jsonl visualize.py
	@python3 visualize.py