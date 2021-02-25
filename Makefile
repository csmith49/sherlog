.PHONY: all
all: install

.PHONY: install
install:
	@opam install . --working-dir
	@python3 -m pip install .

# for entering interactive mode
.PHONY: live
live: lib
	@dune utop lib

# for cleaning
.PHONY: clean
clean:
	@dune clean
