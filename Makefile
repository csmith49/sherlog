.PHONY: all
all: run server

# building the executables
run: lib bin/run.ml
	@dune build bin/run.exe
	@mv _build/default/bin/run.exe run

server: lib bin/server.ml
	@dune build bin/server.exe
	@mv _build/default/bin/server.exe server

# for entering interactive mode
.PHONY: live
live: lib
	@dune utop lib

# for cleaning
.PHONY: clean
clean:
	@dune clean
	@rm -rf _build run server