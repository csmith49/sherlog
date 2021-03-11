# SCFG

This directory contains example programs that encode a simple [SCFG](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar). The implementations currently do not work, as they rely on unsupported syntactic features, specifically:

* Lists. The function-based list encodings (using `nil` for empty and `cons(_, _)` for cons lists) *might* work, depending on your particular branch and use case, but the more recognizable list constructions (`::` for cons, `[]` for empty lists, and `[a, b, c]` for `a :: b :: c :: []`) have yet to be implemented.
* Distribution parameters. The parameter construction `!parameter x : dist[k]` representing all positive k-vectors with an l1-norm of 1 has yet to be implemented.

As such, these files should be treated as *motivational examples only* for the time being.

`counter.sl` and `derivation.sl` are based on [Problog reference implementations](https://dtai.cs.kuleuven.be/problog/tutorial/various/06_pcfg.html).