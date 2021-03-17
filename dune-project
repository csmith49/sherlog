(lang dune 2.6)
(name sherlog)
(version 0.9)

(generate_opam_files true)

(source (github csmith49/sherlog))
(authors "Calvin Smith")
(maintainers "cjsmith@cs.wisc.edu")

(package
    (name sherlog)
    (synopsis "An OCaml library for deep, probabilistic, generative logic programming")
    (depends
        (ocaml (>= 4.11))
        (containers (>= 3.0))
        (dune (>= 2.6))
        (menhir (>= 20200624))
        (yojson (>= 1.7))
        (lwt (>= 5.3))
        (fmt (>= 0.8))
    )
)
(using menhir 2.1)
