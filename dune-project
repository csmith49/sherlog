(lang dune 2.6)
(name sherlog)
(version 0.9)

(generate_opam_files true)

(source (github csmith49/sherlog))
(authors "Calvin Smith")

(package
    (name sherlog)
    (synopsis "An OCaml library for deep, probabilistic, generative logic programming")
    (depends
        (containers (> 3.0))
        (dune (> 1.5))
    )
)
(using menhir 2.1)
