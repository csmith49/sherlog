# assume a scale-free social graph where:
# 1. nodes are given by person(X)
# 2. edges are given by friend(X, Y)
# 3. edges are directionless, encoded by the rule

friend(X, Y) <- friend(Y, X).

# PARAMETERS -----------------

!parameter stress : Unit.
!parameter influence : Unit.
!parameter intensional_cancer : Unit.
!parameter extensional_cancer : Unit.

# PROBABILISTIC STRUCTURE ----

# we want to write a ProbLog-style rule of the form
# p :: stress(X) <- person(X).
# to do so with value introductions, use an intermediate relation, e.g.:
# stress(X) <- stress_latent(X, true).
# stress_latent(X; Bernoulli[p] @ stress_latent, X) <- person(X).

stress :: stress(X) <- person(X).
influence :: influences(X, Y) <- friend(X, Y).          # directed, unlike friend(...)!

# a person smokes if they're stressed, or if they're influenced by someone who does

smokes(X) <- stress(X).
smokes(X) <- influences(Y, X), smokes(Y).

# smoking can cause cancer, but it can also arrive spontaneously

intensional_cancer :: cancer(X) <- person(X).
extensional_cancer :: cancer(X) <- smokes(X).

# INFERENCE ------------------

# given partial information about smoking and cancer, derive optimal parameter values

!datapoint
    smokes(alice), smokes(bob), cancer(charlie).

!datapoint
    cancer(alice), cancer(bob).

!datapoint
    smokes(alice), smokes(charlie).

# what is the complexity of computing the likelihood of a datapoint?

# THE PROBLEM ----------------

# for all the data given above, there's a natural solution:
# stress : 1.0
# influence : _
# intensional_cancer : 1.0
# extensional_cancer : _

# cannot present *evidence of absence* in a datapoint
# e.g., we want to be able to encode:
# !cancer(bob), !smokes(alice).

# instead, rely on *ontologies* to reinforce the closed-world assumption
# to do so, give all predicates a second Boolean-valued index
# (such a transformation can be done automatically)

stress :: stress(X, true) <- person(X).
influence :: influences(X, Y, true) <- friend(X, Y).

smokes(X, true) <- stress(X, true).
smokes(X, true) <- influences(X, Y, true), smokes(Y, true).

intensional_cancer :: cancer(X, true) <- person(X).
extensional_cancer :: cancer(X, true) <- smokes(X, true).

# then we can provide ontology rules such as:

person(X) -> stress(X, B).
friend(X, Y) -> influences(X, Y, B).
person(X) -> smokes(X, B).
person(X) -> cancer(X, B).

# and constraints that result in a failure state !:

stress(X, true), stress(X, false) -> !
influences(X, Y, true), influences(X, Y, false) -> !
smokes(X, true), smokes(X, false) -> !
cancer(X, true), cancer(X, false) -> !

# challenge now becomes answering queries wrt ontologies