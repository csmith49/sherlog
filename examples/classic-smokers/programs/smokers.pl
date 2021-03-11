t(_) :: stress(X) :- person(X).
t(_) :: asthma_latent(X) :- person(X).
t(_) :: asthma_smokes(X) :- person(X).
t(_) :: influences(X, Y) :- friend(X, Y).

smokes(X) :- stress(X).
smokes(X) :- influences(X, Y), smokes(Y).
asthma(X) :- asthma_latent(X).
asthma(X) :- smokes(X), asthma_smokes(X).