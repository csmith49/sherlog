# RSA Reference Game

# things being referenced
object(one).
object(two).
object(three).

# properties for the objects
shape(one, square).
color(one, blue).

shape(two, circle).
color(two, blue).

shape(three, square).
color(three, green).

# the utterances usable by the speaker
utterance(square).
utterance(circle).
utterance(green).
utterance(blue).

# denotation for utterances
denotation(X, U) <- shape(X, U).
denotation(X, U) <- color(X, U).

# uniform prior over objects
prior(; {one, two, three} <~ uniform[3]).

# levels, for defining iteration
base([]).
level(X) <- base(X).
level([] :: X) <- level(X).
prec([] :: X, X).

listener(Level, Object, Utterance) <-
    base(Level),
    denotation(Object, Utterance),
    prior(Object).

listener(Level, Object, Utterance) <-
    recursive(Level),
    preceding(Level, Level'),
    speaker(Level', Utterance, Object),
    prior(Object).

speaker(Level, Utterance, Object) <-
    recursive(Level),
    preceding(Level, Level'),
    listener(Level', Object, Utterance).

