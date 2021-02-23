module State : sig
    type t

    val compare : t -> t -> int
    val equal : t -> t -> bool

    val of_fact : Fact.t -> t

    val discharge : t -> (Atom.t * t) option

    val apply : Substitution.t -> t -> t

    val extend : Atom.t list -> t -> t

    val is_empty : t -> bool
    
    val variables : t -> string list

    val resolve : t -> Rule.t -> (Atom.t * t) option
end

type resolution = Atom.t * State.t
type t

val extend : t -> resolution -> t

val compare : t -> t -> int
val equal : t -> t -> bool

val resolutions : t -> resolution list

val of_fact : Fact.t -> t
val to_fact : t -> Fact.t
val is_complete : t -> bool
val remaining_obligation : t -> Fact.t
val length : t -> int

val resolve : t -> Rule.t -> t option