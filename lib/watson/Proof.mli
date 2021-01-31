module State : sig
    type t

    val of_fact : Fact.t -> t

    val discharge : t -> (Atom.t * t) option

    val apply : Substitution.t -> t -> t

    val extend : Atom.t list -> t -> t

    val is_empty : t -> bool
    
    val variables : t -> Identifier.t list
end

val resolve : State.t -> Rule.t -> (Atom.t * State.t) option