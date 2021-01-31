type t

val empty : t
val singleton : Identifier.t -> Term.t -> t
val of_list : (Identifier.t * Term.t) list -> t
val to_list : t -> (Identifier.t * Term.t) list

val apply : t -> Term.t -> Term.t

val simplify : t -> t

val compose : t -> t -> t

module Unification : sig
    type equality
    val equate : Term.t -> Term.t -> equality

    val resolve_equalities : equality list -> t option
end