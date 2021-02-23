type t

val empty : t
val singleton : string -> Term.t -> t
val of_list : (string* Term.t) list -> t
val to_list : t -> (string * Term.t) list

val apply : t -> Term.t -> Term.t

val simplify : t -> t

val compose : t -> t -> t

val to_string : t -> string

module Unification : sig
    type equality
    val equate : Term.t -> Term.t -> equality

    val resolve_equalities : equality list -> t option
end