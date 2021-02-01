type t

val make : string -> Term.t list -> t

val relation : t -> string
val terms : t -> Term.t list

val compare : t -> t -> int
val equal : t -> t -> bool

val to_string : t -> string

val variables : t -> Identifier.t list

val apply : Substitution.t -> t -> t

val unify : t -> t -> Substitution.t option

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end