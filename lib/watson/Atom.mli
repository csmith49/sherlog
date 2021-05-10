type t

val make : string -> Term.t list -> t

val relation : t -> string
val terms : t -> Term.t list

val compare : t -> t -> int
val equal : t -> t -> bool

val to_string : t -> string

val variables : t -> string list

val apply : Substitution.t -> t -> t

val unify : t -> t -> Substitution.t option

val embed : t -> t list -> Substitution.t list
val embed_all : t list -> t list -> Substitution.t list

val pp : t Fmt.t

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end