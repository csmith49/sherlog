type t

val make : string -> Term.t list -> t

val relation : t -> string
val terms : t -> Term.t list

val compare : t -> t -> int
val equal : t -> t -> bool

val to_string : t -> string

val variables : t -> string list

val apply : Substitution.t -> t -> t

val pp : t Fmt.t

module Unification : sig
    val unify : t -> t -> Substitution.t option
end

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end

module Infix : sig
    (* alias for apply *)
    val ($) : Substitution.t -> t -> t
end
