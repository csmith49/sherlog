type t

val make : Atom.t -> Atom.t list -> t

val head : t -> Atom.t
val body : t -> Atom.t list

val compare : t -> t -> int
val equal : t -> t -> bool

val variables : t -> string list

val apply : Substitution.t -> t -> t

val avoiding_rename : string list -> t -> t

val to_string : t -> string

val pp : t Fmt.t

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end