type t

val make : Atom.t -> Atom.t list -> t

val head : t -> Atom.t
val body : t -> Atom.t list

val variables : t -> Identifier.t list

val apply : Substitution.t -> t -> t

val avoiding_rename : Identifier.t list -> t -> t

val to_string : t -> string

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end