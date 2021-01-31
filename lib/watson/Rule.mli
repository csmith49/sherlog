type t

val head : t -> Atom.t
val body : t -> Atom.t list

val variables : t -> Identifier.t list

val apply : Substitution.t -> t -> t

val avoiding_rename : Identifier.t list -> t -> t