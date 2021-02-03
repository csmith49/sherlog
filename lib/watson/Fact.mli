type t

val compare : t -> t -> int
val equal : t -> t -> bool

val empty : t
val singleton : Atom.t -> t
val mem : Atom.t -> t -> bool

val atoms : t -> Atom.t list
val of_atoms : Atom.t list -> t

val conjoin : t -> t -> t
val add : Atom.t -> t -> t

val apply : Substitution.t -> t -> t

val to_string : t -> string

val discharge : t -> (Atom.t * t) option

val variables : t -> Identifier.t list