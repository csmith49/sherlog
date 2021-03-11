type t

val make : Watson.Atom.t list -> t

val to_atoms : t -> Watson.Atom.t list

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end