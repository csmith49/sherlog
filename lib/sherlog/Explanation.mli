module Introduction : sig
	type t

    val target : t -> Watson.Term.t
    val mechanism : t -> string
    val context : t -> Watson.Term.t list
    val parameters : t -> Watson.Term.t list

    val make : string -> Watson.Term.t list -> Watson.Term.t list -> Watson.Term.t -> t

    val to_atom : t -> Watson.Atom.t
    val of_atom : Watson.Atom.t -> t option
end

type t

val introductions : t -> Introduction.t list

val empty : t
val of_proof : Watson.Proof.t -> t

val join : t -> t -> t