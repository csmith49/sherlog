module Introduction : sig
	type t

    val target : t -> Watson.Term.t
    val mechanism : t -> string
    val context : t -> Watson.Term.t list
    val parameters : t -> Watson.Term.t list

    val equal : t -> t -> bool
    val make : string -> Watson.Term.t list -> Watson.Term.t list -> Watson.Term.t -> t

    val tag : t -> Watson.Term.t list

    val to_atom : t -> Watson.Atom.t
    val of_atom : Watson.Atom.t -> t option

    val to_string : t -> string

    val pp : t Fmt.t
end

type t

val introductions : t -> Introduction.t list

val empty : t
val of_proof : Watson.Proof.t -> t

val join : t -> t -> t

val is_extension : t -> t -> bool
val extension_witness : t -> t -> t option

val to_string : t -> string

val pp : t Fmt.t