type t

val make : Watson.Atom.t list -> t

val to_fact : t -> Watson.Fact.t

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end