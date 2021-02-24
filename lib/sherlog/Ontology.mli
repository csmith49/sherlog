type t

val empty : t
val make : Watson.Rule.t list -> (Watson.Atom.t list) list -> t

val dependencies : t -> Watson.Rule.t list
val constraints : t -> (Watson.Atom.t list) list

val add_dependency : Watson.Rule.t -> t -> t
val add_constraint : Watson.Atom.t list -> t -> t

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end