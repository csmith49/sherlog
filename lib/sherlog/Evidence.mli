type t

val make : Watson.Atom.t list -> t

val to_atoms : t -> Watson.Atom.t list

val to_json : t -> JSON.t
val of_json : JSON.t -> t option
