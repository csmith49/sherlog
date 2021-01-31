type t

val name : t -> string
val index : t -> int

val reindex : int -> t -> t

val compare : t -> t -> int
val equal : t -> t -> bool

val to_string : t -> string
val of_string : string -> t

val uniq : t list -> t list

val avoiding_index : t list -> int