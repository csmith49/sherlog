type domain =
	| Unit
	| Positive
	| Real
	
type t

val name : t -> string
val domain : t -> domain

val make : string -> domain -> t

val to_string : t -> string

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end