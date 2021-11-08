module Domain : sig
	type t =
		| Unit
		| Positive
		| Real

	val pp : t Fmt.t
	val to_string : t -> string

	module JSON : sig
		val encode : t -> JSON.t
		val decode : JSON.t -> t option
	end
end
	
type t

module Functional : sig
	val name : t -> string
	val domain : t -> Domain.t
	val dimension : t -> int

	val make : string -> Domain.t -> int -> t
end

val pp : t Fmt.t
val to_string : t -> string

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end