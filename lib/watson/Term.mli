type t =
    | Variable of string
    | Symbol of string
    | Integer of int
    | Float of float
    | Boolean of bool
    | Function of string * t list
    | Unit

val to_string : t -> string

val compare : t -> t -> int
val equal : t -> t -> bool
val hash : t -> int

val is_ground : t -> bool
val occurs : string -> t -> bool

val variables : t -> string list

val pp : t Fmt.t

module Make : sig
    module Shorthand : sig
        val v : string -> t
        val s : string -> t
        val i : int    -> t
        val r : float  -> t
        val f : string -> t list -> t
        val b : bool   -> t
        val u :           t
    end

    module Variable : sig
        val avoiding : string list -> t
        val tagged : string -> int -> t
        val wildcard : unit -> t
    end
end

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end
