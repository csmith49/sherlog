type t =
    | Variable of string
    | Symbol of string
    | Integer of int
    | Float of float
    | Boolean of bool
    | Unit
    | Function of string * t list
    | Wildcard

val to_string : t -> string

val compare : t -> t -> int
val equal : t -> t -> bool

val is_ground : t -> bool
val occurs : string -> t -> bool

val variables : t -> string list

val pp : t Fmt.t

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end