(* utility functions for parsing and constructing Yojson structures *)

(* alias for shorter signatures *)
type t = Yojson.Basic.t
type 'a parser = Yojson.Basic.t -> 'a option

(* common operations for decomposing JSON objects *)
module Parse : sig
    val list : 'a parser -> 'a list parser
    val assoc : 'a parser -> (string * 'a) list parser
    val find : 'a parser -> string -> 'a parser
    val identity : Yojson.Basic.t parser
    val string : string parser
    val int : int parser
    val float : float parser
    val bool : bool parser
end

(* wrappers for constructing JSON objects *)
module Make : sig
    val string : string -> t
    val int : int -> t
    val float : float -> t
    val bool : bool -> t
    val list : t list -> t
    val assoc : (string * t) list -> t
    val null : t
end