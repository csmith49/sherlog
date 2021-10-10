(** Utility module for producing and consuming Yojson objects. *)

type t = Yojson.Basic.t

type 'a parser = t -> 'a option
type 'a encoder = 'a -> t

(** Functions for constructing JSON parsers. *)
module Parse : sig
    val string : string parser
    val int : int parser
    val float : float parser
    val bool : bool parser
    val null : unit parser

    val list : 'a parser -> 'a list parser
    val assoc : 'a parser -> (string * 'a) list parser

    val find : string -> 'a parser -> 'a parser
end

(** Functions for constructing JSON encoders. *)
module Encode : sig
    val string : string encoder
    val int : int encoder
    val float : float encoder
    val bool : bool encoder
    val null : unit encoder

    val list : 'a encoder -> 'a list encoder
    val assoc : 'a encoder -> (string * 'a) list encoder
end

(* Interface signature *)
module type JSONable = sig
    type value

    val encode : value -> t
    val decode : t -> value option
end