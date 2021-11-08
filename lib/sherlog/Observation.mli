(* Literals relate model variabels to values *)
module Literal : sig
    type t =
        | Equal of string * Model.Value.t
        | NotEqual of string * Model.Value.t

    (* getters *)
    val variable : t -> string
    val value : t -> Model.Value.t

    (* constructors *)
    val eq : string -> Model.Value.t -> t
    val neq : string -> Model.Value.t -> t

    (* pretty printing *)
    val pp : t Fmt.t 

    (* JSON interface *)
    module JSON : sig
        (* only encoding for now (blocked by no Model.Value.JSON.decode) *)
        val encode : t -> JSON.t
    end
end

(* observations store a list of literals *)
type t = Literal.t list

(* construction *)
val eq_of_assoc : (string * Model.Value.t) list -> t

(* pretty printing *)
val pp : t Fmt.t

(* JSON interface *)
module JSON : sig
    (* only encoding for now (blocked by no Model.Value.JSON.decode) *)
    val encode : t -> JSON.t
end