(*  *)
module Observation : sig
    (* assocs from names to concrete values *)
    type t = (string * Model.Value.t) list

    (* constructed from introduction sets *)
    (* val of_introductions : Introduction.t list -> t *)

    val pp : t Fmt.t

    (* JSON encoding only *)
    module JSON : sig
        val encode : t -> JSON.t
    end
end

(*  *)
type t

module Functional : sig
    val pipeline : t -> Model.t
    val observations : t -> Observation.t list
    val history : t -> Search.History.t

    val make : Model.t -> Observation.t list -> Search.History.t -> t
end

(* Construction *)

val of_proof : Proof.proof -> Search.History.t -> t

(* IO *)

(* pretty-printing *)
val pp : t Fmt.t

(* JSON encoding only *)
module JSON : sig
    val encode : t -> JSON.t
end