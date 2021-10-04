(* Introductions *)

module Context : sig
    (* Contexts subscript rules and distinguish different generative pathways through the same atoms. *)
    type t

    (* construction *)
    val fresh : unit -> t

    (* conversion *)
    val to_string : t -> string
    val of_string : string -> t

    (* comparison *)
    val compare : t -> t -> int
    val equal : t -> t -> bool
    val hash : t -> int
end

(* Introductions record existential value introductions and their contexts. *)
type t = {
    relation : string;
    context : Context.t;
    terms : Watson.Term.t list;
    function_id : string;
    arguments : Watson.Term.t list;
    target : Watson.Term.t;
}

(* Functional interface *)
module Functional : sig
    val make : string -> Context.t -> Watson.Term.t list -> string -> Watson.Term.t list -> Watson.Term.t -> t

    val relation : t -> string
    val context : t -> Context.t
    val terms : t -> Watson.Term.t list
    val function_id : t -> string
    val arguments : t -> Watson.Term.t list
    val target : t -> Watson.Term.t
end

(* atomic embedding *)
val to_atom : t -> Watson.Atom.t
val of_atom : Watson.Atom.t -> t option

(* Utilities *)

val observed : t -> bool
(* [observed intro] is true if the target of [intro] is not a variable *)

val sample_site : t -> string

val equal : t -> t -> bool