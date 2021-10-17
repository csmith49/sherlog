(* Types *)

type proof =
    | Leaf of leaf
    | Interior of edge list
and leaf =
    | Frontier of Watson.Proof.Obligation.t
    | Success
    | Failure
and edge = Edge of Watson.Proof.Witness.t * proof

(* Construction *)

(** [of_conjunct atoms] constructs an un-explored proof from a list of atoms *)
val of_conjunct : Watson.Atom.t list -> proof

(** [trim proof] returns a modified proof with all failed branches removed *)
val trim : proof -> proof option

(* Accessors *)

(** [obligation proof] is the obligation stored at the root of the proof (if such an obligation exists) *)
val obligation : proof -> Watson.Proof.Obligation.t option

(* IO *)

(* pretty-printer for proof objects *)
val pp : proof Fmt.t

(* Manipulation *)

module Zipper : sig
    type t

    (* getters and setters *)
    val focus : t -> proof
    val set_focus : proof -> t -> t

    (* basic movement *)
    val left : t -> t option
    val right : t -> t option
    val up : t -> t option
    val down : t -> t option

    (* advanced movement *)
    val next : t -> t option
    val preorder : t -> t option
    val find : (proof -> bool) -> t -> t option
    val find_all : (proof -> bool) -> t -> t list

    (* construction / conversion *)
    val of_proof : proof -> t
    val to_proof : t -> proof
end

(* Evaluation *)

(** algebras describe how a proof structure is collapsed to a single result value *)
module type Algebra = sig
    type result

    val leaf : leaf -> result
    val edge : Watson.Proof.Witness.t -> result -> result
    val interior : result list -> result
end

(** [eval algebra proof] applies [algebra] to the proof object to produce a value of type [algebra.result] *)
val eval : (module Algebra with type result = 'a) -> proof -> 'a