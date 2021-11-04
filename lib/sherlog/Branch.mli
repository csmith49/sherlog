type t
type status =
    | Frontier of Watson.Proof.Obligation.t
    | Terminal of bool

val compare : t -> t -> int
val equal : t -> t -> bool

(* accessors *)

val witnesses : t -> Watson.Proof.Witness.t list
val status : t -> status

val obligation : t -> Watson.Proof.Obligation.t option
val success : t -> bool option

(* construction *)

val of_conjunct : Watson.Atom.t list -> t

(* manipulation *)

val resolve : t -> Watson.Rule.t -> t option
val is_resolved : t -> bool

val extend : Watson.Rule.t list -> t -> t list

(* Evaluation *)

(* Algebras describe how a proof structure collapes to a single result value *)
module type Algebra = sig
    type result

    val terminal : bool -> result
    val frontier : Watson.Proof.Obligation.t -> result
    val witness : result -> Watson.Proof.Witness.t -> result
end

val eval : (module Algebra with type result = 'a) -> t -> 'a