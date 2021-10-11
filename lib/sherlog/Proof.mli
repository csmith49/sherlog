type proof =
    | Leaf of leaf
    | Interior of edge list
and leaf =
    | Frontier of Watson.Proof.Obligation.t
    | Success
    | Failure
and edge = Edge of Watson.Proof.Witness.t * proof

val of_conjunct : Watson.Atom.t list -> proof

val introductions : proof -> Introduction.t list

val branches : proof -> Introduction.t list list

val obligation : proof -> Watson.Proof.Obligation.t option

val pp : proof Fmt.t

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

module type Algebra = sig
    type result

    val leaf : leaf -> result
    val edge : Watson.Proof.Witness.t -> result -> result
    val interior : result list -> result
end

val eval : (module Algebra with type result = 'a) -> proof -> 'a