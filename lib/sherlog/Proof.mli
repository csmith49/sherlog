module Tag : sig
    type t =
        | Root
        | Success
        | Failure
        | Witness of Watson.Proof.Witness.t

    val interior : t -> bool
    val exterior : t -> bool

    val witness : t -> Watson.Proof.Witness.t option
end

module Node : sig
    type t

    val tag : t -> Tag.t
    val obligation : t -> Watson.Proof.Obligation.t
end

type t = Node.t Search.tree

val of_conjunct : Watson.Atom.t list -> t
val of_witness : Watson.Proof.Witness.t -> Watson.Proof.Obligation.t -> t

val success : Watson.Proof.Obligation.t -> t
val failure : Watson.Proof.Obligation.t -> t