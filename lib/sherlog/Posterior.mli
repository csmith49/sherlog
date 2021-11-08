module Operation : sig
    type t

    val apply : t -> (Watson.Proof.Witness.t * Watson.Proof.Obligation.t) -> float

    module JSON : (JSON.JSONable with type value := t)
end

module Feature : sig
    type t

    val apply : t -> (Watson.Proof.Witness.t * Watson.Proof.Obligation.t) -> float
    val weight : t -> float
    val tuple : t -> float * (Watson.Proof.Witness.t * Watson.Proof.Obligation.t -> float)

    module JSON : (JSON.JSONable with type value := t)
end

type t

val apply : t -> (Watson.Proof.Witness.t * Watson.Proof.Obligation.t) -> Search.Embedding.t

val of_operations : (float * Operation.t) list -> t
val default : t

module JSON : (JSON.JSONable with type value := t)