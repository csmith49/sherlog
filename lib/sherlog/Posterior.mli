module Feature : sig
    type t

    val apply : Proof.proof -> t -> float

    module JSON : (JSON.JSONable with type value := t)
end

module Ensemble : sig
    type t

    val apply : Search.Embedding.t -> t -> float

    module JSON : (JSON.JSONable with type value := t)
end

type t
(* combines an ensemble with multiple embeddings *)

val embed : Proof.proof -> t -> Search.Embedding.t

val score : Search.Embedding.t -> t -> float
(* score a featurization using the ensemble *)

val default : t
(* default parameterization *)

module JSON : (JSON.JSONable with type value := t)