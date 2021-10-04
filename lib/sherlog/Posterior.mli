module Feature : sig
    type t

    val apply : Proof.proof -> t -> float

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

module Ensemble : sig
    type t

    val apply : Proof.Search.Embedding.t -> t -> float

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

type t
(* combines an ensemble with multiple embeddings *)

val embed : Proof.proof -> t -> Proof.Search.Embedding.t

val score : Proof.Search.Embedding.t -> t -> float
(* score a featurization using the ensemble *)

val default : t
(* default parameterization *)

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end