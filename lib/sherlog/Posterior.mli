module Feature : sig
    type t
    (* embeds a proof into the reals *)

    val apply : Proof.t -> t -> float
    (* reverse application *)

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

module Ensemble : sig
    type t
    (* collapses multiple embeddings into a single score *)

    val apply : Search.Featurization.t -> t -> float
    (* reverse application *)

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

type t
(* combines an ensemble with multiple embeddings *)

val featurize : Proof.t -> t -> Search.Featurization.t
(* embed a proof into a multi-dimensional feature space *)

val score : Search.Featurization.t -> t -> float
(* score a featurization using the ensemble *)

val default : t
(* default parameterization *)

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end