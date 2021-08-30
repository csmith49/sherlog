module Feature : sig
    type t = Watson.Proof.t -> float
    (* features score proofs *)

    val intros : t
    (* counts the number of intros in the proof *)

    val constrained_intros : t
    (* counts the number of constrained intros in the proof *)
    
    val length : t
    (* counts the number of resolution steps in the proof *)

    val context : string -> t
    (* counts the number of instances of the given string in contexts in the proof *)

    val apply : Watson.Proof.t -> t -> float
    (* inverse application *)
end

module Embedding : sig
    type t = float list
    (* embeddings of proofs into R^n *)

    val to_json : t -> JSON.t
    val of_json : JSON.t -> t option
end

module Operator : sig
    type t
    (* operators embed proofs *)

    val apply : t -> Watson.Proof.t -> Embedding.t
    (* embed the proof using the operator *)

    val default : t
    (* default operator relying on bog-standard features *)

    val of_context_clues : string list -> t
    (* construct operator from a finite set of features *)

    val join : t -> t -> t
    (* combine two operators *)

    val to_json : t -> JSON.t
    val of_json : JSON.t -> t option
end

module Ensemble : sig
    type t
    (* ensembles combine evaluated features *)

    val apply : t -> Embedding.t -> float
    (* project an embedding to a real-valued score *)

    val to_json : t -> JSON.t
    val of_json : JSON.t -> t option
end

type t
(* posteriors score proofs *)

(* application *)

val embed_and_score : t -> Watson.Proof.t -> (Embedding.t * float)
val apply : t -> Watson.Proof.t -> float

(* construction *)

val make : Operator.t -> Ensemble.t -> t
val uniform : t
val default : t

(* accessors *)

val operator : t -> Operator.t
val ensemble : t -> Ensemble.t

(* serialization *)

val to_json : t -> JSON.t
val of_json : JSON.t -> t option
