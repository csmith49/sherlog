type t = Watson.Proof.t CCRandom.t
(* posteriors are distributions over proofs *)

type score = Watson.Proof.t -> float
(* posteriors parameterized by score functions *)

val sample : t -> Watson.Proof.t
(* [sample posterior] converts [posterior] to a proof *)

module Feature : sig
    type t = Watson.Proof.t -> float
    (* features convert proofs to numerical indicators *)

    val intros : t
    (** [intros proof] counts the number of intros in [proof] *)
    
    val constrained_intros : t
    (** [constrained_intros proof] counts the number of constrained introductions in [proof] *)
    
    val length : t
    (** [length proof] is the number of resolution steps in [proof] *)

    val apply : Watson.Proof.t -> t -> float
    (** [apply proof f] applies [f] to [proof] - inverse application *)
end

module Parameterization : sig
    type t = float list
    (* parameterizations weight individual features *)
end

val dot : Parameterization.t -> Feature.t list -> score
(** [dot p fs] constructs a score that weights each feature in [fs] by the corresponding index in [p] *)

val ( @. ) : Parameterization.t -> Feature.t list -> score
(** [p @. fs] is short-hand for [dot p fs] *)

val random_proof : score -> Watson.Proof.t list -> t
(** [random_proof score proofs] samples a proof from proofs proportional to [score proof] *)