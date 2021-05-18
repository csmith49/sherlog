type 'a t = ('a * Watson.Proof.t) CCRandom.t
(* posteriors are distributions over proofs *)

val sample : 'a t -> ('a * Watson.Proof.t)
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

    val context : string -> t
    (** [context str proof] counts the number of instances of str in a context in [proof] *)

    val context_operator : string list -> t list

    val apply : Watson.Proof.t -> t -> float
    (** [apply proof f] applies [f] to [proof] - inverse application *)
end

module Parameterization : sig
    type t = float list
    (* parameterizations weight individual features *)

    module JSON : sig
        val encode : t -> Yojson.Basic.t
        val decode : Yojson.Basic.t -> t option
    end
end

module Score : sig
    type t = {
        parameters : Parameterization.t;
        operator : Feature.t list;
    }

    val dot : Parameterization.t -> Feature.t list -> t
    (** [dot p fs] constructs a score that weights each feature in [fs] by the corresponding index in [p] *)

    val ( @. ) : Parameterization.t -> Feature.t list -> t
    (** [p @. fs] is short-hand for [dot p fs] *)

    val of_assoc : (float * Feature.t) list -> t

    module Record : sig
        type t = float list
        module JSON: sig
            val encode : t -> Yojson.Basic.t
            val decode : Yojson.Basic.t -> t option
        end
    end

    val apply : t -> Watson.Proof.t -> float
    val project : t -> Watson.Proof.t -> Record.t

    val apply_and_project : t -> Watson.Proof.t -> (float * Record.t)
end

val random_proof : Score.t -> Watson.Proof.t list -> unit t
(** [random_proof score proofs] samples a proof from proofs proportional to [score proof] *)

val random_proof_and_records : Score.t -> Watson.Proof.t list -> (Score.Record.t * Score.Record.t list) t