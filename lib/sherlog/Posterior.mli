module Operator : sig
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

        val apply : Watson.Proof.t -> t -> float
        (** [apply proof f] applies [f] to [proof] - inverse application *)
    end

    type t = Feature.t list
    (* operators map proofs to a feature embedding *)

    val of_contexts : string list -> t
    (* constructs operator from context clues - contains default operators too *)

    val apply : t -> Watson.Proof.t -> float list
    (* apply an operator to a proof to derive a feature embedding *)
end

module Parameterization : sig
    type t = float list
    (* parameterizations weight individual features *)

    val linear_combination : t -> float list -> float
    (* dot product with list of evaluated features *)

    module JSON : sig
        val encode : t -> Yojson.Basic.t
        val decode : Yojson.Basic.t -> t option
    end
end

type t
(* posteriors combine operators and parameterizations *)

val make : Operator.t -> Parameterization.t -> t
(* make posterior from operator and parameterization *)

val operator : t -> Operator.t
(* deconstruct posterior to access operator *)

val parameterization : t -> Parameterization.t
(* deconstruct posterior to access parameterization *)