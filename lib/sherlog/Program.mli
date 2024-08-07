type t

module Functional : sig
    val rules : t -> Watson.Rule.t list
    val parameters : t -> Parameter.t list
    val posterior : t -> Posterior.t

    val introduction_rules : t -> Watson.Rule.t list
    val classical_rules : t -> Watson.Rule.t list

    val make : Watson.Rule.t list -> Parameter.t list -> Posterior.t -> t
end

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end

val space : t -> (module Search.Algorithms.Space with type state = Watson.Proof.Obligation.t and type witness = Watson.Proof.Witness.t)
