type t

module Functional : sig
    val rules : t -> Watson.Rule.t list

    val introduction_rules : t -> Watson.Rule.t list
    val classical_rules : t -> Watson.Rule.t list

    val parameters : t -> Parameter.t list

    val make : Watson.Rule.t list -> Parameter.t list -> t
end

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end

val domain : t -> Posterior.t -> (module Search.Domain with type t = Proof.Node.t)