module GroundTerm : sig
    type t

    val of_term : Watson.Term.t -> t option

    module JSON : sig
        val encode : t -> JSON.t
    end
end

module Observation : sig
    type t = (string * GroundTerm.t Pipeline.Value.t) list

    module JSON : sig
        val encode : t -> JSON.t
    end
end

type t

module Functional : sig
    val pipeline : t -> GroundTerm.t Pipeline.t
    val observations : t -> Observation.t list
    val history : t -> Search.History.t

    val make : GroundTerm.t Pipeline.t -> Observation.t list -> Search.History.t -> t
end

module JSON : sig
    val encode : t -> JSON.t
end

val of_proof : Proof.t -> Search.History.t -> t