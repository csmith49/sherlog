module GroundTerm : sig
    type t

    val of_term : Watson.Term.t -> t option

    module JSON : sig
        val encode : t -> JSON.t
    end

    val lift : Watson.Term.t -> t Pipe.Value.t option
end

module Observation : sig
    type t = (string * GroundTerm.t Pipe.Value.t) list

    module JSON : sig
        val encode : t -> JSON.t
    end
end

type t

module Functional : sig
    val pipeline : t -> GroundTerm.t Pipe.Pipeline.t
    val observations : t -> Observation.t list
    val history : t -> Search.History.t

    val make : GroundTerm.t Pipe.Pipeline.t -> Observation.t list -> Search.History.t -> t
end

module JSON : sig
    val encode : t -> JSON.t
end

val of_proof : Proof.proof -> Search.History.t -> t