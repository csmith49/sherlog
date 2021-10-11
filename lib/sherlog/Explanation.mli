module GroundTerm : sig
    type t

    val pp : t Fmt.t

    val of_term : Watson.Term.t -> t option

    module JSON : sig
        val encode : t -> JSON.t
    end

    val lift : Watson.Term.t -> t Pipe.Value.t option
end

module Observation : sig
    type t = (string * GroundTerm.t Pipe.Value.t) list

    val pp : t Fmt.t

    module JSON : sig
        val encode : t -> JSON.t
    end
end

type t

val pp : t Fmt.t

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