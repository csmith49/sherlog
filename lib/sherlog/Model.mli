module Assignment : sig
    type t

    val make : string -> string -> Watson.Term.t list -> t

    val target : t -> string
    val guard : t -> string 
    val parameters : t -> Watson.Term.t list

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end

    val pp : t Fmt.t
    val to_string : t -> string
end

module Observation : sig
    type t = (string * Watson.Term.t) list

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end

    val pp : t Fmt.t
    val to_string : t -> string
end

type t

val make : Assignment.t list -> Observation.t -> Observation.t -> t

val assignments : t -> Assignment.t list
val meet : t -> Observation.t
val avoid : t -> Observation.t

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end

val of_proof : Watson.Proof.t -> t
val of_proof_and_contradiction : Watson.Proof.t -> Watson.Proof.t -> t

val pp : t Fmt.t
val to_string : t -> string