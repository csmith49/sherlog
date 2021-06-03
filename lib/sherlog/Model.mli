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

val make : Assignment.t list -> Observation.t -> Search.History.t -> t

val assignments : t -> Assignment.t list
val meet : t -> Observation.t
val history : t -> Search.History.t

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end

val of_proof : Watson.Proof.t -> Search.History.t -> t

val pp : t Fmt.t
val to_string : t -> string