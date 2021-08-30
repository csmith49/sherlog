module Term : sig
    type t

    val of_watson_term : Watson.Term.t -> t option

    val to_json : t -> JSON.t
end

module Observation : sig
    type 'a t = (string * 'a Pipeline.Value.t) list

    val to_json : ('a -> JSON.t) -> 'a t -> JSON.t
end

type 'a t = {
    pipeline : 'a Pipeline.t;
    observation : 'a Observation.t;
    history : Search.History.t;
}

val pipeline : 'a t -> 'a Pipeline.t
val observation : 'a t -> 'a Observation.t
val history : 'a t -> Search.History.t

val to_json : ('a -> JSON.t) -> 'a t -> JSON.t

val of_proof : Watson.Proof.t -> Search.History.t -> Term.t t