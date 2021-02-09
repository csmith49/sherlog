module Node : sig
    type t = {
        target : string;
        guard : string;
        parameters : Watson.Term.t list;
    }
end

module Observation : sig

end

type t



module JSON : sig
    val encode : t -> JSON.t
end

type t

val of_solutions : Watson.Proof.Solution.t list -> t
val of_proof : Watson.Proof.t -> t
val to_json : t -> Yojson.Basic.t