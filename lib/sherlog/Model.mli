type t

val of_solutions : Watson.Proof.Solution.t list -> t
val of_proof : Watson.Proof.t -> t
val to_json : t -> Yojson.Basic.t