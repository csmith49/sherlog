type t

val of_proof : Watson.Proof.t -> t
val to_json : t -> Yojson.Basic.t