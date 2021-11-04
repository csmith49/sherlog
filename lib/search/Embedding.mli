(* linear combination of features *)
type t

(* magnitude of each feature *)
val features : t -> float list
(* weight of each feature in linear combination *)
val weights : t -> float list

(* compute the linear combination of features and weights *)
val score : t -> float

(* construct an embedding from a bunch of score functions *)
val of_features : (float * ('a -> float)) list -> 'a -> t

module JSON : (JSON.JSONable with type value := t)