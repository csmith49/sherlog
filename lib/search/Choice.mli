type t

val embedding : t -> Embedding.t
val context : t -> Embedding.t list

val make : Embedding.t -> Embedding.t list -> t

val choose : 'a list -> ('a -> Embedding.t) -> ('a * t) CCRandom.t

module JSON : (JSON.JSONable with type value := t)