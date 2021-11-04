module Feature : sig
    type t

    val apply : Branch.t -> t -> float

    module JSON : (JSON.JSONable with type value := t)
end

type t

val apply : t -> Branch.t -> Search.Embedding.t

module JSON : (JSON.JSONable with type value := t)