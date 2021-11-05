module Operation : sig
    type t =
        | Size

    val apply : t -> Branch.t -> float

    module JSON : (JSON.JSONable with type value := t)
end

module Feature : sig
    type t

    val apply : t -> Branch.t -> float
    val weight : t -> float
    val tuple : t -> float * (Branch.t -> float)

    module JSON : (JSON.JSONable with type value := t)
end

type t

val apply : t -> Branch.t -> Search.Embedding.t

val of_operations : (float * Operation.t) list -> t
val default : t

module JSON : (JSON.JSONable with type value := t)