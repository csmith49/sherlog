type t

val choices : t -> Choice.t list

val append : Choice.t -> t -> t

val empty : t

val score : t -> float

module JSON : (JSON.JSONable with type value := t)