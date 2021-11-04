module Embedding : sig
    type t

    val linear : float list -> t -> float

    val stack : float list -> t

    module JSON : (JSON.JSONable with type value := t)
end

module Choice : sig
    type t

    module JSON : (JSON.JSONable with type value := t)

    module Functional : sig
        val embedding : t -> Embedding.t
        val context : t -> Embedding.t list
    end

    val choose : 'a list -> ('a -> Embedding.t) -> (Embedding.t -> float) -> ('a * t) CCRandom.t
end

module History : sig
    (** A history is a sequence of choices *)
    type t

    module JSON : (JSON.JSONable with type value := t)
end

module type Structure = sig
    type candidate

    val stop : candidate -> bool
    val next : candidate -> candidate list

    val embed : candidate -> Embedding.t
    val score : Embedding.t -> float
end

(** randomly walk the search structure *)
val random_walk : (module Structure with type candidate = 'a) -> 'a -> ('a * History.t) CCRandom.t
