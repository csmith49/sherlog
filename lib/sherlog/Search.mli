module Featurization : sig
    type t = float list

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

type 'a tree = 'a Data.Tree.tree

module type Domain = sig
    type t

    val features : t tree -> Featurization.t
    val score : Featurization.t -> float
    val expand : t -> bool
    val expansions : t -> (t tree) list
end

module Choice : sig
    type t

    val make : Featurization.t -> Featurization.t list -> t

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

module History : sig
    type t = Choice.t list

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

module Random : sig
    val prop_to : 'a list -> ('a -> float) -> 'a CCRandom.t

    val choice : (module Domain with type t = 'a) -> 'a tree list -> ('a tree * Choice.t) CCRandom.t
end

val beam : (module Domain with type t = 'a) -> int -> 'a tree -> ('a tree * History.t)