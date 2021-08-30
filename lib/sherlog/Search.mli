val prop_to : 'a list -> ('a -> float) -> 'a CCRandom.t
(* sample a value proportional to the produced weight *)

(* Search Domain *)

module type Domain = sig
    type t

    val features : t -> float list
    val score : float list -> float
    val accept : t -> bool
    val reject : t -> bool
    val successors : t -> t list
end

(* Search History *)

module History : sig

    (* Search Records *)

    module Record : sig
        type t
        (* Records maintain the selected features and the context in which they were chosen *)

        val make : float list -> float list list -> t
        val features : t -> float list

        module JSON : sig
            val encode : t -> JSON.t
            val decode : JSON.t -> t option
        end
    end

    type t
    
    val extend : t -> Record.t -> t
    val empty : t

    val most_recent_features : t -> float list

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

module State : sig
    type 'a t

    val make : 'a -> History.t -> 'a t
    val value : 'a t -> 'a
    val history : 'a t -> History.t

    val to_tuple : 'a t -> ('a * History.t)

    val check : ('a -> bool) -> 'a t -> bool

    val extend_history : 'a t -> History.Record.t -> 'a t
    val features : 'a t -> float list
end

val stochastic_beam_search : (module Domain with type t = 'a) -> int -> 'a State.t list -> 'a State.t list -> 'a State.t list
