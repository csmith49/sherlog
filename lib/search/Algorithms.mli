(* SEARCH SIGNATURES *)

module type Space = sig
    (* underlying state space *)
    type state

    (* witness of choices taken in search *)
    type witness

    (* check if a state is "complete", in some sense *)
    val is_goal : state -> bool

    (* get candidate choices and results from a state *)
    val next : state -> (witness * state) list
    
    (* rank each choice *)
    val embed : (witness * state) -> Embedding.t
end

module ExtendedSpace (S : Space) : sig
    (* wraps space with histories *)
    type state = S.state * History.t
    type witness = S.witness

    (* produces a full search space *)
    val is_goal : state -> bool
    val next : state -> (witness * state) list
    val embed : (witness * state) -> Embedding.t

    (* and a few utilities *)
    val of_state : S.state -> state
end

(** ['a search] represents a lazy search for an object of type ['a] *)
type ('a, 'b) search

(** [run search] evaluates [search] to produce a concrete solution and derivation history *)
val run : ('a, 'b) search -> ('a, 'b) Tree.path * History.t

(* ALGORITHMS *)
val complete_search : (module Space with type witness = 'a and type state = 'b) -> 'b -> ('a, 'b) search