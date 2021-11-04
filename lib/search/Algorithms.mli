(* SEARCH SIGNATURES *)

module type Space = sig
    (** type representing candidate solutions *)
    type candidate

    (** Required search interface *)

    (** [compare left right] returns a negative value if [left > right], a positive value if [left < right], and 0 on equality *)
    val compare : candidate -> candidate -> int

    (** [is_solution candidate] is true iff [candidate] is an acceptable solution from the search *)
    val is_solution : candidate -> bool

    (** [successors candidate] provides a list of follow-up candidates to explore *)
    val successors : candidate -> candidate list

    (** [embed candidate] produces a score embedding for [candidate] - used to break ties in searches *)
    val embed : candidate -> Embedding.t
end

(** ['a search] represents a lazy search for an object of type ['a] *)
type 'a search

(** [run search] evaluates [search] to produce a concrete solution and derivation history *)
val run : 'a search -> 'a * History.t

(* ALGORITHMS *)

val beam_search : (module Space with type candidate = 'a) -> int -> 'a -> 'a search