module Obligation : sig
    type t
    (* Obligations maintain atoms to-be-resolved *)

    val compare : t -> t -> int
    val equal : t -> t -> bool

    val discharge : t -> (Atom.t * t) option

    val is_empty : t -> bool

    val extend : Atom.t list -> t -> t

    val apply : Substitution.t -> t -> t
    val variables : t -> string list

    val of_conjunct : Atom.t list -> t
end

module Witness : sig
    type t
    (* Witness to obligation updates *)

    val compare : t -> t -> int
    val equal : t -> t -> bool

    val atom : t -> Atom.t
    (* [atom witness] is the conjunct whose resolution is captured by [witness]. *)
    
    val rule : t -> Rule.t
    (* [rule witness] *)

    val substitution : t -> Substitution.t
    (* [rule substitution] *)

    val resolved_atom : t -> Atom.t
    (* [resolved_atom witness] is the fact derived during the resolution captured by [witness]. *)

    val to_string : t -> string
end

val resolve : Obligation.t -> Rule.t -> (Witness.t * Obligation.t) option
(* attempt to resolve an atom from the obligation using the provided rule *)

(* Infix Notation *)

module Infix : sig
    val (>>) : Atom.t list -> Obligation.t -> Obligation.t
    (* Alias for [Obligation.extend] *)
    
    val ($) : Substitution.t -> Obligation.t -> Obligation.t
    (* Alias for [Obligation.apply] *)
    
    val (|=)  : Obligation.t -> Rule.t -> (Witness.t * Obligation.t) option
    (* Alias for [resolve] *)
end