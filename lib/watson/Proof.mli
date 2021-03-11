module State : sig
    type t

    val goal : t -> Atom.t list
    val cache : t -> Atom.t list
    
    val of_atoms : Atom.t list -> t
    val discharge: t -> (Atom.t * t) option

    val apply : Substitution.t -> t -> t
    val extend : Atom.t list -> t -> t
    val variables : t -> string list

    val is_empty : t -> bool

    val reset_goal : t -> Atom.t list -> t

    val pp : t Fmt.t
    val to_string : t -> string
end

module Witness : sig
    type t

    val atom : t -> Atom.t
    (** [atom w] returns the atom resolved in [w] *)
    
    val rule : t -> Rule.t
    (** [rule w] returns the rule used to resolve [atom w] *)

    val substitution : t -> Substitution.t
    (** [substitution w] returns the substitution used to resolve [atom w] with [rule w] *)
   
    val pp : t Fmt.t
    (** pretty printer for witnesses *)

    val to_string : t -> string
    (** string conversion for witnesses (derived from [pp]) *)
end

type t

val witnesses : t -> Witness.t list
(** [witnesses proof] returns a list of all witnesses of resolutions in [proof] *)

val state : t -> State.t
(** [state proof] returns the proof state containing the unproven obligations *)

val of_atoms : Atom.t list -> t
(** [of_atoms goal] builds a proof obligation for [goal] *)

val of_state : State.t -> t

val to_atoms : t -> Atom.t list
(** [to_atoms proof] returns a list of all resolved atoms from [proof] *)

val resolve : t -> Rule.t -> t option
(** [resolve proof rule] attempts to resolve an atom from the goal of the most recent resolution *)

val is_resolved : t -> bool
(** [is_resolved proof] returns [true] if all obligations have been satisfied *)

val pp : t Fmt.t
(** pretty printer for proofs *)