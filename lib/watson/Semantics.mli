module Rule : sig
    type t

    val map : t -> Language.Term.Map.t -> t
    (** [map rule m] applies map [m] to all parts of [rule] *)

    val rename : t -> t
    (** [rename rule] renames all variables in [rule] to be fresh *)

    val resolve : Language.Atom.t -> t -> (Language.Term.Map.t * Logic.Obligation.t * Language.Atom.t list) option
    (** [resolve atom rule] attempts to unify the head of [rule] with [atom], producing a {!type:Language.Term.Map.t}, a {!type:Logic.Obligation.t}, and a list of atoms sufficient to produce [atom] if possible; returns [None] otherwise *)
end

module Program : sig
    type t
    (** a program is a collection of rules *)

    val of_list : Rule.t list -> t
    (** [of_list rules] builds a program from the list [rules] *)

    val rules : t -> Rule.t list
    (** [rules program] gives a list of all rules in [program] *)
end