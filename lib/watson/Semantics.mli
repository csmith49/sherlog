module Rule : sig
    type t = Rule of Syntax.Atom.t * Syntax.Atom.t list
    (** [Rule (head, body)] represents the Datalog rule [head :- body] *)

    val to_string : t -> string
    (** [to_string rule] produces a human-readable string representing [rule] *)

    val variables : t -> string list
    (** [variables rule] gives a list of all variables (as strings) in [rule] *)

    val apply : Syntax.Map.t -> t -> t
    (** [apply map rule] applies [map] to each variable in [rule] *)

    val rename : t -> t
    (** [rename rule] renames all variables in [rule] to be universally-fresh *)

    val resolve : Syntax.Atom.t -> t -> (Syntax.Atom.t * Syntax.Map.t * Syntax.Atom.t list) option
    (** [resolve atom rule] attempts to unify the head of [rule] with [atom], producing the resolved atom, a unifying map, and a list of atoms sufficient to produce [atom] if possible; returns [None] otherwise *)
end

module Program : sig
    type t
    (** a program is a collection of rules *)

    val of_list : Rule.t list -> t
    (** [of_list rules] builds a program from the list [rules] *)

    val resolve : Syntax.Atom.t -> t -> (Syntax.Atom.t * Syntax.Map.t * Syntax.Atom.t list) list
    (** [resolve atom program] attempts to resolve [atom] using all rules in [program] non-deterministically *)
end