(** {0 Watson} *)

(** {1 Syntax} *)

module Term : sig
    type value =
        | Variable of string
        | Constant of string
        | Integer of int
        | Float of float
        | Boolean of bool
    and t =
        | Wildcard
        | Unit
        | Value of value
        | Pair of t * t

    val to_string : t -> string
    (** [to_string term] produces a human-readable string representation of [term] *)

    val compare : t -> t -> int
    (** [compare left right] compares [left] and [right] with an arbitrary, but consistent, total order *)

    val equal : t -> t -> bool
    (** [equal left right] checks if [left] and [right] are syntactically equivalent *)

    val variables : t -> string list
    (** [variables term] produces a list of all variables (as strings) in [term] *)

    val is_ground : t -> bool
    (** [is_ground term] returns [true] if [term] has no variables, and [false] otherwise *)

    val occurs : string -> t -> bool
    (** [occurs id term] returns [true] if [Variable id] is a sub-term of [term] and [false] otherwise *)

    val map : (value -> t) -> t -> t
    (** [map f term] applies function [f] to all values in [term] while preserving the structure*)
end

module Map : sig
    type t
    (** a map transforms variables into terms *)

    val apply : t -> Term.t -> Term.t
    (** [apply map term] simultaneously maps all variables in [term] to their image in [map] *)

    val empty : t
    (** [empty] is a default map that acts as the identity when applied *)

    val singleton : string -> Term.t -> t
    (** [singleton id term] produces the map that maps [Variable id] to [term] and leaves every other variable unchanged *)

    val of_list : (string * Term.t) list -> t
    (** [of_list assoc] converts the association list [assoc] into a map *)

    val to_list : t -> (string * Term.t) list
    (** [to_list map] converts [map] to an association list *)

    val compose : t -> t -> t
    (** [compose left right] returns a map equivalent to applying [left] and then applying [right] *)

    module Unification : sig
        type equality
        (** equalities are unification constraints *)

        val equate : Term.t -> Term.t -> equality
        (** [equate left right] produces the unification constraint indicating we want a map [m] such that [m(left) == m(right)] *)

        val resolve_equalities : equality list -> t option
        (** [resolve_equalities eqs] resolves the list [eqs] and produces a map satisfying their indicated constraints, if able *)
    end
end

module Guard : sig
    type t = string
    
    val to_string : t -> string
    val applied_representation : Term.t -> Term.t list -> t -> string

    val compare : t -> t -> int
    val equal : t -> t -> bool
end

module Atom : sig
    type t =
        | Atom of string * Term.t list
        | Intro of Guard.t * Term.t list * Term.t list * Term.t
    (** atoms are the fundamental relational values in Watson *)

    val to_string : t -> string
    (** [to_string atom] produces a human-readable string representation of [atom] *)

    val variables : t -> string list
    (** [variables atom] produces a list of all variables (as strings) contained in terms inside [atom] *)

    val unify : t -> t -> Map.t option
    (** [unify left right] produces the most-general map [m] such that [m(left) == m(right)], if such a map exists *)

    val apply : Map.t -> t -> t
    (** [apply map atom] applies [map] to all variables in terms inside [atom] *)

    val compare : t -> t -> int
    (** [compare left right] compares [left] and [right] with an arbitrary, but consistent, total order *)

    val equal : t -> t -> bool
    (** [equal left right] checks for syntactic equality between [left] and [right] *)
end

(** {1 Semantics} *)

module Rule : sig
    type t = Rule of Atom.t * Atom.t list
    (** [Rule (head, body)] represents the Datalog rule [head :- body] *)

    val to_string : t -> string
    (** [to_string rule] produces a human-readable string representing [rule] *)

    val variables : t -> string list
    (** [variables rule] gives a list of all variables (as strings) in [rule] *)

    val apply : Map.t -> t -> t
    (** [apply map rule] applies [map] to each variable in [rule] *)

    val rename : t -> t
    (** [rename rule] renames all variables in [rule] to be universally-fresh *)

    val resolve : Atom.t -> t -> (Atom.t * Map.t * Atom.t list) option
    (** [resolve atom rule] attempts to unify the head of [rule] with [atom], producing the resolved atom, a unifying map, and a list of atoms sufficient to produce [atom] if possible; returns [None] otherwise *)
end

module Program : sig
    type t
    (** a program is a collection of rules *)

    val of_list : Rule.t list -> t
    (** [of_list rules] builds a program from the list [rules] *)

    val to_list : t -> Rule.t list

    val resolve : Atom.t -> t -> (Atom.t * Map.t * Atom.t list) list
    (** [resolve atom program] attempts to resolve [atom] using all rules in [program] non-deterministically *)
end

(** {1 Resolution} *)

module Goal : sig
    type t
    (** goals are conjunctions of atoms whose entailment we're interested in *)

    val of_list : Atom.t list -> t
    (** [of_list atoms] constructs the goal consisting of the conjunction of [atoms] *)

    val discharge : t -> (Atom.t * t) option
    (** [discharge goal] splits off an atom to be discharged, or returns [None] if one doesn't exist *)

    val apply : Map.t -> t -> t 
    (** [apply map goal] applies [map] to each atom in the conjunct [goal] *)

    val extend : Atom.t list -> t -> t
    (** [extend atoms goal] extends [goal] by conjoining [atoms] *)
end

module Proof : sig
    module State : sig
        type t
        (** states are combinations of goals-to-be-resolved and caches of already-resolved atoms *)

        val of_goal : Goal.t -> t
        (** [of_goal goal] constructs an initial state from a goal *)
        
        val discharge : t -> (Atom.t * t) option
        (** [discharge state] breaks out an atom to be resolved, and [None] if one doesn't exist; the returned state has the cache updated with the discharged atom *)

        val extend : Atom.t list -> t -> t
        (** [extend_goal atoms state] extends the goal in [state] by conjoining the sub-goal [atoms] *)

        val apply : Map.t -> t -> t
        (** [apply map state] applies [map] to all atoms in the goal of [state] *)
    end

    module Label : sig
        type t
        (** labels record the by-products of successful resolutions *)

        val initial : t
        (** [initial] is a default label recording no resolution *)

        val atom : t -> Atom.t
        (** [atom label] is the atom whose resolution produced the state paired with [label] *)

        val map : t -> Map.t
        (** [map label] is the map that resulted from the immediate resolution producing the state paired with [label] *)

        val subgoal : t -> Atom.t list
        (** [subgoal label] is the subgoal conjoined with the previous goal to produce the paired state *)
    end

    type node =
        | Success
        | Failure
        | Resolution of Label.t * State.t
    (** each node records a terminal state -- [Success] or [Failure] -- or a resolution to be explored further *)

    type t = node Data.Tree.tree
    (** proofs are trees with nodes labeled by {!type:node} objects *)

    val of_query : Atom.t list -> t
    (** [of_query atoms] builds an initial, unexplored resolution proof with the intent of answering the conjunctive query [atoms] *)

    val resolve : Program.t -> t -> t
    (** [resolve program proof] uses the rules in [program] to resolve [proof] by expanding resolution nodes until all leaves in the proof tree are labeled with [Success] or [Failure] *)

    module Random : sig
        type configuration = {
            depth : int;
            width : int;
            seeds : int;
        }

        val default_configuration : configuration

        val resolve : configuration -> Program.t -> Atom.t list -> t list
    end

    module Solution : sig
        type t

        val of_proof : node Data.Tree.tree -> t list
        val resolution_sequence : t -> Atom.t list
        val map : t -> Map.t

        val introductions : t -> (Guard.t * Term.t list * Term.t list * Term.t) list
    end
end
