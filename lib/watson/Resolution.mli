module Goal : sig
    type t
    (** goals are conjunctions of atoms whose entailment we're interested in *)

    val of_list : Syntax.Atom.t list -> t
    (** [of_list atoms] constructs the goal consisting of the conjunction of [atoms] *)

    val discharge : t -> (Syntax.Atom.t * t) option
    (** [discharge goal] splits off an atom to be discharged, or returns [None] if one doesn't exist *)

    val apply : Syntax.Map.t -> t -> t 
    (** [apply map goal] applies [map] to each atom in the conjunct [goal] *)

    val extend : Syntax.Atom.t list -> t -> t
    (** [extend atoms goal] extends [goal] by conjoining [atoms] *)
end

module Proof : sig
    module State : sig
        type t
        (** states are combinations of goals-to-be-resolved and caches of already-resolved atoms *)

        val of_goal : Goal.t -> t
        (** [of_goal goal] constructs an initial state from a goal *)
        
        val discharge : t -> (Syntax.Atom.t * t) option
        (** [discharge state] breaks out an atom to be resolved, and [None] if one doesn't exist; the returned state has the cache updated with the discharged atom *)

        val extend : Syntax.Atom.t list -> t -> t
        (** [extend_goal atoms state] extends the goal in [state] by conjoining the sub-goal [atoms] *)

        val apply : Syntax.Map.t -> t -> t
        (** [apply map state] applies [map] to all atoms in the goal of [state] *)
    end

    module Label : sig
        type t
        (** labels record the by-products of successful resolutions *)

        val initial : t
        (** [initial] is a default label recording no resolution *)

        val atom : t -> Syntax.Atom.t
        (** [atom label] is the atom whose resolution produced the state paired with [label] *)

        val map : t -> Syntax.Map.t
        (** [map label] is the map that resulted from the immediate resolution producing the state paired with [label] *)

        val subgoal : t -> Syntax.Atom.t list
        (** [subgoal label] is the subgoal conjoined with the previous goal to produce the paired state *)
    end

    type node =
        | Success
        | Failure
        | Resolution of Label.t * State.t
    (** each node records a terminal state -- [Success] or [Failure] -- or a resolution to be explored further *)

    type t = node Data.Tree.tree
    (** proofs are trees with nodes labeled by {!type:node} objects *)

    val of_query : Syntax.Atom.t list -> t
    (** [of_query atoms] builds an initial, unexplored resolution proof with the intent of answering the conjunctive query [atoms] *)

    val resolve : Semantics.Program.t -> t -> t
    (** [resolve program proof] uses the rules in [program] to resolve [proof] by expanding resolution nodes until all leaves in the proof tree are labeled with [Success] or [Failure] *)

    module Random : sig
        type configuration = {
            depth : int;
            width : int;
            seeds : int;
        }

        val default_configuration : configuration

        val resolve : configuration -> Semantics.Program.t -> Syntax.Atom.t list -> t list
    end

    module Solution : sig
        type t

        val of_proof : node Data.Tree.tree -> t list
        val resolution_sequence : t -> Syntax.Atom.t list
        val map : t -> Syntax.Map.t

        val introductions : t -> (Syntax.Guard.t * Syntax.Term.t list * Syntax.Term.t list * Syntax.Term.t) list
    end
end
