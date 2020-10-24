module Cache : sig
    type t
    (** caches are records of successfully discharged atoms, and are used to ensure resolution doesn't loop infinitely *)

    val empty : t
    (** [empty] is a default cache recording no discharged atoms *)

    val add : Language.Atom.t -> t -> t
    (** [add atom cache] extends [cache] with a record of the resolution of [atom] *)

    val mem : Language.Atom.t -> t -> bool
    (** [mem atom cache] checks if [atom] has been recorded in [cache] *)
end

module Goal : sig
    type t
    (** goals are conjunctions of atoms whose entailment we're interested in *)

    val of_list : Language.Atom.t list -> t
    (** [of_list atoms] constructs the goal consisting of the conjunction of [atoms] *)

    val discharge : t -> (Language.Atom.t * t) option
    (** [discharge goal] splits off an atom to be discharged, or returns [None] if one doesn't exist *)

    val map : t -> Language.Term.Map.t -> t
    (** [map goal m] applies map [m] to each atom in the conjunct [goal] *)
end

module Proof : sig
    module State : sig
        type t
        (** states are combinations of goals-to-be-resolved and caches of already-resolved atoms *)

        val of_goal : Goal.t -> t
        (** [of_goal goal] constructs an initial state from a goal *)
        
        val discharge : t -> (Language.Atom.t * t) option
        (** [discharge state] breaks out an atom to be resolved, and [None] if one doesn't exist; the returned state has the cache updated with the discharged atom *)

        val extend_goal : Language.Atom.t list -> t -> t
        (** [extend_goal atoms state] extends the goal in [state] by conjoining the sub-goal [atoms] *)

        val map : t -> Language.Term.Map.t -> t
        (** [map state m] applies map [m] to all atoms in the goal of [state] *)
    end

    module Label : sig
        type t
        (** labels record the by-products of successful resolutions *)

        val initial : t
        (** [initial] is a default label recording no resolution *)

        val map : t -> Language.Term.Map.t
        (** [map label] is the map that resulted from the immediate resolution producing the state paired with [label] *)

        val obligation : t -> Logic.Obligation.t
        (** [obligation label] is the obligation generated from the immediate resolution that produced the state paired with [label] *)
    end

    type node =
        | Success
        | Failure
        | Resolution of Label.t * State.t
    (** each node records a terminal state -- [Success] or [Failure] -- or a resolution to be explored further *)

    type t = node Data.Tree.tree
    (** proofs are trees with nodes labeled by {!type:node} objects *)

    val of_query : Language.Atom.t list -> t
    (** [of_query atoms] builds an initial, unexplored resolution proof with the intent of answering the conjunctive query [atoms] *)

    val resolve : Semantics.Program.t -> t -> t
    (** [resolve program proof] uses the rules in [program] to resolve [proof] by expanding resolution nodes until all leaves in the proof tree are labeled with [Success] or [Failure] *)
end