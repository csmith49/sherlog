type t = [
    | `Rule of Watson.Rule.t
    | `Evidence of Evidence.t
    | `Parameter of Parameter.t
]

type line = t

module Generation : sig
    type form = [
        | `Introduction of string * Watson.Term.t list
        | `DiscreteEmbedding of Watson.Term.t list * string * Watson.Term.t list
        | `SoftmaxEmbedding of Watson.Term.t list * string * Watson.Term.t list
    ]
    
    type t

    val make : string -> Watson.Term.t list -> form -> Watson.Atom.t list -> t

    val compile : t -> line list
end