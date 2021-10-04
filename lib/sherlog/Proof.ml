module Tag  = struct
    type t =
        | Root
        | Success
        | Failure
        | Witness of Watson.Proof.Witness.t

    (* utilities and filters *)
    let interior = function
        | Root | Witness _ -> true
        | _ -> false

    let exterior = function
        | Success | Failure -> true
        | _ -> false

    let witness = function
        | Witness w -> Some w
        | _ -> None
end

module Node = struct
    type t = Tag.t * Watson.Proof.Obligation.t

    let tag = fst
    let obligation = snd
end

type t = Node.t Search.tree

let of_conjunct atoms = Data.Tree.leaf (Tag.Root, Watson.Proof.Obligation.of_conjunct atoms)
let of_witness witness obligation = Data.Tree.leaf (Tag.Witness witness, obligation)

let success obligation = Data.Tree.leaf (Tag.Success, obligation)
let failure obligation = Data.Tree.leaf (Tag.Failure, obligation)