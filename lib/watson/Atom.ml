type t = {
    relation : string;
    terms : Term.t list;
}

let compare = Stdlib.compare
let equal l r = (compare l r) == 0

let relation atom = atom.relation
let terms atom = atom.terms

let arity atom = atom |> terms |> CCList.length

let to_string atom =
    let terms' = atom
        |> terms
        |> CCList.map Term.to_string
        |> CCString.concat ", " in
    let relation' = atom |> relation in
        relation' ^ "(" ^ terms' ^ ")"

let variables atom = atom
    |> terms
    |> CCList.flat_map Term.variables
    |> Identifier.uniq

let apply h atom = { atom with
    terms = atom.terms |> CCList.map (Homomorphism.apply h);
}

let unifiable left right =
    (CCString.equal (relation left) (relation right)) && (CCInt.equal (arity left) (arity right))
