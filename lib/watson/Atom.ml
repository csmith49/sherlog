type t = {
    relation : string;
    terms : Term.t list;
}

let make relation terms = {
    relation = relation;
    terms = terms;
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
    |> CCList.uniq ~eq:CCString.equal

let apply h atom = { atom with
    terms = atom.terms |> CCList.map (Substitution.apply h);
}

let unifiable left right =
    (CCString.equal_caseless (relation left) (relation right)) &&
        (CCInt.equal (arity left) (arity right))

let unify left right = 
    if unifiable left right then 
        let constraints = CCList.map2 Substitution.Unification.equate (terms left) (terms right) in
            Substitution.Unification.resolve_equalities constraints
    else 
        None

let pp ppf atom = let open Fmt in
    pf ppf "%s(@[<0>%a@])" atom.relation (list ~sep:comma Term.pp) atom.terms

module JSON = struct
    let encode atom = `Assoc [
        ("type", `String "atom");
        ("relation", `String (relation atom));
        ("terms", `List (atom |> terms |> CCList.map Term.JSON.encode));
    ]

    let decode json = let open CCOpt in
        let* relation = JSON.Parse.(find string "relation" json) in
        let* terms = JSON.Parse.(find (list Term.JSON.decode) "terms" json) in
            return (make relation terms)
end