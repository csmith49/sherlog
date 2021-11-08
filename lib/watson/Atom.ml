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

let pp ppf atom = let open Fmt in
    pf ppf "%s(@[<0>%a@])" atom.relation (list ~sep:comma Term.pp) atom.terms

module Unification = struct
    type uni_c = Uni of Term.t * Term.t
    let uni_mk l r = Uni (l, r)
    let uni_map f = function Uni (l, r) -> Uni (f l, f r)

    let rec unify left right =
        if not (CCString.equal_caseless (relation left) (relation right)) then None else
        if not (CCInt.equal (arity left) (arity right)) then None else
        let eqs = CCList.map2 uni_mk (terms left) (terms right) in
            unify_aux eqs Substitution.empty
    and unify_aux eqs h = match eqs with
        | [] -> Some h
        | Uni (x, y) :: rest -> let open CCOpt in
            let* s = Substitution.Unification.unify x y in
            let h = Substitution.compose s h in
            let rest = CCList.map (uni_map (Substitution.apply h)) rest in
                (unify_aux rest h)
end

module JSON = struct
    let encode atom = `Assoc [
        ("type", JSON.Encode.string "atom");
        ("relation", atom |> relation |> JSON.Encode.string);
        ("terms", atom |> terms |> JSON.Encode.list Term.JSON.encode);
    ]

    let decode json = let open CCOpt in
        let* relation = JSON.Parse.(find "relation" string json) in
        let* terms = JSON.Parse.(find "terms" (list Term.JSON.decode) json) in
            return (make relation terms)
end

module Infix = struct
    let ($) = apply
end
