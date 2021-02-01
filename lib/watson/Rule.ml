type t = {
    head : Atom.t;
    body : Atom.t list;
}

let make head body = {
    head = head;
    body = body;
}

let head rule = rule.head
let body rule = rule.body

let variables rule =
    let hvars = rule |> head |> Atom.variables in
    let bvars = rule |> body |> CCList.flat_map Atom.variables in
    Identifier.uniq (hvars @ bvars)

let apply h rule = {
    head = rule |> head |> Atom.apply h;
    body = rule |> body |> CCList.map (Atom.apply h);
}

let avoiding_rename ids rule =
    let index = Identifier.avoiding_index ids in
    let rename x = Term.Variable (Identifier.reindex index x) in
    let assoc = rule
        |> variables
        |> CCList.map (fun x -> (x, rename x)) in
    let substitution = Substitution.of_list assoc in
        apply substitution rule

module JSON = struct
    let encode rule = `Assoc [
        ("type", `String "rule");
        ("head", rule |> head |> Atom.JSON.encode);
        ("body", `List (rule |> body |> CCList.map Atom.JSON.encode));
    ]

    let decode json =
        let head = JSON.Parse.(find Atom.JSON.decode "head" json) in
        let body = JSON.Parse.(find (list Atom.JSON.decode) "body" json) in
        CCOpt.map2 make head body
end