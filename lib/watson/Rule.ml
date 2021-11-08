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

let compare left right =
    (* compare heads *)
    let head_compare = Atom.compare left.head right.head in
    if head_compare != 0 then head_compare else
    (CCList.compare Atom.compare) left.body right.body
let equal left right = (compare left right) == 0

let variables rule =
    let hvars = rule |> head |> Atom.variables in
    let bvars = rule |> body |> CCList.flat_map Atom.variables in
    CCList.uniq ~eq:CCString.equal (hvars @ bvars)

let apply h rule = {
    head = rule |> head |> Atom.apply h;
    body = rule |> body |> CCList.map (Atom.apply h);
}

let avoiding_rename vars rule =
    let rec avoid name index = match index with
        | None ->
            if CCList.mem ~eq:CCString.equal name vars
                then avoid name (Some 0)
                else Term.Variable name
        | Some i -> let name' = name ^ "_" ^ (CCInt.to_string i) in
            if CCList.mem ~eq:CCString.equal name' vars
                then avoid name (Some (i + 1))
                else Term.Variable name' in
    let assoc = rule
        |> variables
        |> CCList.map (fun x -> (x, avoid x None)) in
    let sub = Substitution.of_assoc assoc in
        apply sub rule

let pp ppf rule = let open Fmt in
    if CCInt.equal (CCList.length rule.body) 0
        then pf ppf "%a" Atom.pp rule.head
        else pf ppf "%a <- %a" Atom.pp rule.head (list ~sep:comma Atom.pp) rule.body

let to_string = Fmt.to_to_string pp

module JSON = struct
    let encode rule = `Assoc [
        ("type", `String "rule");
        ("head", rule |> head |> Atom.JSON.encode);
        ("body", `List (rule |> body |> CCList.map Atom.JSON.encode));
    ]

    let decode json = let open CCOpt in
        let* head = JSON.Parse.(find "head" Atom.JSON.decode json) in
        let* body = JSON.Parse.(find "body" (list Atom.JSON.decode) json) in
        return (make head body)
end
