type 'a tree = Node of 'a * 'a tree list

let label = function
    | Node (n, _) -> n
let children = function
    | Node (_, children) -> children

let leaf n = Node (n, [])
let is_leaf = function
    | Node (_, []) -> true
    | _ -> false

type 'a path = 'a list
let rec paths = function
    | Node (n, []) -> [ [n] ]
    | Node (n, children) -> children
        |> CCList.flat_map paths
        |> CCList.map (fun path -> n :: path)
let rec of_path = function
    | [] -> None
    | n :: ns -> match of_path ns with
        | Some tree -> Some (Node (n, [tree]))
        | None -> Some (leaf n)

type 'a zipper = View of 'a tree * 'a choice list
and 'a choice = Choice of 'a * 'a tree list * 'a tree list

let zipper tree = View (tree, [])

let left = function
    | View (tree, Choice (node, left :: ls, rs) :: choices) ->
        let choice' = Choice (node, ls, tree :: rs) in
        Some (View (left, choice' :: choices))
    | _ -> None

let right = function
    | View (tree, Choice (node, ls, right :: rs) :: choices) ->
        let choice' = Choice (node, tree :: ls, rs) in
        Some (View (right, choice' :: choices))
    | _ -> None

let up = function
    | View (tree, Choice (node, ls, rs) :: choices) ->
        let children = (CCList.rev ls) @ (tree :: rs) in
        let tree' = Node (node, children) in
        Some (View (tree', choices))
    | _ -> None

let down = function
    | View (Node (n, child :: cs), choices) ->
        let choice = Choice (n, [], cs) in
        Some (View (child, choice :: choices))
    | _ -> None

let rec next z = let open CCOpt.Infix in
    (right z) <+> ((up z) >>= next)

let preorder z = let open CCOpt.Infix in
    (down z) <+> (next z)

let focus = function
    | View (tree, _) -> tree

let set_focus tree = function
    | View (_, choices) -> View (tree, choices)

let rec find pred z = let open CCOpt.Infix in
    let check = CCOpt.if_ (fun f -> pred (focus f)) z in
    check <+> ((preorder z) >>= (find pred))

let rec find_all pred z = match find pred z with
    | Some z' -> z' :: (find_all pred z')
    | None -> []

let rec path_to_focus zipper =
    let path = [label (focus zipper)] in
    match up zipper with
        | Some zipper -> (path_to_focus zipper) @ path
        | None -> path

let rec of_zipper zipper = match up zipper with
    | Some zipper -> of_zipper zipper
    | None -> focus zipper

type ('a, 'b) algebra = ('a -> 'b list -> 'b)
let rec eval alg = function
    | Node (n, children) ->
        let args = CCList.map (eval alg) children in
        alg n args