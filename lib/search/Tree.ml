(* TREES *)
type ('a, 'b) t =
    | Interior of ('a, 'b) edge list
    | Exterior of status * 'b
and ('a, 'b) edge = Edge of 'a * ('a, 'b) t
and status =
    | Frontier
    | Success
    | Failure

let state = function
    | Exterior (_, b) -> Some b
    | _ -> None

(* simple constructors *)
let success state = Exterior (Success, state)
let failure state = Exterior (Failure, state)
let frontier state = Exterior (Frontier, state)
let interior pairs = 
    let mk (w, s) = Edge (w, frontier s) in
    Interior (pairs |> CCList.map mk)

let is_frontier = function
    | Exterior (Frontier, _) -> true
    | _ -> false

(* PATHS *)

type ('a, 'b) path = Path of 'a list * 'b

let of_state state = Path ([], state)
let prepend_witness witness = function
    | Path (witnesses, state) -> Path (witness :: witnesses, state)

let rec paths_to_success = function
    (* we want successes! *)
    | Exterior (Success, state) -> [of_state state]
    (* and no other exterior nodes! *)
    | Exterior (_, _) -> []
    (* and we'll recurse internally! *)
    | Interior edges -> CCList.flat_map paths_to_success_aux edges
and paths_to_success_aux = function
    | Edge (witness, tree) -> tree
        |> paths_to_success
        |> CCList.map (prepend_witness witness)

let witnesses_along_path = function
    | Path (witnesses, _) -> witnesses

(* ZIPPERS *)

module Zipper = struct
    type ('a, 'b) zipper = Zipper of ('a, 'b) t * ('a, 'b) context list
    and ('a, 'b) context = Context of ('a, 'b) edge list * 'a * ('a, 'b) edge list

    (* getters and setters *)
    let focus = function
        | Zipper (tree, _) -> tree
    
    let set_focus tree = function
        | Zipper (_, ctxs) -> Zipper (tree, ctxs)

    (* basic movement *)
    let left = function
        | Zipper (tree, Context (Edge (a', tree') :: ls, a, rs) :: rest) ->
            let edge = Edge (a, tree) in
            let ctx = Context (ls, a', edge :: rs) in
            Some (Zipper (tree', ctx :: rest))
        | _ -> None

    let right = function
        | Zipper (tree, Context (ls, a, Edge (a', tree') :: rs) :: rest) ->
            let edge = Edge (a, tree) in
            let ctx = Context (edge :: ls, a', rs) in
            Some (Zipper (tree', ctx :: rest))
        | _ -> None

    let up = function
        | Zipper (tree, Context (ls, a, rs) :: rest) ->
            let edges = (CCList.rev ls) @ (Edge (a, tree) :: rs) in
            Some (Zipper (Interior edges, rest))
        | _ -> None
    
    let down = function
        | Zipper (Interior (Edge (a, tree) :: edges), contexts) ->
            let ctx = Context ([], a, edges) in
            Some (Zipper (tree, ctx :: contexts))
        | _ -> None

    (* advanced movement *)
    let rec next z = let open CCOpt.Infix in
        (right z) <+> ((up z) >>= next)

    let preorder z = let open CCOpt.Infix in
        (down z) <+> (next z)

    let rec find pred z = let open CCOpt.Infix in
        let check = CCOpt.if_ (fun f -> pred (focus f)) z in
        check <+> ((preorder z) >>= (find pred))

    (* conversion *)
    let of_tree tree = Zipper (tree, [])
    let rec to_tree z = match up z with
        | Some z -> to_tree z
        | None -> focus z
end