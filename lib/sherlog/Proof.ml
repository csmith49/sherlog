open Watson.Proof

type proof =
    | Leaf of leaf
    | Interior of edge list
and edge = Edge of Witness.t * proof
and leaf =
    | Frontier of Obligation.t
    | Success
    | Failure

let of_conjunct atoms =
    let obligation = Obligation.of_conjunct atoms in
        Leaf (Frontier obligation)

let rec trim = function
    | Leaf Failure -> None
    | Leaf _ as leaf -> Some leaf
    | Interior edges ->
        let edges = CCList.filter_map trim_edge edges in
        if CCList.is_empty edges then None else
            Some (Interior edges)
and trim_edge = function Edge (witness, proof) -> match trim proof with
    | None -> None
    | Some proof -> Some (Edge (witness, proof))

let obligation = function
    | Leaf (Frontier ob) -> Some ob
    | _ -> None

let rec pp ppf proof = let open Fmt in match proof with
    | Interior edges -> pf ppf "* => @[<1>{%a}@]" (list ~sep:comma edge_pp) edges
    | Leaf (Frontier obligation) -> begin match Watson.Proof.Obligation.discharge obligation with
            | Some (atom, _) -> pf ppf "{%a}" Watson.Atom.pp atom
            | _ -> pf ppf "{}"
        end
    | Leaf Success -> pf ppf "⟙"
    | Leaf Failure -> pf ppf "⟘"
and edge_pp ppf = function Edge (witness, proof) -> let open Fmt in pf ppf "%s : %a"
    (witness |> Watson.Proof.Witness.resolved_atom |> Watson.Atom.to_string)
    pp proof

module Zipper = struct
    type t = View of proof * context list
    and context = Context of Witness.t * edge list * edge list

    (* getters and setters *)
    let focus = function
        | View (proof, _) -> proof
    
    let set_focus proof = function
        | View (_, contexts) -> View (proof, contexts)

    (* basic movement *)
    let left = function
        | View (proof, Context (w, Edge (w', proof') :: ls, rs) :: rest) ->
            let edge = Edge (w, proof) in
            let context = Context (w', ls, edge :: rs) in
            Some (View (proof', context :: rest))
        | _ -> None

    let right = function
        | View (proof, Context (w, ls, Edge (w', proof') :: rs) :: rest) ->
            let edge = Edge (w, proof) in
            let context = Context (w', edge :: ls, rs) in
            Some (View (proof', context :: rest))
        | _ -> None

    let up = function
        | View (proof, Context (w, ls, rs) :: rest) ->
            let edge = Edge (w, proof) in
            let edges = (CCList.rev ls) @ (edge :: rs) in
            Some (View (Interior edges, rest))
        | _ -> None

    let down = function
        | View (Interior (Edge (w, proof) :: edges), contexts) ->
            let context = Context (w, [], edges) in
            Some (View (proof, context :: contexts))
        | _ -> None

    (* advanced movement *)
    let rec next z = let open CCOpt.Infix in
        (right z) <+> ((up z) >>= next)

    let preorder z = let open CCOpt.Infix in
        (down z) <+> (next z)

    let rec find pred z = let open CCOpt.Infix in
        let check = CCOpt.if_ (fun f -> pred (focus f)) z in
        check <+> ((preorder z) >>= (find pred))

    let rec find_all (pred : proof -> bool) (z : t) : t list = match find pred z with
        | Some z ->
            let children = z
                |> preorder
                |> CCOpt.map (find_all pred)
                |> CCOpt.get_or ~default: [] in
            z :: children
        | None -> []

    (* construction / conversion *)
    let of_proof proof = View (proof, [])

    let rec to_proof zipper = match up zipper with
        | Some zipper -> to_proof zipper
        | None -> focus zipper
end

module type Algebra = sig
    type result

    val leaf : leaf -> result
    val edge : Witness.t -> result -> result
    val interior : result list -> result
end

let rec eval : type a . (module Algebra with type result = a) -> proof -> a = fun (module A) -> function
    | Leaf leaf -> A.leaf leaf
    | Interior edges ->
        let f = function Edge (witness, proof) -> A.edge witness (eval (module A) proof) in
        edges
            |> CCList.map f
            |> A.interior
