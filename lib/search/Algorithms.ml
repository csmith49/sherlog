(* SEARCH SIGNATURES *)

module type Space = sig
    type state
    type witness

    val is_goal : state -> bool
    val next : state -> (witness * state) list
    val embed : (witness * state) -> Embedding.t
end

type ('a, 'b) search = ('a, 'b * History.t) Tree.t CCRandom.t
let run search =
    let successes = search
        |> CCRandom.run
        |> Tree.paths_to_success
        |> CCList.map (function Tree.Path (witnesses, (state, history)) -> Tree.Path (witnesses, state), history) in
    let scores = successes
        |> CCList.map snd
        |> CCList.map History.score in
    Utility.prop_to successes scores |> CCRandom.run

module ExtendedSpace (S : Space) = struct
    (* wrap S.state with histories *)

    (* capture the types *)
    type state = S.state * History.t
    type witness = S.witness

    (* and extend the functional interface *)
    let is_goal (state, _) = S.is_goal state
    let next (state, history) = state
        |> S.next
        |> CCList.map (fun (witness, state) -> (witness, (state, history)))
    let embed (witness, (state, _)) = S.embed (witness, state)

    (* and add some extra utility *)
    let of_state state = (state, History.empty)
end

let complete_search (type a) (type b) (module S : Space with type state = a and type witness = b) (state : a) =
    let module ESpace = ExtendedSpace(S) in
    let initial = state |> ESpace.of_state |> Tree.frontier in
    let rec expand tree = match tree |> Tree.Zipper.of_tree |> Tree.Zipper.find Tree.is_frontier with
        | Some zipper ->
            (* get the state *)
            let state = zipper |> Tree.Zipper.focus |> Tree.state |> CCOpt.get_exn_or "Tree is guaranteed to be a frontier" in
            let focus =
                if ESpace.is_goal state then Tree.success state else
                begin match ESpace.next state with
                    | [] -> Tree.failure state
                    | pairs -> Tree.interior pairs
                end in
            zipper
                |> Tree.Zipper.set_focus focus
                |> Tree.Zipper.to_tree
                |> expand
        | None -> tree in
    expand initial |> CCRandom.return
