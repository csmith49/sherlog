open Syntax
open Semantics

module Goal = struct
    type t = Atom.t list

    let of_list x = x
    let to_list x = x

    let apply map goal = goal
        |> to_list
        |> CCList.map (Atom.apply map)
        |> of_list

    let extend atoms goal = goal
        |> to_list
        |> CCList.append atoms
        |> of_list

    let discharge goal = match to_list goal with
        | [] -> None
        | x :: xs -> Some (x, of_list xs)
end

module Cache = struct
    (** Utility module for caching previously-unrolled values *)
    module AtomSet = CCSet.Make(Atom)
    type t = AtomSet.t

    let empty = AtomSet.empty
    let add = AtomSet.add
    let mem = AtomSet.mem
end

module Proof = struct
    module State = struct
        type t = {
            goal : Goal.t;
            cache : Cache.t;
        }

        let of_goal goal = {
            goal = goal;
            cache = Cache.empty;
        }

        let apply map state = {
            state with goal = Goal.apply map state.goal;
        }

        let extend atoms state = {
            state with goal = Goal.extend atoms state.goal;
        }

        let rec discharge state = match Goal.discharge state.goal with
            | Some (atom, goal) -> let state = {state with goal = goal} in
                if Cache.mem atom state.cache then
                    discharge state
                else
                    let state = {state with cache = Cache.add atom state.cache} in
                    Some (atom, state)
            | None -> None
    end

    module Label = struct
        type t = {
            atom : Atom.t;
            map : Map.t;
            subgoal : Atom.t list;
        }

        let initial = {
            atom = Atom.Atom ("root", []);
            map = Map.empty;
            subgoal = [];
        }

        let of_resolution (atom, map, subgoal) = {
            atom = atom;
            map = map;
            subgoal = subgoal;
        }

        let atom lbl = lbl.atom
        let map lbl = lbl.map
        let subgoal lbl = lbl.subgoal
    end

    type node =
        | Success
        | Failure
        | Resolution of Label.t * State.t

    type t = node Data.Tree.tree

    let of_query atoms =
        let label = Label.initial in
        let state = atoms
            |> Goal.of_list
            |> State.of_goal in
        let node = Resolution (label, state) in
            Data.Tree.leaf node

    (* filter for determining where we can and can't expand *)
    let is_expandable tree =
        if Data.Tree.is_leaf tree then 
            match Data.Tree.label tree with
                | Resolution _ -> true
                | _ -> false
        else false

    let expand_node program node = match node with
        | Success | Failure -> []
        | Resolution (_, state) -> begin match State.discharge state with
            | Some (atom, state) ->
                let resolutions = Program.resolve atom program in
                if CCList.is_empty resolutions then [Failure] else
                let make_node (atom, map, subgoal) =
                    let label = Label.of_resolution (atom, map, subgoal) in
                    let state = state
                        |> State.extend subgoal
                        |> State.apply map in
                    Resolution (label, state)
                in CCList.map make_node resolutions
            | None -> [Success] end

    let depth zipper = zipper
        |> Data.Tree.path_to_focus
        |> CCList.length

    let rec resolve_zipper max_depth program zipper = match Data.Tree.find is_expandable zipper with
        | Some zipper when depth zipper <= max_depth ->
            (* compute the current visible node *)
            let node = zipper |> Data.Tree.focus |> Data.Tree.label in
            (* use the strategy to compute the successor states in the search *)
            let children = expand_node program node |> CCList.map Data.Tree.leaf in
            let subtree = Data.Tree.Node (node, children) in
            (* rebuild zipper and recurse *)
            let zipper = zipper |> Data.Tree.set_focus subtree in
            resolve_zipper max_depth program zipper
        | _ -> zipper

    let resolve ?max_depth:(depth=CCInt.max_int) program tree = tree
        |> Data.Tree.zipper
        |> resolve_zipper depth program
        |> Data.Tree.of_zipper

    module Solution = struct
        type t = Label.t list

        let of_proof tree =
            let is_successful path = match CCList.last_opt path with
                | Some Success -> true
                | _ -> false in
            let labels path = path
                |> CCList.filter_map (function
                    | Resolution (lbl, _) -> Some lbl
                    | _ -> None
                ) in
            Data.Tree.paths tree
                |> CCList.filter is_successful
                |> CCList.map labels

        let resolution_sequence sol = CCList.map Label.atom sol

        let map sol = sol
            |> CCList.map Label.map
            |> CCList.fold_left Syntax.Map.compose Syntax.Map.empty

        let introductions sol = sol
            |> CCList.filter_map (fun lbl -> match Label.atom lbl with
                    | Syntax.Atom.Intro (ob, p, _, v) -> Some (ob, v, p)
                    | _ -> None
                )
    end
end
