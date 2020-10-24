module Cache = struct
    module AtomSet = CCSet.Make(Language.Atom)

    type t = AtomSet.t

    let empty = AtomSet.empty
    let add = AtomSet.add
    let mem = AtomSet.mem
end

module Goal = struct
    type t = Language.Atom.t list

    let of_list xs = xs

    let discharge goal = 
        if CCList.is_empty goal then None
        else Some (CCList.hd_tl goal)

    let map goal m = CCList.map (fun atom -> Language.Atom.map atom m) goal
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

        (* use Goal.discharge, but check that the discharged atom hasn't already been unfolded and cache the result *)
        let rec discharge state = match Goal.discharge state.goal with
        | Some (atom, goal) ->
            let state = {state with goal = goal} in
            if Cache.mem atom state.cache then
                discharge state
            else
                let state = {state with cache = Cache.add atom state.cache} in
                Some (atom, state)
        | None -> None

        let extend_goal atoms state = {
            state with goal = state.goal @ atoms
        }

        let map state m = {
            state with goal = Goal.map state.goal m
        }
    end

    module Label = struct
        type t = {
            obligation : Logic.Obligation.t;
            map : Language.Term.Map.t;
        }

        let initial = {
            obligation = Logic.Obligation.True;
            map = Language.Term.Map.empty;
        }

        let obligation lbl = lbl.obligation

        let map lbl = lbl.map
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

    (* filter for determining which nodes we can expand *)
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
                let resolutions = CCList.filter_map (Semantics.Rule.resolve atom) program in
                let make_node (map, obligation, atoms) =
                    let label = {
                        Label.obligation = obligation;
                        map = map;
                    } in
                    let state = state
                        |> State.extend_goal atoms
                        |> (fun s -> State.map s map) in
                    Resolution (label, state)
                in CCList.map make_node resolutions
            | None -> [Success] end

    let rec resolve_zipper program zipper = match Data.Tree.find is_expandable zipper with
        | Some zipper ->
            (* compute the current visible node *)
            let node = zipper |> Data.Tree.focus |> Data.Tree.label in
            (* use the strategy to compute the successor states in the search *)
            let children = expand_node program node |> CCList.map Data.Tree.leaf in
            let subtree = Data.Tree.Node (node, children) in
            (* rebuild zipper and recurse *)
            let zipper = zipper |> Data.Tree.set_focus subtree in
            resolve_zipper program zipper
        | None -> zipper

    let resolve program tree = tree
        |> Data.Tree.zipper
        |> resolve_zipper program
        |> Data.Tree.of_zipper
end