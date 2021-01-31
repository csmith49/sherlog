module State = struct
    type t = {
        goal : Fact.t;
        cache : Fact.t;
    }

    let of_fact fact = {
        goal = fact;
        cache = Fact.empty;
    }

    let rec discharge state = match Fact.discharge state.goal with
        | Some (atom, fact) -> let state = { state with goal = fact } in
            if Fact.mem atom state.cache then discharge state
            else let state = { state with cache = Fact.add atom state.cache } in
                Some (atom, state)
        | _ -> None
    
    let is_empty state = match discharge state with
        | Some _ -> false
        | None -> true

    let variables state = Fact.variables state.goal

    let apply h state = { state with goal = state.goal |> Fact.apply h }

    let extend atoms state = let atoms = Fact.of_atoms atoms in
        { state with goal = state.goal |> Fact.conjoin atoms}
end

let resolve state rule = match State.discharge state with
    | Some (atom, state) ->
        let rule = Rule.avoiding_rename (State.variables state) rule in
        begin match Atom.unify atom (Rule.head rule) with
            | Some sub ->
                let state = state
                    |> State.extend (Rule.body rule)
                    |> State.apply sub in
                let atom = Atom.apply sub atom in
                    Some (atom, state)
            | None -> None
        end
    | None -> None