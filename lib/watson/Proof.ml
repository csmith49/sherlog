module State = struct
    type t = {
        goal : Fact.t;
        cache : Fact.t;
    }

    let compare left right = match Fact.compare left.goal right.goal with
        | 0 -> Fact.compare left.cache right.cache
        | (_ as c) -> c
    
    let equal left right = (compare left right) == 0

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

    let resolve state rule = match discharge state with
        | Some (atom, state) ->
            let rule = Rule.avoiding_rename (variables state) rule in
            begin match Atom.unify (Rule.head rule) atom with
                | Some sub ->
                    let state = state
                        |> extend (Rule.body rule)
                        |> apply sub in
                    let atom = Atom.apply sub atom in
                        Some (atom, state)
                | None -> None
            end
        | None -> None
end

type resolution = Atom.t * State.t
type t = resolution list

let compare = CCList.compare (CCPair.compare Atom.compare State.compare)
let equal = CCList.equal (CCPair.equal Atom.equal State.equal)

let resolutions proof = proof
let extend proof resolution = proof @ [resolution]

let of_fact fact =
    let atom = Atom.make "true" [] in
    let state = State.of_fact fact in
    let resolution = (atom, state) in
        [resolution]
        
let to_fact proof = proof
    |> resolutions
    |> CCList.map fst
    |> Fact.of_atoms

let is_complete proof = match CCList.last_opt proof with
    | Some (_, state) -> State.is_empty state
    | _ -> false

let remaining_obligation proof = match CCList.last_opt proof with
    | Some (_, state) -> state.State.goal
    | _ -> Fact.empty

let length = CCList.length

let resolve proof rule = match CCList.last_opt proof with
    | Some (_, state) -> begin match State.resolve state rule with
        | Some resolution -> Some (extend proof resolution)
        | None -> None
    end
    | _ -> None