module State = struct
    type t = {
        goal : Atom.t list;
        cache : Atom.t list;
    }

    let goal state = state.goal
    let cache state = state.cache

    let of_atoms goal = {
        goal = goal; cache = [];
    }

    let rec discharge state = match goal state with
        | [] -> None
        | atom :: rest -> if CCList.mem ~eq:Atom.equal atom (cache state)
            then discharge { state with goal = rest }
            else let state = {
                goal = rest;
                cache = atom :: (cache state);
            } in Some (atom, state)

    let apply sub state = {
        goal = state |> goal |> CCList.map (Atom.apply sub);
        cache = state |> cache |> CCList.map (Atom.apply sub);
    }

    let extend atoms state = {
        state with goal = atoms @ (goal state);
    }

    let variables state = (goal state) @ (cache state)
        |> CCList.flat_map Atom.variables
        |> CCList.uniq ~eq:CCString.equal

    let is_empty state = match discharge state with
        | Some _ -> false
        | None -> true

    let reset_goal state goal = {
        state with goal = goal;
    }

    let pp ppf state = let open Fmt in
        pf ppf "%a" (list ~sep:comma Atom.pp) (goal state)

    let to_string = Fmt.to_to_string pp
end

module Witness = struct
    type t = {
        atom : Atom.t;
        rule : Rule.t;
        substitution : Substitution.t;
    }

    let atom w = w.atom
    let rule w = w.rule
    let substitution w = w.substitution

    let pp ppf w = let open Fmt in
        pf ppf "%a" Atom.pp (atom w)

    let to_string = Fmt.to_to_string pp
end

type t = O of State.t * (resolution option)
and resolution = R of Witness.t * t

let of_atoms goal =
    let state = State.of_atoms goal in
        O (state, None)

let rec rev_witnesses = function
    | O (_, None) -> []
    | O (_, Some R (w, o)) ->
        w :: (rev_witnesses o)
and witnesses proof = proof
    |> rev_witnesses
    |> CCList.rev

let is_resolved = function O (state, _) -> State.is_empty state

let state = function O (state, _) -> state

let of_state state = O (state, None)

let to_atoms proof = proof
    |> witnesses
    |> CCList.map Witness.atom

(* let to_atoms proof = proof
    |> state
    |> State.cache *)

(* let to_atoms proof =
    let witnesses = proof |> rev_witnesses in
    let sub = witnesses
        |> CCList.map Witness.substitution
        |> CCList.fold_left Substitution.compose Substitution.empty in
    let atoms = witnesses
        |> CCList.map Witness.atom
        |> CCList.map (Atom.apply sub) in
    CCList.rev atoms *)

let resolve proof rule = let open CCOpt in
    (* change the rule variables to avoid those in the state *)
    let state = proof |> state in
    let rule = rule |> Rule.avoiding_rename (State.variables state) in
    (* unify a discharged atom with the rule head *)
    let* (atom, state) = state |> State.discharge in
    let* sub = Atom.unify (Rule.head rule) atom in
    (* build the witness with the sub, etc. *)
    let witness = {
        Witness.atom = atom |> Atom.apply sub;
        rule = rule;
        substitution = sub;
    } in
    (* rebuild the new state *)
    let state = state
        |> State.extend (Rule.body rule)
        |> State.apply sub in
    (* and extend the proof *)
    let resolution = R (witness, proof) in
        O (state, Some resolution) |> return

let pp ppf proof = let open Fmt in
    pf ppf "%a" (list ~sep:(any "@ => ") Witness.pp) (witnesses proof)