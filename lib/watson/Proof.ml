module Obligation = struct
    type t = {
        goal : Atom.t list;
        cache : Atom.t list;
    }

    let compare left right =
        let goal_compare = CCList.compare Atom.compare left.goal right.goal in
        if goal_compare != 0 then goal_compare else
            CCList.compare Atom.compare left.cache right.cache

    let equal left right = (compare left right) == 0

    let rec discharge state = match state.goal with
        | [] -> None
        | atom :: rest when CCList.mem ~eq:Atom.equal atom state.cache ->
            discharge { state with goal = rest }
        | atom :: rest ->
            let state = {
                goal = rest;
                cache = atom :: state.cache;
            } in Some (atom, state)

    let is_empty state = state
        |> discharge
        |> CCOpt.is_none

    let extend atoms state = { state with
        goal = atoms @ state.goal;
    }

    let apply sub state = {
        goal = state.goal |> CCList.map (Atom.apply sub);
        cache = state.cache |> CCList.map (Atom.apply sub);
    }

    let variables state = state.goal @ state.cache
        |> CCList.flat_map Atom.variables
        |> CCList.uniq ~eq:CCString.equal

    let of_conjunct atoms = {
        goal = atoms;
        cache = [];
    }
end

module Witness = struct
    type t = {
        atom : Atom.t;
        rule : Rule.t;
        substitution : Substitution.t;
    }

    let compare left right =
        (* check atoms *)
        let atom_compare = Atom.compare left.atom right.atom in
        if atom_compare != 0 then atom_compare else
        (* then rules *)
        let rule_compare = Rule.compare left.rule right.rule in
        if rule_compare != 0 then rule_compare else
        (* then subs *)
        Substitution.compare left.substitution right.substitution
    let equal left right = (compare left right) == 0

    let atom witness = witness.atom
    let rule witness = witness.rule
    let substitution witness = witness.substitution

    let resolved_atom witness =
        Atom.apply witness.substitution witness.atom

    let to_string witness = witness
        |> resolved_atom
        |> Atom.to_string
end

let resolve obligation rule = let open CCOpt in
    (* change the rule vars to avoid those in the obligation *)
    let rule = rule
        |> Rule.avoiding_rename (Obligation.variables obligation) in
    (* unify a discharged atom w/ the rule head *)
    let* (atom, obligation) = Obligation.discharge obligation in
    let* substitution = Atom.Unification.unify (Rule.head rule) atom in
    let witness = {
        Witness.atom = atom;
        rule = rule;
        substitution = substitution;
    } in
    (* rebuild the new obligation *)
    let obligation = obligation
        |> Obligation.extend (Rule.body rule)
        |> Obligation.apply substitution in
    return (witness, obligation)

module Infix = struct
    let (>>) = Obligation.extend
    let ($) = Obligation.apply
    let (|=) = resolve
end