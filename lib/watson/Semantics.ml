module Rule = struct
    open Language
    open Logic

    type t = {
        obligation : Obligation.t;
        head : Atom.t;
        body : Atom.t list;
    }

    let map rule m = {
        obligation = Obligation.map rule.obligation m;
        head = Atom.map rule.head m;
        body = CCList.map (fun a -> Atom.map a m) rule.body;
    }

    let renaming_index = ref 0

    let rename rule =
        (* get all the variables in the head and body *)
        let head_variables = Atom.variables rule.head in
        let body_variables = CCList.flat_map Atom.variables rule.body in
        let variables = CCList.uniq ~eq:CCString.equal (head_variables @ body_variables) in
        (* generate new names *)
        let renamings = CCList.map (fun v -> 
            let v' = v ^ "_" ^ (string_of_int !renaming_index) in
            (v, Term.Variable v')) variables in
        let _ = renaming_index := !renaming_index + 1 in
        let m = Term.Map.of_list renamings in
        map rule m

    let resolve atom rule =
        let rule = rename rule in
        match Atom.unify rule.head atom with
            | Some m ->
                let obligation = Obligation.map rule.obligation m in
                let subgoal = CCList.map (fun a -> Atom.map a m) rule.body in
                Some (m, obligation, subgoal)
            | None -> None
end

module Program = struct
    type t = Rule.t list

    let of_list xs = xs

    let rules program = program
end