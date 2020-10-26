open Syntax

module Rule = struct
    type t = Rule of Atom.t * Atom.t list

    let to_string = function
        | Rule (Atom _ as head, body) ->
            let head = Atom.to_string head in
            let body = body
                |> CCList.map Atom.to_string
                |> CCString.concat ", " in
            head ^ " ← " ^ body
        | Rule (Intro (ob, _, _, v) as head, body) ->
            let head = Atom.to_string head in
            let body = body
                |> CCList.map Atom.to_string
                |> CCString.concat ", " in
            let values = v
                |> CCList.map Term.to_string
                |> CCString.concat ", " in
            let obligation = Obligation.to_string ob in
            "∃" ^ values ^ " ⊧ " ^ obligation ^ ". " ^ head ^ " ← " ^ body

    let variables = function
        | Rule (head, body) ->
            let hvars = Atom.variables head in
            let bvars = body
                |> CCList.flat_map Atom.variables in
            hvars @ bvars

    let apply map = function
        | Rule (head, body) ->
            let head = Atom.apply map head in
            let body = CCList.map (Atom.apply map) body in
                Rule (head, body)

    let renaming_index = ref 0

    let rename rule =
        let vars = rule
            |> variables
            |> CCList.uniq ~eq:CCString.equal in
        let map = vars
            |> CCList.map (fun v -> 
                    let v' = v ^ (string_of_int !renaming_index) in
                    (v, Term.Variable v')
                )
            |> Map.of_list in
        apply map rule

    let resolve atom rule = match rename rule with
        | Rule (head, body) -> match Atom.unify head atom with
            | Some map ->
                let subgoal = CCList.map (Atom.apply map) body in
                Some (Atom.apply map atom, map, subgoal)
            | None -> None
end

module Program = struct
    type t = Rule.t list

    let to_list x = x
    let of_list x = x

    let resolve atom program = program
        |> to_list
        |> CCList.filter_map (Rule.resolve atom)
end