module Parameter = struct
    type domain =
        | Unit
        | Positive
        | Real
        | Categorical of int

    let domain_to_string = function
        | Unit -> "[0, 1]"
        | Positive -> "ℝ⁺"
        | Real -> "ℝ"
        | Categorical n -> "Cat[" ^ (string_of_int n) ^ "]"

    type t = Parameter of string * domain

    let to_string = function Parameter (name, domain) -> name ^ " : " ^ (domain_to_string domain)
end

module Location = struct
    type t = Location of string * string

    let to_string = function Location (f, m) -> f ^ " @ " ^ m
end

module Function = struct
    type t = Function of string * Location.t

    let to_string = function Function (f, loc) -> f ^ " <- " ^ (Location.to_string loc)
end

module Line = struct
    type t =
        | Rule of Watson.Semantics.Rule.t
        | Parameter of Parameter.t
        | Query of Watson.Syntax.Atom.t list
        | Function of Function.t
        | Evidence of Watson.Syntax.Atom.t list

    let to_string = function
        | Rule r -> Watson.Semantics.Rule.to_string r
        | Parameter p -> "! " ^ (Parameter.to_string p)
        | Query q -> q
            |> CCList.map Watson.Syntax.Atom.to_string
            |> CCString.concat ", "
            |> (fun str -> str ^ "?")
        | Function f -> "! " ^ (Function.to_string f)
        | Evidence e -> e
            |> CCList.map Watson.Syntax.Atom.to_string
            |> CCString.concat ", "
            |> (fun str -> "! " ^ str)

    let lift_rule r = Rule r
end

let simplify_introduction relation arguments f parameters context body =
    let open Watson.Syntax in let open Watson.Semantics in
    (* fresh var to represent the value getting introduced *)
    let var = Term.Variable "_I" in
    (* building the introduction atom *)
    let obligation = Obligation.Assign f in
    let intro = Atom.Intro (obligation, parameters, context, [var]) in
    (* intro <- body *)
    let intro_rule = Rule.Rule (intro, body) in
    (* head <- body, intro *)
    let head = Atom.Atom (relation, arguments @ [var]) in
    let conclusion_rule = Rule.Rule (head, intro :: body) in
        [ intro_rule ; conclusion_rule ]

type t = {
    parameters : Parameter.t list;
    functions : Function.t list;
    evidence : (Watson.Syntax.Atom.t list) list;
    queries : (Watson.Syntax.Atom.t list) list;
    program : Watson.Semantics.Program.t;
}

let of_lines lines =
    let params = CCList.filter_map (function
        | Line.Parameter p -> Some p
        | _ -> None) lines in
    let funcs = CCList.filter_map (function
        | Line.Function f -> Some f
        | _ -> None) lines in
    let evidence = CCList.filter_map (function
        | Line.Evidence e -> Some e
        | _ -> None) lines in
    let queries = CCList.filter_map (function
        | Line.Query q -> Some q
        | _ -> None) lines in
    let rules = CCList.filter_map (function
        | Line.Rule r -> Some r
        | _ -> None) lines in
    {
        parameters = params;
        functions = funcs;
        evidence = evidence;
        queries = queries;
        program = Watson.Semantics.Program.of_list rules;
    }