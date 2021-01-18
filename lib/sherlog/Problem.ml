module Parameter = struct
    type domain =
        | Unit
        | Positive
        | Real
        (* | Categorical of int *)

    let domain_to_string = function
        | Unit -> "[0, 1]"
        | Positive -> "ℝ⁺"
        | Real -> "ℝ"
        (* | Categorical n -> "Cat[" ^ (string_of_int n) ^ "]" *)

    type t = Parameter of string * domain

    let to_string = function Parameter (name, domain) -> name ^ " : " ^ (domain_to_string domain)

    let to_json = function
        | Parameter (name, dom) ->
            let domain = match dom with
                | Unit -> `String "unit"
                | Positive -> `String "positive"
                | Real -> `String "real" in
            `Assoc [
                ("name", `String name);
                ("domain", domain);
            ]
end

module Namespace = struct
    type t = Namespace of string

    let to_string = function
        | Namespace n -> n

    let to_json = function
        | Namespace n -> `String n
end

module Evidence = struct
    type t =
        | Evidence of Watson.Atom.t list
        | ParameterizedEvidence of (string list) * string * Watson.Atom.t list

    let to_string = function
        | Evidence atoms -> atoms
            |> CCList.map Watson.Atom.to_string
            |> CCString.concat ", "
        | ParameterizedEvidence (bindings, source, atoms) ->
            let atoms = atoms
                |> CCList.map Watson.Atom.to_string
                |> CCString.concat ", " in
            let bindings = bindings
                |> CCString.concat ", " in
            bindings ^ " in " ^ source ^ " : " ^ atoms

    let to_json = function
        | Evidence atoms -> `Assoc [
            ("atoms", Utility.Atom.conjunct_to_json atoms);
        ]
        | ParameterizedEvidence (bindings, source, atoms) -> `Assoc [
            ("atoms", Utility.Atom.conjunct_to_json atoms);
            ("bindings", `List (CCList.map Interface.JSON.Make.string bindings));
            ("source", `String source)
        ]
end

type line = [
    | `Rule of Watson.Rule.t
    | `Query of Watson.Atom.t list
    | `Parameter of Parameter.t
    | `Namespace of Namespace.t
    | `Evidence of Evidence.t
]

let simplify_introduction
    ~relation:relation
    ~arguments:arguments
    ~generator:f
    ~parameters:parameters
    ~context:context
    ~body:body =
        let open Watson in
        (* fresh var to represent the value getting introduced *)
        let var = Term.Value (Term.Variable "_I") in
        (* building the introduction atom *)
        let guard = f in
        let intro = Atom.Intro (guard, parameters, context, var) in
        (* intro <- body *)
        let intro_rule = Rule.Rule (intro, body) in
        (* head <- body, intro *)
        let head = Atom.Atom (relation, arguments @ [var]) in
        let conclusion_rule = Rule.Rule (head, intro :: body) in
            [ `Rule intro_rule ; `Rule conclusion_rule ]

let simplify_fuzzy_fact
    ~probability:probability
    ~relation:relation
    ~arguments:arguments = simplify_introduction 
        ~relation:relation 
        ~arguments:arguments
        ~generator:"bernoulli"
        ~parameters:[probability]
        ~context:(Watson.Term.Value (Watson.Term.Constant relation) :: arguments)
        ~body:[]

type t = {
    parameters : Parameter.t list;
    namespaces : Namespace.t list;
    evidence : Evidence.t list;
    queries : (Watson.Atom.t list) list;
    program : Watson.Program.t;
}

let parameters p = p.parameters
let namespaces p = p.namespaces
let evidence p = p.evidence
let queries p = p.queries
let program p = p.program

let of_lines lines =
    let params = CCList.filter_map (function
        | `Parameter p -> Some p
        | _ -> None) lines in
    let names = CCList.filter_map (function
        | `Namespace n -> Some n
        | _ -> None) lines in
    let evidence = CCList.filter_map (function
        | `Evidence e -> Some e
        | _ -> None) lines in
    let queries = CCList.filter_map (function
        | `Query q -> Some q
        | _ -> None) lines in
    let rules = CCList.filter_map (function
        | `Rule r -> Some r
        | _ -> None) lines in
    {
        parameters = params;
        namespaces = names;
        evidence = evidence;
        queries = queries;
        program = Watson.Program.of_list rules;
    }

let to_string problem =
    let params = problem.parameters
        |> CCList.map (fun p -> "!parameter: " ^ (Parameter.to_string p)) in
    let names = problem.namespaces
        |> CCList.map (fun n -> "!namespace: " ^ (Namespace.to_string n)) in
    let rules = problem.program
        |> Watson.Program.to_list
        |> CCList.map Watson.Rule.to_string in
    let queries = problem.queries
        |> CCList.map (fun q -> q |> CCList.map Watson.Atom.to_string |> CCString.concat ", ")
        |> CCList.map (fun q -> q ^ "?") in
    let evidence = problem.evidence
        |> CCList.map Evidence.to_string
        |> CCList.map (fun e -> "!evidence: " ^ e) in
    (params @ names @ rules @ queries @ evidence)
        |> CCString.concat "\n"

let to_json problem = let open Interface.JSON in `Assoc [
    ("parameters", problem.parameters |> CCList.map Parameter.to_json |> Make.list);
    ("namespaces", problem.namespaces |> CCList.map Namespace.to_json |> Make.list);
    ("evidence", problem.evidence |> CCList.map Evidence.to_json |> Make.list);
    ("queries", problem.queries |> CCList.map Utility.Atom.conjunct_to_json|> Make.list);
    ("program", problem.program |> Utility.Program.to_json);
]
