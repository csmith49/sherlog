type t = Watson.Semantics.Program.t

open Interface.JSON

let rec to_json program = `List (program 
    |> Watson.Semantics.Program.to_list 
    |> CCList.map rule_to_json)
and rule_to_json = function
    | Watson.Semantics.Rule.Rule (head, body) -> `Assoc [
        ("head", atom_to_json head);
        ("body", `List (CCList.map atom_to_json body));
    ]
and atom_to_json = function
    | Watson.Syntax.Atom.Atom (s, args) -> `Assoc [
        ("kind", `String "atom");
        ("relation", `String s);
        ("arguments", `List (CCList.map term_to_json args));]
    | Watson.Syntax.Atom.Intro (ob, p, c, v) -> `Assoc [
        ("kind", `String "intro");
        ("obligation", obligation_to_json ob);
        ("parameters", `List (CCList.map term_to_json p));
        ("context", `List (CCList.map term_to_json c));
        ("values", `List (CCList.map term_to_json v));]
and term_to_json = function
    | Watson.Syntax.Term.Variable x -> `Assoc [
        mk_type "variable";
        x |> Make.string |> mk_value;
    ]
    | Integer i -> `Assoc [
        mk_type "integer";
        i |> Make.int |> mk_value;
    ]
    | Float f -> `Assoc [
        mk_type "float";
        f |> Make.float |> mk_value;
    ]
    | Boolean b -> `Assoc [
        mk_type "boolean";
        b |> Make.bool |> mk_value;
    ]
    | Constant c -> `Assoc [
        mk_type "constant";
        c |> Make.string |> mk_value;
    ]
    | Function (f, fs) -> `Assoc [
        mk_type "function";
        ("function", `String f);
        ("arguments", fs |> CCList.map term_to_json |> Make.list);
    ]
and mk_type typ = ("type", `String typ)
and mk_value v = ("value", v)
and obligation_to_json = function
    | Watson.Syntax.Obligation.Assign f -> `Assoc [
        ("obligation", `String "assign");
        ("value", `String f);]

let rec of_json json = Parse.(list rule_of_json json) |> CCOpt.map Watson.Semantics.Program.of_list
and rule_of_json = function
    | (`Assoc _) as json ->
        let head = Parse.(find atom_of_json "head" json) in
        let body = Parse.(find (list atom_of_json) "body" json) in
        CCOpt.map2 (fun head -> fun body -> Watson.Semantics.Rule.Rule (head, body)) head body
    | _ -> None
and atom_of_json = function
    | (`Assoc _) as json ->
        begin match Parse.(find string "kind" json) with
            | Some "atom" ->
                let rel = Parse.(find string "relation" json) in
                let args = Parse.(find (list term_of_json) "arguments" json) in
                    CCOpt.map2 (fun rel -> fun args -> Watson.Syntax.Atom.Atom (rel, args)) rel args
            | Some "intro" ->
                let obligation = Parse.(find obligation_of_json "obligation" json) in
                let parameters = Parse.(find (list term_of_json) "parameters" json) in
                let context = Parse.(find (list term_of_json) "context" json) in
                let values = Parse.(find (list term_of_json) "values" json) in
                begin match obligation, parameters, context, values with
                    | Some ob, Some p, Some c, Some v -> Some (Watson.Syntax.Atom.Intro (ob, p, c, v))
                    | _ -> None end
            | _ -> None
        end
    | _ -> None
and term_of_json = function
    | (`Assoc _) as json ->
        (* get the type *)
        begin match Parse.(find string "type" json) with
            | Some "integer" -> json
                |> Parse.(find int "value")
                |> CCOpt.map (fun i -> Watson.Syntax.Term.Integer i)
            | Some "float" -> json
                |> Parse.(find float "value")
                |> CCOpt.map (fun f -> Watson.Syntax.Term.Float f)
            | Some "boolean" -> json
                |> Parse.(find bool "value")
                |> CCOpt.map (fun b-> Watson.Syntax.Term.Boolean b)
            | Some "constant" -> json
                |> Parse.(find string "value")
                |> CCOpt.map (fun c -> Watson.Syntax.Term.Constant c)
            | Some "variable" -> json
                |> Parse.(find string "value")
                |> CCOpt.map (fun x -> Watson.Syntax.Term.Variable x)
            | Some "function" ->
                let f = Parse.(find string "function" json) in
                let args = Parse.(find (list term_of_json) "arguments" json) in
                CCOpt.map2 (fun f -> fun args -> Watson.Syntax.Term.Function (f, args)) f args
            | _ -> None
        end
    | _ -> None
and obligation_of_json = function
    | (`Assoc _) as json ->
        begin match Parse.(find string "obligation" json) with
            | Some "assign" ->
                let value = Parse.(find string "value" json) in
                CCOpt.map (fun v -> Watson.Syntax.Obligation.Assign v) value
            | _ -> None
        end
    | _ -> None

let query_of_json json = Parse.(list atom_of_json json)

let rec problem_to_json problem = let open Interface.JSON in `Assoc [
    ("parameters", problem.Problem.parameters |> CCList.map Problem.Parameter.to_json |> Make.list);
    ("functions", problem.Problem.functions |> CCList.map Problem.Function.to_json |> Make.list);
    ("evidence", problem.Problem.evidence |> CCList.map atom_list_to_json |> Make.list);
    ("queries", problem.Problem.queries |> CCList.map atom_list_to_json |> Make.list);
    ("program", problem.Problem.program |> to_json);
]
and atom_list_to_json atoms = `List (atoms |> CCList.map atom_to_json)