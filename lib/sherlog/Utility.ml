module Term = struct
    type t = Watson.Term.t

    open Interface.JSON

    let rec to_json = function
        | Watson.Term.Value v -> begin match v with
            | Watson.Term.Variable x -> `Assoc [
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
            ] end
        | Unit -> `Assoc [mk_type "unit"]
        | Pair (l, r) -> `Assoc [
            mk_type "pair";
            ("left", l |> to_json);
            ("right", r |> to_json);
        ]
        | Wildcard -> `Null
    and mk_type typ = ("type", `String typ)
    and mk_value v = ("value", v)

    let rec of_json = function
        | (`Assoc _) as json -> begin match Parse.(find string "type" json) with
            | Some "unit" -> Some Watson.Term.Unit
            | Some "pair" ->
                let l = Parse.(find of_json "left" json) in
                let r = Parse.(find of_json "right" json) in
                begin match l, r with
                    | Some l, Some r -> Some (Watson.Term.Pair (l, r))
                    | _ -> None
                end
            | Some _ -> value_of_json json
            | None -> None
        end
        | _ -> None
    and value_of_json = function
        | (`Assoc _) as json ->
            let value = begin match Parse.(find string "type" json) with
                | Some "integer" -> json
                    |> Parse.(find int "value")
                    |> CCOpt.map (fun i -> Watson.Term.Integer i)
                | Some "float" -> json
                    |> Parse.(find float "value")
                    |> CCOpt.map (fun f -> Watson.Term.Float f)
                | Some "boolean" -> json
                    |> Parse.(find bool "value")
                    |> CCOpt.map (fun b-> Watson.Term.Boolean b)
                | Some "constant" -> json
                    |> Parse.(find string "value")
                    |> CCOpt.map (fun c -> Watson.Term.Constant c)
                | Some "variable" -> json
                    |> Parse.(find string "value")
                    |> CCOpt.map (fun x -> Watson.Term.Variable x) 
                | _ -> None end in
                CCOpt.map (fun v -> Watson.Term.Value v) value
        | _ -> None
end

module Guard = struct
    type t = Watson.Guard.t

    open Interface.JSON

    let to_json guard = `String (Watson.Guard.to_string guard) 

    let of_json = Parse.string
end

module Atom = struct
    type t = Watson.Atom.t

    open Interface.JSON

    let to_json = function
    | Watson.Atom.Atom (s, args) -> `Assoc [
        ("kind", `String "atom");
        ("relation", `String s);
        ("arguments", `List (CCList.map Term.to_json args));]
    | Watson.Atom.Intro (ob, p, c, v) -> `Assoc [
        ("kind", `String "intro");
        ("guard", Guard.to_json ob);
        ("parameters", `List (CCList.map Term.to_json p));
        ("context", `List (CCList.map Term.to_json c));
        ("target", Term.to_json v);
    ]

    let of_json = function
    | (`Assoc _) as json ->
        begin match Parse.(find string "kind" json) with
            | Some "atom" ->
                let rel = Parse.(find string "relation" json) in
                let args = Parse.(find (list Term.of_json) "arguments" json) in
                    CCOpt.map2 (fun rel -> fun args -> Watson.Atom.Atom (rel, args)) rel args
            | Some "intro" ->
                let guard = Parse.(find Guard.of_json "guard" json) in
                let parameters = Parse.(find (list Term.of_json) "parameters" json) in
                let context = Parse.(find (list Term.of_json) "context" json) in
                let target = Parse.(find Term.of_json "target" json) in
                begin match guard, parameters, context, target with
                    | Some ob, Some p, Some c, Some t -> Some (Watson.Atom.Intro (ob, p, c, t))
                    | _ -> None end
            | _ -> None
        end
    | _ -> None

    let conjunct_to_json c = `List (CCList.map to_json c)
    let conjunct_of_json json = Parse.(list of_json json)
end

module Rule = struct
    type t = Watson.Rule.t

    open Interface.JSON

    let to_json = function
        | Watson.Rule.Rule (head, body) -> `Assoc [
            ("head", Atom.to_json head);
            ("body", Atom.conjunct_to_json body);
        ]

    and of_json = function
    | (`Assoc _) as json ->
        let head = Parse.(find Atom.of_json "head" json) in
        let body = Parse.(find Atom.conjunct_of_json "body" json) in
        CCOpt.map2 (fun head -> fun body -> Watson.Rule.Rule (head, body)) head body
    | _ -> None
end

module Program = struct
    type t = Watson.Program.t

    open Interface.JSON

    let of_json json = Parse.(list Rule.of_json json) |> CCOpt.map Watson.Program.of_list

    let to_json program = `List (program 
        |> Watson.Program.to_list 
        |> CCList.map Rule.to_json
    )
end
