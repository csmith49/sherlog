module Term = struct
    type t =
        | Integer of int
        | Float of float
        | Boolean of bool
        | Function of string * t list
        | Unit

    let rec of_watson_term = function
        | Watson.Term.Integer i -> Some (Integer i)
        | Watson.Term.Float f -> Some (Float f)
        | Watson.Term.Boolean b -> Some (Boolean b)
        | Watson.Term.Unit -> Some Unit
        | Watson.Term.Function (f, args) -> args
                |> CCList.map of_watson_term
                |> CCList.all_some
                |> CCOpt.map (fun args -> Function (f, args))
        | _ -> None

    let rec to_json = function
        | Integer i -> `Int i
        | Float f -> `Float f
        | Boolean b -> `Bool b
        | Unit -> `Null
        | Function (f, args) -> `Assoc [
            ("type", `String "function");
            ("function_id", `String f);
            ("arguments", `List (args |> CCList.map to_json));
        ]
end

module Observation = struct
    type 'a t = (string * 'a Pipeline.Value.t) list

    let to_json value_to_json obs = `Assoc [
        ("type", `String "observation");
        ("items", obs 
            |> CCList.map (CCPair.map_snd (Pipeline.Value.to_json value_to_json))
            |> JSON.Make.assoc);
    ]
end

type 'a t = {
    pipeline : 'a Pipeline.t;
    observation : 'a Observation.t;
    history : Search.History.t;
}

let pipeline ex = ex.pipeline
let observation ex = ex.observation
let history ex = ex.history

let to_json value_to_json ex = 
    let pipeline = ex
        |> pipeline
        |> Pipeline.to_json value_to_json in
    let observation = ex
        |> observation
        |> Observation.to_json value_to_json in
    let history = ex
        |> history
        |> Search.History.JSON.encode in
    
    `Assoc [
        ("type", `String "explanation");
        ("pipeline", pipeline);
        ("observation", observation);
        ("history", history);
    ]

module Compile = struct
    type 'a tag = ('a * Introduction.t)
    
    type 'a t = 'a tag list

    (* lift watson terms to pipeline values *)
    let lift : Watson.Term.t -> Term.t Pipeline.Value.t option = function 
        | Watson.Term.Variable id | Watson.Term.Symbol id -> Some (Pipeline.Value.Identifier id)
        | (_ as term) -> term
            |> Term.of_watson_term
            |> CCOpt.map (fun term -> Pipeline.Value.Literal term)

    module Target = struct
        type target = (string * Term.t Pipeline.Statement.t)

        let pipeline (intros : target t) : Term.t Pipeline.t = intros
            |> CCList.map (fun ((_, statement), _) -> statement)

        let observation (intros : target t) : Term.t Observation.t = 
            let f ((_, statement), intro) = 
                (* check if we've seen a value we must constrain *)
                (* note the difference in treatment between variables and symbols *)
                (* that's due to a semantic shift between Watson variables and Pipeline identifiers *)
                (* TODO: convert to Introduction.is_constrained *)
                let seen = match Introduction.Functional.target intro with
                    | Watson.Term.Variable _ -> None
                    | (_ as term) -> lift term in
                (* take `seen` to `(target, seen)` *)
                seen |> CCOpt.map (CCPair.make statement.Pipeline.Statement.target) in
            CCList.filter_map f intros
    end

    let of_proof (proof : Watson.Proof.t) : unit t = proof
        |> Watson.Proof.to_atoms
        |> CCList.filter_map Introduction.of_atom
        |> CCList.map (CCPair.make ())

    let associate_names (intros : unit t) : string t = intros
        |> CCList.map (function (_, intro) -> (Introduction.Functional.context intro |> Introduction.Context.to_string, intro))

    let target_renaming (intros : string t) : Watson.Substitution.t =
        (* maps the target to the pre-computed name of the intro *)
        let f (name, intro) = match Introduction.Functional.target intro with
            | Watson.Term.Variable x ->
                let target = Watson.Term.Variable name in
                (Some (x, target), intro)
            | _ -> (None, intro) in
        (* collect the non-None tags and combine into sub *)
        CCList.map f intros
            |> CCList.map fst
            |> CCList.keep_some
            |> Watson.Substitution.of_assoc

    let associate_statements (intros : string t) : Target.target t=
        (* get the target renamings *)
        let substitution = target_renaming intros in
        (* and build statements by remapping arguments *)
        let f (name, intro) = 
            let pipeline = {
                Pipeline.Statement.target = name;
                function_id = Introduction.Functional.function_id intro;
                arguments = Introduction.Functional.arguments intro
                    |> CCList.map (Watson.Substitution.apply substitution)
                    |> CCList.map lift
                    |> CCList.all_some
                    |> CCOpt.get_exn_or "Invalid Watson terms in introduction.";
            } in ((name, pipeline), intro) 
        in
        CCList.map f intros
end

let of_proof proof history =
    let compilation = proof
        |> Compile.of_proof
        |> Compile.associate_names
        |> Compile.associate_statements in
    {
        pipeline = compilation |> Compile.Target.pipeline;
        observation = compilation |> Compile.Target.observation;
        history = history
    }
