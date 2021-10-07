module GroundTerm = struct
    type t =
        | Integer of int
        | Float of float
        | Boolean of bool
        | Function of string * t list
        | Unit

    let rec of_term = function
        | Watson.Term.Integer i -> Some (Integer i)
        | Watson.Term.Float f -> Some (Float f)
        | Watson.Term.Boolean b -> Some (Boolean b)
        | Watson.Term.Unit -> Some Unit
        | Watson.Term.Function (f, args) -> args
            |> CCList.map of_term
            |> CCList.all_some
            |> CCOpt.map (fun args -> Function (f, args))
        | _ -> None

    module JSON = struct
        let rec encode = function
            | Integer i -> `Int i
            | Float f -> `Float f
            | Boolean b -> `Bool b
            | Unit -> `Null
            | Function (f, args) -> `Assoc [
                ("type", `String "function");
                ("function_id", `String f);
                ("arguments", `List (args |> CCList.map encode));
            ]
    end

    let lift : Watson.Term.t -> t Pipeline.Value.t option = function
        | Watson.Term.Variable id | Watson.Term.Symbol id -> Some (Pipeline.Value.Identifier id)
        | (_ as term) -> term
            |> of_term
            |> CCOpt.map (fun term -> Pipeline.Value.Literal term)
end

module Observation = struct
    type t = (string * GroundTerm.t Pipeline.Value.t) list

    let of_branch branch =
        let assoc_of_intro intro =
            if Introduction.observed intro then
                let domain = intro |> Introduction.sample_site in
                let codomain = intro
                    |> Introduction.Functional.target
                    |> GroundTerm.lift
                    |> CCOpt.get_exn_or "Invalid branch construction." in
                Some (domain, codomain)
            else None in
        CCList.filter_map assoc_of_intro branch

    module JSON = struct
        let encode obs = `Assoc [
            ("type", `String "observation");
            ("items", obs |> JSON.Encode.assoc (Pipeline.Value.to_json GroundTerm.JSON.encode));
        ]
    end
end

type t = {
    pipeline : GroundTerm.t Pipeline.t;
    observations : Observation.t list;
    history : Proof.Search.History.t;
}

module Functional = struct
    let pipeline ex = ex.pipeline
    let observations ex = ex.observations
    let history ex = ex.history

    let make pipeline observations history = {
        pipeline = pipeline;
        observations = observations;
        history = history;
    }
end

module JSON = struct
    let encode ex = `Assoc [
        ("type", `String "explanation");
        ("pipeline", ex |> Functional.pipeline |> Pipeline.to_json GroundTerm.JSON.encode);
        ("observations", ex |> Functional.observations |> JSON.Encode.list Observation.JSON.encode);
        ("history", ex |> Functional.history |> Proof.Search.History.JSON.encode);
    ]
end

let of_proof proof history =
    (* get all introductions in the proof *)
    let introductions = proof
        |> Proof.introductions in
    (* build the renaming *)
    let substitution = introductions
        |> CCList.filter_map (fun intro -> match Introduction.Functional.target intro with
                | Watson.Term.Variable x -> Some (x, Watson.Term.Variable (Introduction.sample_site intro))
                | _ -> None
            )
        |> Watson.Substitution.of_assoc in
    (* construct statements *)
    let statement sub intro = {
            Pipeline.Statement.target = Introduction.sample_site intro;
            function_id = Introduction.Functional.function_id intro;
            arguments = Introduction.Functional.arguments intro
                |> CCList.map (Watson.Substitution.apply sub)
                |> CCList.map GroundTerm.lift
                |> CCList.all_some
                |> CCOpt.get_exn_or "Invalid Watson terms in introduction.";
        } in
    let pipeline = introductions
        |> CCList.map (statement substitution) in
    (* and observations *)
    let observations = proof
        |> Proof.branches
        |> CCList.map Observation.of_branch in
    (* put it all together *)
    {
        pipeline=pipeline;
        observations=observations;
        history=history;
    }