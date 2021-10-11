module GroundTerm = struct
    type t =
        | Integer of int
        | Float of float
        | Boolean of bool
        | Function of string * t list
        | Unit

    let rec pp ppf = function
        | Integer i -> Fmt.pf ppf "%d" i
        | Float f -> Fmt.pf ppf "%f" f
        | Boolean true -> Fmt.pf ppf "T"
        | Boolean false -> Fmt.pf ppf "F"
        | Unit -> Fmt.pf ppf "*"
        | Function (f, args) -> Fmt.pf ppf "%s(%a)"
            f
            (Fmt.list ~sep:Fmt.comma pp) args

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

    let lift : Watson.Term.t -> t Pipe.Value.t option = function
        | Watson.Term.Variable id | Watson.Term.Symbol id -> Some (Pipe.Value.Identifier id)
        | (_ as term) -> term
            |> of_term
            |> CCOpt.map (fun term -> Pipe.Value.Literal term)
end

module Observation = struct
    type t = (string * GroundTerm.t Pipe.Value.t) list

    let pp ppf _ = Fmt.pf ppf "OBS"

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
            ("items", obs |> JSON.Encode.assoc (Pipe.Value.JSON.encode GroundTerm.JSON.encode));
        ]
    end
end

type t = {
    pipeline : GroundTerm.t Pipe.Pipeline.t;
    observations : Observation.t list;
    history : Search.History.t;
}

let pp ppf ex = Fmt.pf ppf "Pipeline:\n%a\nObservations:\n%a\n"
    (Pipe.Pipeline.pp GroundTerm.pp) ex.pipeline
    (Fmt.list Observation.pp) ex.observations

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
        ("pipeline", ex |> Functional.pipeline |> Pipe.Pipeline.JSON.encode GroundTerm.JSON.encode);
        ("observations", ex |> Functional.observations |> JSON.Encode.list Observation.JSON.encode);
        ("history", ex |> Functional.history |> Search.History.JSON.encode);
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
    let statement sub intro =
            let target = Introduction.sample_site intro in
            let function_id = Introduction.Functional.function_id intro in
            let arguments = Introduction.Functional.arguments intro
                |> CCList.map (Watson.Substitution.apply sub)
                |> CCList.map GroundTerm.lift
                |> CCList.all_some
                |> CCOpt.get_exn_or "Invalid Watson terms in introduction." in
            Pipe.Statement.Functional.make target function_id arguments in
    let pipeline = introductions
        |> CCList.map (statement substitution)
        |> Pipe.Pipeline.Functional.make in
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