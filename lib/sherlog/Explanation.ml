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
end

module Observation = struct
    type t = (string * GroundTerm.t Pipeline.Value.t) list

    module JSON = struct
        let encode obs = `Assoc [
            ("type", `String "observation");
            ("items", obs
                |> CCList.map (CCPair.map_snd (Pipeline.Value.to_json GroundTerm.JSON.encode))
                |> JSON.Make.assoc);
        ]
    end
end

type t = {
    pipeline : GroundTerm.t Pipeline.t;
    observations : Observation.t list;
    history : Search.History.t;
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
        ("observations", ex |> Functional.observations |> CCList.map Observation.JSON.encode |> JSON.Make.list);
        ("history", ex |> Functional.history |> Search.History.JSON.encode);
    ]
end

module Compile = struct
    
    let lift : Watson.Term.t -> GroundTerm.t Pipeline.Value.t option = function
        | Watson.Term.Variable id | Watson.Term.Symbol id -> Some (Pipeline.Value.Identifier id)
        | (_ as term) -> term
            |> GroundTerm.of_term
            |> CCOpt.map (fun term -> Pipeline.Value.Literal term)
    
    module Branch = struct
        type t = Proof.Tag.t Data.Tree.path

        let is_successful branch = match CCList.last_opt branch with
            | Some Proof.Tag.Success -> true
            | _ -> false

        let introductions branch = branch
            |> CCList.filter_map Proof.Tag.witness
            |> CCList.map Watson.Proof.Witness.resolved_atom
            |> CCList.filter_map Introduction.of_atom

        let observation : t -> Observation.t = fun branch ->
            let assoc intro =
                if Introduction.observed intro then
                    let value = Introduction.Functional.target intro
                        |> lift
                        |> CCOpt.get_exn_or "Target not a ground term" in
                    let site = Introduction.sample_site intro in
                    Some (site, value)
                else None in
            branch
                |> introductions
                |> CCList.filter_map assoc
    end

    module Tree = struct
        type t = Proof.Tag.t Data.Tree.tree

        let of_proof (proof : Proof.t) : t =
            let algebra node children = Data.Tree.Node (Proof.Node.tag node, children) in
            Data.Tree.eval algebra proof

        let branches = Data.Tree.paths

        let introductions tree = tree
            |> branches
            |> CCList.filter Branch.is_successful
            |> CCList.flat_map Branch.introductions
            |> CCList.uniq ~eq:Introduction.equal
    end
end

let of_proof proof history =
    (* get all introductions in the proof *)
    let introductions = proof
        |> Compile.Tree.of_proof
        |> Compile.Tree.introductions in
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
                |> CCList.map Compile.lift
                |> CCList.all_some
                |> CCOpt.get_exn_or "Invalid Watson terms in introduction.";
        } in
    let pipeline = introductions
        |> CCList.map (statement substitution) in
    (* and observations *)
    let observations = proof
        |> Compile.Tree.of_proof
        |> Compile.Tree.branches
        |> CCList.filter Compile.Branch.is_successful
        |> CCList.map Compile.Branch.observation in
    (* put it all together *)
    {
        pipeline=pipeline;
        observations=observations;
        history=history;
    }