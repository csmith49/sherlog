type t = {
    pipeline : Model.t;
    observations : Observation.t list;
    history : Search.History.t;
}

let pp ppf ex = Fmt.pf ppf "Pipeline:\n%a\nObservations:\n%a\n"
    Model.pp ex.pipeline
    (Fmt.list ~sep:Fmt.comma Observation.pp) ex.observations

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
        ("pipeline", ex |> Functional.pipeline |> Model.JSON.encode);
        ("observations", ex |> Functional.observations |> JSON.Encode.list Observation.JSON.encode);
        ("history", ex |> Functional.history |> Search.History.JSON.encode);
    ]
end

module Branch = struct
    let rec of_proof = function
        | Proof.Leaf (Proof.Success) -> [[]]
        | Interior edges ->
            let extend = function Proof.Edge (witness, proof) -> proof
                |> of_proof
                |> CCList.map (CCList.cons witness) in
            CCList.flat_map extend edges
        | _ -> []

    (* build a statement from a context and a witness *)
    let statement context witness = let open CCOpt in
        let* intro = witness
            |> Watson.Proof.Witness.resolved_atom
            |> Watson.Atom.apply context
            |> Introduction.of_atom in
        let target = Introduction.sample_site intro in
        let function_id = Introduction.Functional.function_id intro in
        let arguments = intro
            |> Introduction.Functional.arguments
            |> CCList.map (Watson.Substitution.apply context) in
        Model.Statement.make target function_id arguments

    (* get the full context from a witness *)
    let full_context context witness =
        let resolution_context = witness
            |> Watson.Proof.Witness.substitution in
        match witness |> Watson.Proof.Witness.resolved_atom |> Watson.Atom.apply context |> Introduction.of_atom with
            | None -> resolution_context
            | Some intro ->
                let sample_site = Introduction.sample_site intro in
                begin match Introduction.Functional.target intro |> Watson.Substitution.apply context with
                    | Watson.Term.Variable id  ->
                        let binding = Watson.Substitution.singleton id (Watson.Term.Variable sample_site) in
                        Watson.Substitution.compose resolution_context binding
                    | _ -> resolution_context
                end

    let observation context witness = match witness |> Watson.Proof.Witness.resolved_atom |> Watson.Atom.apply context |> Introduction.of_atom with
        | None -> []
        | Some intro ->
            let sample_site = Introduction.sample_site intro in
            begin match Introduction.Functional.target intro |> Watson.Substitution.apply context with
                | Watson.Term.Variable _ -> []
                | (_ as term) ->
                    let value = term
                        |> Model.Value.of_term
                        |> CCOpt.get_exn_or "Invalid value construction" in
                    [ (sample_site, value) ]
            end
    
    let rec compile witnesses = witnesses |> CCList.rev |> compile_aux Watson.Substitution.empty
    and compile_aux context = function
        | [] -> [], []
        | witness :: rest ->
            (* update context *)
            let context = context |> Watson.Substitution.compose (full_context context witness) in
            (* recurse *)
            let statements, observations = compile_aux context rest in
            let observations = (observation context witness) @ observations in
            match statement context witness with
                | Some statement -> statement :: statements, observations
                | _ -> statements, observations
            (* convert intro to statement *)
end
let of_proof proof history =
    let compilation_results = proof
        |> Branch.of_proof
        |> CCList.map Branch.compile in
    let statements = compilation_results
        |> CCList.flat_map fst
        |> CCList.uniq ~eq:Model.Statement.equal in
    let observations = compilation_results
        |> CCList.map snd
        |> CCList.map Observation.eq_of_assoc in
    {
        pipeline = Model.of_statements statements;
        observations = observations;
        history = history;
    }