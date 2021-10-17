module Observation = struct
    type t = (string * Model.Value.t) list

    let pp ppf _ = Fmt.pf ppf "Observation"

    let of_introductions branch =
        let assoc_of_intro intro =
            if Introduction.observed intro then
                let domain = intro |> Introduction.sample_site in
                let codomain = intro
                    |> Introduction.Functional.target
                    |> Model.Value.of_term
                    |> CCOpt.get_exn_or "Invalid branch construction." in
                Some (domain, codomain)
            else None in
        CCList.filter_map assoc_of_intro branch

    module JSON = struct
        let encode obs = `Assoc [
            ("type", `String "observation");
            ("items", obs |> JSON.Encode.assoc Model.Value.JSON.encode);
        ]
    end
end

type t = {
    pipeline : Model.t;
    observations : Observation.t list;
    history : Search.History.t;
}

let pp ppf ex = Fmt.pf ppf "Pipeline:\n%a\nObservations:\n%a\n"
    Model.pp ex.pipeline
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

    let substitution branch = branch
        |> CCList.map Watson.Proof.Witness.substitution
        |> CCList.fold_left Watson.Substitution.compose Watson.Substitution.empty

    let introductions branch =
        let sub = substitution branch in
        branch
            |> CCList.map Watson.Proof.Witness.atom
            |> CCList.map (Watson.Atom.apply sub)
            |> CCList.filter_map Introduction.of_atom
end

let of_proof proof history =
    let branches = proof
        |> Branch.of_proof
        |> CCList.map Branch.introductions in
    let introductions = branches
        |> CCList.concat
        |> CCList.uniq ~eq:Introduction.equal in
    let pipeline = introductions
        |> CCList.filter_map Model.Statement.of_introduction
        |> Model.of_statements in
    {
        pipeline = pipeline;
        observations = branches |> CCList.map Observation.of_introductions;
        history = history;
    }
