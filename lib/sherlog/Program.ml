(* BASICS *)

type t = {
    rules : Watson.Rule.t list;
    parameters : Parameter.t list;
    posterior : Posterior.t;
}

(* FUNCTIONAL *)

module Functional = struct
    let make rules params posterior = {
        rules=rules;
        parameters=params;
        posterior=posterior;
    }

    let rules prog = prog.rules
    let parameters prog = prog.parameters
    let posterior prog = prog.posterior

    let is_introduction_rule rule = rule
        |> Watson.Rule.head
        |> Introduction.of_atom
        |> CCOpt.is_some

    let introduction_rules prog = prog
        |> rules
        |> CCList.filter is_introduction_rule

    let classical_rules prog = prog
        |> rules
        |> CCList.filter (fun r -> not (is_introduction_rule r))
end

(* ENCODING *)

module JSON = struct
    let encode prog = `Assoc [
        ("type", "program" |> JSON.Encode.string);
        ("rules", prog |> Functional.rules |> JSON.Encode.list Watson.Rule.JSON.encode);
        ("parameters", prog |> Functional.parameters |> JSON.Encode.list Parameter.JSON.encode);
        ("posterior", prog |> Functional.posterior |> Posterior.JSON.encode);
    ]

    let decode json = let open CCOpt in
        let* rules = JSON.Parse.(find "rules" (list Watson.Rule.JSON.decode) json) in
        let* params = JSON.Parse.(find "parameters" (list Parameter.JSON.decode) json) in
        let* posterior = JSON.Parse.(find "posterior" Posterior.JSON.decode json) in
            return (Functional.make rules params posterior)
end

(* APPLICATION *)

module Application = struct
    type t = Watson.Proof.Obligation.t -> Proof.proof option

    let success = Proof.Leaf Proof.Success
    let failure = Proof.Leaf Proof.Failure
    let edge (w, ob) = Proof.Edge (w, Proof.Leaf (Proof.Frontier ob))

    let check : ('a -> bool) -> 'b -> 'a -> 'b option = fun pred -> fun default -> fun v ->
        if pred v then Some default else None

    let branch (rules : Watson.Rule.t list) : t = fun ob ->
        let resolutions = rules
            |> CCList.map (Watson.Proof.resolve ob)
            |> CCList.keep_some in
        if CCList.is_empty resolutions then None else
        let edges = resolutions |> CCList.map edge in
        Some (Proof.Interior edges)

    let rec seq : ('a -> 'b option) list -> 'a -> 'b option = fun fs -> fun v -> match fs with
        | [] -> None
        | f :: fs -> CCOpt.Infix.(f v <+> seq fs v)
end

let apply obligation program =
    let classical = program
        |> Functional.classical_rules
        |> Application.branch in
    let introduction = program
        |> Functional.introduction_rules
        |> Application.branch in
    let result = obligation
        |> Application.seq [
            Application.check Watson.Proof.Obligation.is_empty Application.success;
            classical;
            introduction;
        ] in
    CCOpt.get_or ~default:Application.failure result

let search_structure : (Proof.proof -> bool) -> t -> (module Search.Structure with type candidate = Proof.proof) = fun pred -> fun program -> (module struct
    type candidate = Proof.proof

    let stop = pred

    let embed proof = program
        |> Functional.posterior
        |> Posterior.embed proof

    let score embedding = program
        |> Functional.posterior
        |> Posterior.score embedding

    let next proof =
        (* check if site has an expandable obligation *)
        let is_site proof = proof |> Proof.obligation |> CCOpt.is_some in
        (* perform expansion via backwards chaining *)
        let expand site = let open CCOpt in
            let* obligation = site
                |> Proof.Zipper.focus 
                |> Proof.obligation in
            let expansion = apply obligation program in
            let candidate = Proof.Zipper.set_focus expansion site
                |> Proof.Zipper.to_proof in
            return candidate in
        (* compute candidates from the proof *)
        proof
            (* find all expansion sites *)
            |> Proof.Zipper.of_proof
            |> Proof.Zipper.find_all is_site
            (* expand if possible *)
            |> CCList.filter_map expand
end)

let explanation query program =
    let stopping_condition = fun _ -> false in
    let search_structure = search_structure stopping_condition program in
    let candidate = query
        |> Proof.of_conjunct in
    let result = Search.random_walk search_structure candidate in
    let proof, history = CCRandom.run result in
        Explanation.of_proof proof history