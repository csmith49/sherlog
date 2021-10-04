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
        ("type", `String "program");
        ("rules", prog |> Functional.rules |> CCList.map Watson.Rule.JSON.encode |> JSON.Make.list);
        ("parameters", prog |> Functional.parameters |> CCList.map Parameter.JSON.encode |> JSON.Make.list);
        ("posterior", prog |> Functional.posterior |> Posterior.JSON.encode);
    ]

    let decode json = let open CCOpt in
        let* rules = JSON.Parse.(find (list Watson.Rule.JSON.decode) "rules" json) in
        let* params = JSON.Parse.(find (list Parameter.JSON.decode) "parameters" json) in
        let* posterior = JSON.Parse.(find Posterior.JSON.decode "posterior" json) in
            return (Functional.make rules params posterior)
end

(* APPLICATION *)

module Application = struct
    let apply node program =
        let obligation = node |> Proof.Node.obligation in
        (* check if we're done *)
        if Watson.Proof.Obligation.is_empty obligation then
            [ Proof.success obligation ]
        (* otherwise get all possible expansions *)
        else let expansions = program
            |> Functional.rules
            |> CCList.map (Watson.Proof.resolve obligation)
            |> CCList.keep_some in
        match expansions with
            | [] -> [ Proof.failure obligation ]
            | xs -> xs |> CCList.map (fun (w, ob) -> Proof.of_witness w ob)
end

(* SEARCH *)

let domain : t -> (module Search.Domain with type t = Proof.Node.t) = fun program -> (module struct
    type t = Proof.Node.t

    let features = fun proof -> Posterior.featurize proof program.posterior
    let score = fun fs -> Posterior.score fs program.posterior
    let expand = fun node -> 
        let _ = node
            |> Proof.Node.tag
            |> Proof.Tag.to_string
            |> print_endline in
        node |> Proof.Node.tag |> Proof.Tag.interior
    let expansions = fun node -> Application.apply node program
end)

let explanation ?width:(width=CCInt.max_int) program conjunct =
    let initial = Proof.of_conjunct conjunct in
    let final = Search.beam (domain program) width initial in
    Explanation.of_proof final []