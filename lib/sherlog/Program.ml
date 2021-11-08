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

let space : t -> (module Search.Algorithms.Space with type state = Watson.Proof.Obligation.t and type witness = Watson.Proof.Witness.t) = fun program -> (module struct
    (* nodes are obligations, with edges annotated by witnesses *)
    type state = Watson.Proof.Obligation.t
    type witness = Watson.Proof.Witness.t

    (* goal is to find empty obligations *)
    let is_goal state = Watson.Proof.Obligation.is_empty state
    let next state = program
            |> Functional.rules
            |> CCList.filter_map (Watson.Proof.resolve state)
    let embed = Posterior.apply (Functional.posterior program)
end)