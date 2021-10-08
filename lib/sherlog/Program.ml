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

let apply obligation program = let open CCOpt.Infix in
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

(* application takes obligation -> proof *)

let success obligation =
    if Watson.Proof.Obligation.is_empty obligation then
        Some (Proof.Leaf Proof.Success)
    else
        None

let apply_rules rules obligation 

let apply_rules obligation rules =
    (* case 1 - obligation already resolved *)

let apply_rule obligation rule =
    if Watson.Proof.Obligation.is_empty obligation then 

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