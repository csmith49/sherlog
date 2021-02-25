type t = [
    | `Rule of Watson.Rule.t
    | `Evidence of Evidence.t
    | `Parameter of Parameter.t
	| `Dependency of Watson.Rule.t
	| `Constraint of Watson.Atom.t list
]

let encode_intro
    ~relation:relation
    ~arguments:arguments
    ~guard:f
    ~parameters:parameters
    ~context:context
    ~body:body =
        let target = Watson.Term.Variable "_I" in
        let intro = Explanation.Introduction.make f parameters context target
            |> Explanation.Introduction.to_atom in
        (* intro <- body *)
        let intro_rule = Watson.Rule.make intro body in
        (* head <- body, intro *)
        let head = Watson.Atom.make relation (arguments @ [target]) in
        let conclusion_rule = Watson.Rule.make head (intro :: body) in
            [ `Rule intro_rule ; `Rule conclusion_rule ]

let encode_fuzzy ~head:head ~body:body ~weight:weight =
    let relation = Watson.Atom.relation head in
    let subterms = (head :: body)
        |> CCList.flat_map Watson.Atom.terms in
    let context = (Watson.Term.Symbol ("fuzzy:" ^ relation)) :: subterms in
    (* intro <- body *)
    let target = Watson.Term.Variable "_I" in
    let intro = Explanation.Introduction.make "bernoulli" [weight] context target
        |> Explanation.Introduction.to_atom in
    let intro_rule = Watson.Rule.make intro body in
    (* head <- body, intro *)
    let target = Watson.Term.Float 1.0 in
    let intro = Explanation.Introduction.make "bernoulli" [weight] context target
        |> Explanation.Introduction.to_atom in
    let conclusion_rule = Watson.Rule.make head (intro :: body) in
        [ `Rule intro_rule ; `Rule conclusion_rule ]

