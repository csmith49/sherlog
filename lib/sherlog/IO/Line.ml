type t = [
    | `Rule of Watson.Rule.t
    | `Evidence of Evidence.t
    | `Parameter of Parameter.t
]

module Encode = struct
    let introduction_rule
        ~relation:relation
        ~terms:terms
        ~function_id:function_id
        ~arguments:arguments
        ~context:context
        ~body:body =
            let target = Watson.Term.Variable "_I" in
            let intro = Introduction.make target function_id arguments context
                |> Introduction.to_atom in
            (* intro <- body *)
            let intro_rule = Watson.Rule.make intro body in
            (* head <- body, intro *)
            let head = Watson.Atom.make relation (terms @ [target]) in
            let conclusion_rule = Watson.Rule.make head (intro :: body) in
                [ `Rule intro_rule ; `Rule conclusion_rule ]

    let fuzzy_rule ~head:head ~body:body ~weight:weight =
        let relation = Watson.Atom.relation head in
        let subterms = (head :: body)
            |> CCList.flat_map Watson.Atom.terms in
        let context = (Watson.Term.Symbol ("fuzzy:" ^ relation)) :: subterms in
        (* intro <- body *)
        let target = Watson.Term.Variable "_I" in
        let intro = Introduction.make target "bernoulli" [weight] context
            |> Introduction.to_atom in
        let intro_rule = Watson.Rule.make intro body in
        (* head <- body, intro *)
        let target = Watson.Term.Float 1.0 in
        let intro = Introduction.make target "bernoulli" [weight] context
            |> Introduction.to_atom in
        let conclusion_rule = Watson.Rule.make head (intro :: body) in
            [ `Rule intro_rule ; `Rule conclusion_rule ]
end