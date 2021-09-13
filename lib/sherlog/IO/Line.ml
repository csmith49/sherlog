type t = [
    | `Rule of Watson.Rule.t
    | `Evidence of Evidence.t
    | `Parameter of Parameter.t
]

type line = t

module Compile = struct
    open Watson

    let introduction_context relation terms arguments =
        let relation_symbol = Term.Symbol relation in
        relation_symbol :: (terms @ arguments)

    let introduction_rule
        ~relation:relation
        ~terms:terms
        ~function_id:function_id
        ~arguments:arguments
        ~body:body =
            let target = Term.Variable "_I" in
            let context = introduction_context relation terms arguments in
            (* intro <- body *)
            let intro = Introduction.make target function_id arguments context
                |> Introduction.to_atom in
            let intro_rule = Rule.make intro body in
            (* head <- body, intro *)
            let head = Atom.make relation (terms @ [target]) in
            let concl_rule = Rule.make head (intro :: body) in
            [
                `Rule intro_rule;
                `Rule concl_rule;
            ]

    let discrete_embedding
        ~relation:relation
        ~terms:terms
        ~domain:domain
        ~function_id:function_id
        ~arguments:arguments
        ~body:body =
            let sampling_relation = relation ^ ":sherlog:sample" in
            (* relation:sherlog:sample(terms; f[args]) <- body. *)
            let introduction = introduction_rule
                ~relation:sampling_relation
                ~terms:terms
                ~function_id:function_id
                ~arguments:arguments
                ~body:body in
            (* relation(terms, value) <- relation:sherlog:sample(terms, index). *)
            let embed index value =
                let index_symbol = Term.Float (index |> float_of_int) in
                let head = Atom.make relation (terms @ [value]) in
                let sample = Atom.make sampling_relation (terms @ [index_symbol]) in
                    `Rule (Rule.make head [sample]) in
            introduction @ (CCList.mapi embed domain)
    
    let softmax_embedding
        ~relation:relation
        ~terms:terms
        ~domain:domain
        ~function_id:function_id
        ~arguments:arguments
        ~body:body =
            let softmax_relation = relation ^ ":sherlog:softmax" in
            let sampling_relation = relation ^ ":sherlog:sample" in
            (* relation:sherlog:softmax(terms; f[args]) <- body. *)
            let softmax_introduction = introduction_rule
                ~relation:softmax_relation
                ~terms:terms
                ~function_id:function_id
                ~arguments:arguments
                ~body:body in
            (* relation:sherlog:sample(terms; categorical[W]) <- relation:sherlog:softmax(terms, W). *)
            let weights = Term.Variable ("_W") in
            let sample_introduction = introduction_rule
                ~relation:sampling_relation
                ~terms:terms
                ~function_id:"categorical"
                ~arguments:[weights]
                ~body:(Atom.make softmax_relation (terms @ [weights]) |> CCList.return) in
            (* relation(terms, value) <- relation:sherlog:sample(terms, index). *)
            let embed index value =
                let index_symbol = Term.Float (index |> float_of_int) in
                let head = Atom.make relation (terms @ [value]) in
                let sample = Atom.make sampling_relation (terms @ [index_symbol]) in
                    `Rule (Rule.make head [sample]) in
            softmax_introduction @ sample_introduction @ (CCList.mapi embed domain)
end

module Generation = struct
    type form = [
        | `Introduction of string * Watson.Term.t list
        | `DiscreteEmbedding of Watson.Term.t list * string * Watson.Term.t list
        | `SoftmaxEmbedding of Watson.Term.t list * string * Watson.Term.t list
    ]
    
    (* representation of relation(terms; generation @ context) <- body. *)
    type t = {
        relation : string;
        terms : Watson.Term.t list;
        body : Watson.Atom.t list;
        generation : form;
    }

    let make relation terms generation body = {
        relation = relation;
        terms = terms;
        generation = generation;
        body = body;
    }

    let compile gen = match gen.generation with
        | `Introduction (function_id, arguments) -> Compile.introduction_rule
            ~relation:gen.relation
            ~terms:gen.terms
            ~function_id:function_id
            ~arguments:arguments
            ~body:gen.body
        | `DiscreteEmbedding (domain, function_id, arguments) -> Compile.discrete_embedding
            ~relation:gen.relation
            ~terms:gen.terms
            ~domain:domain
            ~function_id:function_id
            ~arguments:arguments
            ~body:gen.body
        | `SoftmaxEmbedding (domain, function_id, arguments) -> Compile.softmax_embedding
            ~relation:gen.relation
            ~terms:gen.terms
            ~domain:domain
            ~function_id:function_id
            ~arguments:arguments
            ~body:gen.body
end