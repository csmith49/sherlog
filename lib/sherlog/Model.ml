(* base types *)
type node = string
type view = node * node list * Generation.t
type observation = (node * Watson.Term.t) list

(* graphs all the way down *)
module Graph = Data.Graph.Make(CCString)
type t = {
    graph : (Generation.t, unit) Graph.t;
    observations : observation list;
}

(* access *)
let nodes model = model.graph |> Graph.vertices
let parents node model = model.graph
    |> Graph.incoming node
    |> CCList.map (fun (source, _, _) -> source)
let gen node model = Graph.label node model.graph |> CCOpt.get_exn

let views model = model
    |> nodes
    |> CCList.map (fun node ->
        (node, parents node model, gen node model))

let observations model = model.observations

(* construction *)
let empty = {
    graph = Graph.empty;
    observations = [];
}

let add_view (node, parents, gen) model =
    let graph = model.graph
        |> Graph.add_vertex node gen (* add the node *)
        |> CCList.fold_right (fun v -> fun g ->
                if Graph.mem v g then g else Graph.add_vertex v Generation.Placeholder g
            ) parents (* add the parents, if necessary, with a dummy dist*)
        |> CCList.fold_right (fun p -> fun g -> Graph.add_edge (p, (), node) g) parents (* add edges *)
    in { model with graph = graph; }
let add_observation observation model = { model with observations = observation :: model.observations; }

(* this module helps contain the compilation pipeline from proof to model *)
module Compile = struct
    (* intros are the nodes we pull from the output of resolution - goal is to convert them to views *)
    type intro = {
        value : Watson.Term.t;
        mechanism : string;
        parameters : Watson.Term.t list;
    }

    (* our overall structure comes from the form of proofs given by resolution *)
    type 'a dnf = Or of 'a conjunct list
    and 'a conjunct = And of ('a * intro) list

    (* utilities over dnfs *)

    (* extract tags in the same list of list structure *)
    let tags (dnf : 'a dnf) : 'a list list = match dnf with
        | Or conjuncts -> conjuncts
            |> CCList.map (fun conjunct -> match conjunct with
                | And pairs -> CCList.map fst pairs)

    (* map uniformly *)
    let map (f : ('a * intro) -> ('b * intro)) (dnf : 'a dnf) : 'b dnf = match dnf with
        | Or conjuncts -> conjuncts
            |> CCList.map (fun conjunct -> match conjunct with
                | And pairs -> And (CCList.map f pairs))
            |> fun cs -> Or cs

    (* map indexed by conjuncts *)
    let cmap (fs : (('a * intro) -> ('b * intro)) list) (dnf : 'a dnf) : 'b dnf = match dnf with
        | Or conjuncts -> conjuncts
            |> CCList.map2 (fun f -> fun conjunct -> match conjunct with
                | And pairs -> And (CCList.map f pairs)) fs
            |> fun cs -> Or cs

    (* since this transformation is lightweight, we'll avoid ornaments and the like and just give a type per stage *)
    type initial_dnf = unit dnf
    type named_dnf = node dnf
    type annotated_dnf = (node * Generation.t) dnf
    type connected_dnf = (node * node list * Generation.t) dnf
    
    (* build initial dnf from "proof" *)
    let rec initialize (proof : Watson.Proof.Solution.t list) : initial_dnf =
        let conjuncts = CCList.map conjunct_of_solution proof in Or conjuncts
    and conjunct_of_solution (solution : Watson.Proof.Solution.t) : unit conjunct =
        let intros = solution
            |> Watson.Proof.Solution.introductions
            |> CCList.map (fun (ob, v, p) -> 
                let value = CCList.hd v in
                let mechanism = match ob with
                    | Watson.Obligation.Assign f -> f in
                ( (), { value = value; mechanism = mechanism; parameters = p; } )
            ) in
        And intros

    (* get names for the samples *)

    (* naming is done uniformly and repeatably, if necessary *)
    let name_intro (intro : intro) : node =
        let name = intro.mechanism in
        let args = intro.parameters |> CCList.map Watson.Term.to_string |> CCString.concat ", " in
            name ^ "[" ^ args ^ "]"

    (* and lifting is straightforward *)
    let name_intros (dnf : initial_dnf) : named_dnf =
        let f ( (), intro ) = ( name_intro intro, intro ) in map f dnf

    (* annotating samples *)

    (* samples need to be tagged with the gen, but that requires renaming across conjuncts *)
    (* indexing by conjuncts keeps each execution independent - not sure if necessary, but scared of sharing *)
    (* TODO: explore the above *)

    (* step 1: get per-conjunct sub *)
    let per_conjunct_renaming (dnf : named_dnf) : Watson.Map.t list =
        let f (name, intro) = match intro.value with
            | Watson.Term.Value (Watson.Term.Variable target) -> 
                (Some (target, Watson.Term.Value (Watson.Term.Variable name)), intro)
            | _ -> (None, intro) in
        let renamings = dnf |> map f |> tags in
        renamings |> CCList.map (fun assocs -> assocs
                |> CCList.keep_some
                |> Watson.Map.of_list
            )

    (* step 2: build distribution in context of sub *)
    let build_generations (dnf : named_dnf) (maps : Watson.Map.t list) : annotated_dnf =
        let gs = maps |> CCList.map (fun map -> fun (name, intro) ->
            let args = CCList.map (Watson.Map.apply map) intro.parameters in
            let gen = Generation.Gen (intro.mechanism, args) in
            let tag = (name, gen) in (tag, intro)) in
        cmap gs dnf

    (* all together *)
    let annotate_intros (dnf : named_dnf) : annotated_dnf =
        let subs = per_conjunct_renaming dnf in build_generations dnf subs

    (* connecting samples *)

    (* simple introspection gives us the parents of each node - we just look for variables in the dists *)
    let connect_intros (dnf : annotated_dnf) : connected_dnf =
        let f ( (name, dist), sample ) =
            let parents = Generation.variables dist in
            let tag = (name, parents, dist) in
                (tag, sample)
        in map f dnf

    (* final utility *)
    let views (dnf : connected_dnf) : view list = dnf
        |> tags
        |> CCList.flatten

    let observations (dnf : connected_dnf) : observation list =
        let f ( (name, _, _), intro ) = match intro.value with
            | Watson.Term.Value Watson.Term.Variable _ -> (None, intro)
            | (_ as term) -> (Some (name, term), intro) in
        dnf |> map f
            |> tags
            |> CCList.map CCList.keep_some
end

(* using the above, we can build a model from a "proof" *)
let of_proof proof =
    let dnf = proof
        |> Watson.Proof.Solution.of_proof
        |> Compile.initialize
        |> Compile.name_intros
        |> Compile.annotate_intros
        |> Compile.connect_intros in
    let views = Compile.views dnf in
    let observations = Compile.observations dnf in
    empty
        |> CCList.fold_right add_view views
        |> CCList.fold_right add_observation observations

(* and writing out *)
let rec to_json model =
    let observations = model
        |> observations
        |> CCList.map observation_to_json in
    let views = model
        |> views
        |> CCList.map view_to_json in
    `Assoc [
        ("model", `List views);
        ("observations", `List observations);
    ]
and observation_to_json observation =
    let pair_to_json (node, value) = `Assoc [
        ("variable", `String node);
        ("value", Utility.Term.to_json value)
    ] in `List (CCList.map pair_to_json observation)
and view_to_json (node, parents, gen) = `Assoc [
    ("variable", `String node);
    ("dependencies", `List (CCList.map Interface.JSON.Make.string parents));
    ("generation", Generation.to_json gen);
]