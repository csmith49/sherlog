module Assignment = struct
    type t = {
        target : string;
        semantics : Watson.Guard.t;
        parameters : Watson.Term.t list;
    }

    let to_json a = `Assoc [
        ("target", `String a.target);
        ("semantics", Utility.Guard.to_json a.semantics);
        ("parameters", `List (CCList.map Utility.Term.to_json a.parameters));
    ]
end

module Observation = struct
    type t = (string * Watson.Term.t) list

    let of_list xs = xs

    let to_json o = `Assoc (CCList.map (CCPair.map_snd Utility.Term.to_json) o) 
end

type t = (Assignment.t list * Observation.t) list

module Compile = struct
    type introduction = {
        target : Watson.Term.t;
        semantics : Watson.Guard.t;
        parameters : Watson.Term.t list;
        context : Watson.Term.t list;
    }
    type 'a conjunct = And of ('a * introduction) list

    (* minor utilities *)
    let tags (conjunct : 'a conjunct) : 'a list = match conjunct with
        | And xs -> CCList.map fst xs

    let map (f : ('a * introduction) -> ('b * introduction)) (conjunct : 'a conjunct) : 'b conjunct = match conjunct with
        | And xs -> And (CCList.map f xs)

    let initialize (solution : Watson.Proof.Solution.t) : unit conjunct =
        let intros = solution
            |> Watson.Proof.Solution.introductions
            |> CCList.map (fun (g, p, c, v) ->
                let intro = {
                    target = v;
                    semantics = g;
                    parameters = p;
                    context = c;
                } in
                ( (), intro )
            ) in
        And intros

    let name_introduction (intro : introduction) : string =
        let name = intro.semantics in
        let args = intro.parameters
            |> CCList.map Watson.Term.to_string
            |> CCString.concat ", " in
        let cons = intro.context
            |> CCList.map Watson.Term.to_string
            |> CCString.concat ", " in
        name ^ "[" ^ args ^ " | " ^ cons ^ "]"    

    let name (conjunct : unit conjunct) : string conjunct =
        let f ((), intro) = (name_introduction intro, intro) in map f conjunct

    let renaming (conjunct : string conjunct) : Watson.Map.t =
        let f (name, intro) = match intro.target with
            | Watson.Term.Value (Watson.Term.Variable target) ->
                (Some (target, Watson.Term.Value (Watson.Term.Variable name)), intro)
            | _ -> (None, intro) in
        conjunct
            |> map f
            |> tags
            |> CCList.keep_some
            |> Watson.Map.of_list

    let construct_assignments (conjunct : string conjunct) (renaming : Watson.Map.t) : Assignment.t conjunct =
        let f (name, intro) =
            let assignment = {
                Assignment.target = name;
                semantics = intro.semantics;
                parameters = CCList.map (Watson.Map.apply renaming) intro.parameters;
            } in (assignment, intro)
        in map f conjunct

    let annotate (conjunct : string conjunct) : Assignment.t conjunct =
        construct_assignments conjunct (renaming conjunct)

    let assignments (conjunct : Assignment.t conjunct) : Assignment.t list = tags conjunct

    let observation (conjunct : Assignment.t conjunct) : Observation.t =
        let f (assignment, intro) = match intro.target with
            | Watson.Term.Value (Watson.Term.Variable _) -> (None, intro)
            | (_ as term) -> let tag = (assignment.Assignment.target, term) in
                (Some tag, intro) in
        conjunct
            |> map f
            |> tags
            |> CCList.keep_some
            |> Observation.of_list
end

let of_proof proof =
    let stories = proof
        |> Watson.Proof.Solution.of_proof
        |> CCList.map Compile.initialize
        |> CCList.map Compile.name 
        |> CCList.map Compile.annotate in
    stories
        |> CCList.map (fun s ->
                let assignments = Compile.assignments s in
                let observation = Compile.observation s in
                    (assignments, observation)
        )

(* and writing out *)
let to_json model =
    let f (assignments, observation) = `Assoc [
            ("story", `List (CCList.map Assignment.to_json assignments));
            ("observation", Observation.to_json observation);
        ] in
    `List (CCList.map f model)
