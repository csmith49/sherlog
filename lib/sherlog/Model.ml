module Assignment = struct
    type t = {
        target : string;
        guard : string;
        parameters : Watson.Term.t list;
    }

    let make target guard params = {
        target = target ; guard = guard ; parameters = params ;
    }

    let target a = a.target
    let guard a = a.guard
    let parameters a = a.parameters

    module JSON = struct
        let encode a = `Assoc [
            ("type", `String "assignment");
            ("target", `String (target a));
            ("guard", `String (guard a));
            ("parameters", `List (a |> parameters |> CCList.map Watson.Term.JSON.encode));
        ]

        let decode json = let open CCOpt in
            let* target = JSON.Parse.(find string "target" json) in
            let* guard = JSON.Parse.(find string "guard" json) in
            let* parameters = JSON.Parse.(find (list Watson.Term.JSON.decode) "parameters" json) in
                return (make target guard parameters)
    end

    let pp ppf a = let open Fmt in
        pf ppf "%s <- %s(%a)"
        (target a)
        (guard a)
        (list ~sep:comma Watson.Term.pp) (parameters a)

    let to_string = Fmt.to_to_string pp
end

module Observation = struct
    type t = (string * Watson.Term.t) list

    let rec pp ppf o = let open Fmt in
        pf ppf "[%a]" (list ~sep:comma pp_pair) o
    and pp_pair ppf = function (k, v) -> let open Fmt in
        pf ppf "%s : %a"
            k 
            Watson.Term.pp v

    let to_string = Fmt.to_to_string pp

    module JSON = struct
        let encode o = o
            |> CCList.map (CCPair.map_snd Watson.Term.JSON.encode)
            |> JSON.Make.assoc

        let decode = JSON.Parse.assoc Watson.Term.JSON.decode
    end
end

type t = {
    assignments : Assignment.t list;
    meet : Observation.t;
    avoid : Observation.t list;
    meet_history : Search.History.t;
    avoid_history : Search.History.t list;
}

let make assignments meet avoid = {
    assignments = assignments;
    meet = meet;
    avoid = avoid;
    meet_history = Search.History.empty;
    avoid_history = [];
}

let assignments m = m.assignments
let meet m = m.meet
let avoid m = m.avoid
let meet_history m = m.meet_history
let avoid_history m = m.avoid_history

module JSON = struct
    let encode model = `Assoc [
        ("type", `String "model");
        ("assignments", `List (model |> assignments |> CCList.map Assignment.JSON.encode));
        ("meet",  model |> meet |> Observation.JSON.encode);
        ("meet_history", model |> meet_history |> Search.History.JSON.encode);
        ("avoid", model |> avoid |> CCList.map Observation.JSON.encode |> JSON.Make.list);
        ("avoid_history", model |> avoid_history |> CCList.map Search.History.JSON.encode |> JSON.Make.list);
    ]

    (* TODO - this drops the histories, but that's okay for now *)
    let decode json = let open CCOpt in
        let* assignments = JSON.Parse.(find (list Assignment.JSON.decode) "assignments" json) in
        let* meet = JSON.Parse.(find Observation.JSON.decode "meet" json) in
        let* avoid = JSON.Parse.(find (list Observation.JSON.decode) "avoid" json) in
            return (make assignments meet avoid)
end

module Compile = struct
    type intro = Explanation.Introduction.t

    (* a simple annotated type to mark the compilation process and hold the intermediate results *)
    type 'a taglist = ('a * intro) list

    (* construct a taglist from an explanation - no processing, no tag *)
    let of_explanation (ex : Explanation.t) : unit taglist = ex
        |> Explanation.introductions
        |> CCList.map (CCPair.make ())

    (* name a single introduction *)
    let name_intro (intro : intro) : string = 
        let f = intro
            |> Explanation.Introduction.mechanism in
        let p = intro
            |> Explanation.Introduction.parameters
            |> CCList.map Watson.Term.to_string
            |> CCString.concat ", " in
        let c = intro
            |> Explanation.Introduction.context
            |> CCList.map Watson.Term.to_string
            |> CCString.concat ", " in
        f ^ "[" ^ p ^ " | " ^ c ^ "]"

    (* tag each intro with the computed name *)
    let associate_names (intros : unit taglist) : string taglist =
    let f ((), intro) = (name_intro intro, intro) in
        CCList.map f intros

    (* build map from targets to computed names *)
    let renaming (intros : string taglist) : Watson.Substitution.t =
        let f (name, intro) = match Explanation.Introduction.target intro with
            | Watson.Term.Variable x ->
                let target = Watson.Term.Variable name in
                (Some (x, target), intro)
            | _ -> (None, intro) in
        CCList.map f intros
            |> CCList.map fst
            |> CCList.keep_some
            |> Watson.Substitution.of_list

    (* build assignments from named intros *)
    let rec assignments (intros : string taglist) : Assignment.t taglist =
        let s = renaming intros in
        CCList.map (build_assignment s) intros
    and build_assignment (s : Watson.Substitution.t) : (string * intro) -> (Assignment.t * intro) = function
        (name, intro) ->
            let assignment = {
                Assignment.target = name;
                guard = Explanation.Introduction.mechanism intro;
                parameters = intro
                    |> Explanation.Introduction.parameters
                    |> CCList.map (Watson.Substitution.apply s);
            } in
            (assignment, intro)
    
    (* get observation *)
    let observation (intros : Assignment.t taglist) : Observation.t =
        let f (assignment, intro) = match Explanation.Introduction.target intro with
            | Watson.Term.Variable _ -> (None, intro)
            | (_ as term) ->
                let tag = (Assignment.target assignment, term) in
                (Some tag, intro) in
        CCList.map f intros
            |> CCList.map fst
            |> CCList.keep_some

end

let of_proof proof = 
    let pa = proof
        |> Explanation.of_proof
        |> Compile.of_explanation
        |> Compile.associate_names
        |> Compile.assignments in
    let assignments = pa |> CCList.map fst in
    let meet = pa |> Compile.observation in
    let avoid = [] in
        make assignments meet avoid

let compile proof = proof
    |> Explanation.of_proof
    |> Compile.of_explanation
    |> Compile.associate_names
    |> Compile.assignments

let of_proof_and_contradictions proof contradictions =
    let pa = compile proof in
    let nas = contradictions |> CCList.map compile in
    let assignments = pa :: nas |> CCList.flat_map (CCList.map fst) in
    let meet = pa |> Compile.observation in
    let avoid = nas |> CCList.map Compile.observation in
        make assignments meet avoid

let of_search_states handle bristles =
    let pa = handle
        |> Search.State.value
        |> compile in
    let nas = bristles
        |> CCList.map Search.State.value
        |> CCList.map compile in
    let assignments = pa :: nas |> CCList.flat_map (CCList.map fst) in
    let meet = pa |> Compile.observation in
    let avoid = nas |> CCList.map Compile.observation in
    let meet_history = handle |> Search.State.history in
    let avoid_history = bristles |> CCList.map Search.State.history in
    {
        assignments = assignments;
        meet = meet;
        avoid = avoid;
        meet_history = meet_history;
        avoid_history = avoid_history;
    }

let pp ppf model = let open Fmt in
    pf ppf "{%a@,; +: %a@,; -: %a}"
    (list ~sep:comma Assignment.pp) (model |> assignments)
    Observation.pp (model |> meet)
    (list ~sep:comma Observation.pp) (model |> avoid)

let to_string = Fmt.to_to_string pp