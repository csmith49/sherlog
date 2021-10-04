module Featurization = struct
    type t = float list

    module JSON = struct
        let encode f = f |> CCList.map JSON.Make.float |> JSON.Make.list
        let decode = JSON.Parse.(list float)
    end
end

type 'a tree = 'a Data.Tree.tree

module type Domain = sig
    type t

    val features : t tree -> Featurization.t
    val score : Featurization.t -> float
    val expand : t -> bool
    val expansions : t -> (t tree) list
end

module Choice = struct
    type t = {
        featurization : Featurization.t;
        context : Featurization.t list;
    }

    let make f ctx = {
        featurization = f;
        context = ctx;
    }

    module JSON = struct
        let encode choice = `Assoc [
                ("type", `String "choice");
                ("featurization", choice.featurization |> Featurization.JSON.encode);
                ("context", choice.context |> CCList.map Featurization.JSON.encode |> JSON.Make.list)
            ]

        let decode json = let open CCOpt in
            let* featurization = JSON.Parse.(find Featurization.JSON.decode "featurization" json) in
            let* context = JSON.Parse.(find (list Featurization.JSON.decode) "context" json) in
                return (make featurization context)
    end
end

module History = struct
    type t = Choice.t list

    module JSON = struct
        let encode history = `Assoc [
            ("type", `String "history");
            ("choices", history |> CCList.map Choice.JSON.encode |> JSON.Make.list);
        ]

        let decode json = let open CCOpt in
            let* choices = JSON.Parse.(find (list Choice.JSON.decode) "choices" json) in
            return choices
    end
end

module Random = struct
    let gumbel_trick weight = let open CCRandom in
        let scale u = u |> log |> log |> fun x -> 0.0 -. x in
        (float_range 0.0 1.0) >|= scale >|= CCFloat.add (log weight)

    let categorical weights : int CCRandom.t = fun state ->
        let scores = weights
            |> CCList.map gumbel_trick
            |> CCList.map (CCRandom.run ~st:state)
            |> CCList.mapi CCPair.make in
        let sort_key (li, ls) (ri, rs) = if ls >= rs then (li, ls) else (ri, rs) in
        let argmax = scores |> CCList.fold_left sort_key (0, -1.0) in
            fst argmax

    let prop_to (values : 'a list) (score : 'a -> float) : 'a CCRandom.t = fun state ->
        let scores = CCList.map score values in
        let index = CCRandom.run ~st:state (categorical scores) in
            values |> CCList.get_at_idx index |> CCOpt.get_exn_or "Invalid sampling index"

    let choice : type a . (module Domain with type t = a) -> a tree list -> (a tree * Choice.t) CCRandom.t = fun (module D) values ->
        let featurizations = values
            |> CCList.map (fun v -> (v, D.features v)) in
        let context = featurizations |> CCList.map snd in
        let choices = featurizations
            |> CCList.map (fun (v, fs) -> (v, Choice.make fs context)) in
        let score (_, choice) = choice.Choice.featurization |> D.score in
        prop_to choices score
end

let rec beam : type a . (module Domain with type t = a) -> int -> a tree -> (a tree * History.t) = fun (module D) width tree ->
    (* get an expansion site *)
    let site = tree
        |> Data.Tree.zipper
        |> Data.Tree.find (fun t -> t |> Data.Tree.label |> D.expand) in
    match site with
        | None -> tree, []
        | Some site ->
            let node = site |> Data.Tree.focus |> Data.Tree.label in
            let expansions = D.expansions node in
            (* gumbel trick *)
            let subtrees, choices =
                if CCList.length expansions <= width then expansions, []
                else
                    let samples = Random.choice (module D) expansions
                        |> CCList.replicate width
                        |> CCRandom.list_seq in
                    CCRandom.run samples |> CCList.split in
            (* rebuild tree *)
            let tree = Data.Tree.Node (node, subtrees) in
            (* recurse *)
            let result, history = beam (module D) width tree in
            (result, choices @ history)