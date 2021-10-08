module Embedding = struct
    type t = float list

    module JSON = struct
        let encode f = f |> JSON.Encode.(list float)
        let decode = JSON.Parse.(list float)
    end
end

module Choice = struct
    type t = {
        embedding : Embedding.t;
        context : Embedding.t list;
    }

    module Functional = struct
        let embedding choice = choice.embedding
        let context choice = choice.context
    end

    module JSON = struct
        let encode choice = `Assoc [
            ("type", `String "choice");
            ("embedding", choice.embedding |> Embedding.JSON.encode);
            ("context", choice.context |> JSON.Encode.list Embedding.JSON.encode);
        ]

        let decode json = let open CCOpt in
            let* embedding = JSON.Parse.(find "embedding" Embedding.JSON.decode json) in
            let* context = JSON.Parse.(find "context" (list Embedding.JSON.decode) json) in
            return {
                embedding = embedding;
                context = context;
            }
    end

    module Random = struct
        let gumbel_trick w = let open CCRandom in
            let scale u = u |> log |> log |> fun x -> 0.0 -. x in
            (float_range 0.0 1.0) >|= scale >|= CCFloat.add (log w)

        let categorical weights = fun state ->
            let scores = weights
                |> CCList.map gumbel_trick
                |> CCList.map (CCRandom.run ~st:state)
                |> CCList.mapi CCPair.make in
            let sort_key (li, ls) (ri, rs) = if ls >= rs then (li, ls) else (ri, rs) in
            let argmax = scores |> CCList.fold_left sort_key (0, -1.0) in
                fst argmax

        let prop_to values scores = fun state ->
            let index = CCRandom.run ~st:state (categorical scores) in
            values |> CCList.get_at_idx index |> CCOpt.get_exn_or "Invalid sampling."
    end

    let choose (values : 'a list) (embed : 'a -> Embedding.t) (score : Embedding.t -> float): ('a * t) CCRandom.t = fun state ->
        let embeddings = CCList.map embed values in
        let scores = CCList.map score embeddings in
        let value, embedding = Random.prop_to (CCList.map2 CCPair.make values embeddings) scores
            |> CCRandom.run ~st:state in
        value, {
            embedding=embedding;
            context=embeddings;
        }
end

module History = struct
    type t = Choice.t list

    module JSON = struct
        let encode history = `Assoc [
            ("type", `String "history");
            ("choices", history |> JSON.Encode.list Choice.JSON.encode);
        ]

        let decode json = let open CCOpt in
            let* choices = JSON.Parse.(find "choices" (list Choice.JSON.decode) json) in
            return choices
    end
end

module type Domain = sig
    val select : Watson.Proof.Obligation.t -> bool
    val expand : Watson.Proof.Obligation.t -> proof
    val score : Watson.Proof.Obligation.t -> Embedding.t
end

let obligation = function
    | Leaf (Frontier obligation) -> Some obligation
    | _ -> None

let expandable proof = proof |> obligation |> CCOpt.is_some

let expand rules obligation =
    (* check if we're already done *)
    if Obligation.is_empty obligation then (Leaf Success) else
    (* get the resolutions from rules *)
    match CCList.filter_map (resolve obligation) rules with
        (* if there aren't any, the expansion fails *)
        | [] -> (Leaf Failure)
        (* otherwise, build the fresh edges and collapes the tree *)
        | rs ->
            let mk_edge (witness, obligation) =
                Edge (witness, Leaf (Frontier obligation)) in
            let edges = CCList.map mk_edge rs in
            Interior edges