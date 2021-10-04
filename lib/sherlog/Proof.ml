open Watson.Proof

type proof =
    | Leaf of leaf
    | Interior of edge list
and edge = Edge of Witness.t * proof
and leaf =
    | Frontier of Obligation.t
    | Success
    | Failure

let of_conjunct atoms =
    let obligation = Obligation.of_conjunct atoms in
        Leaf (Frontier obligation)

let rec introductions proof = proof |> introductions_aux |> CCOpt.get_or ~default:[]
and introductions_aux = function
    | Leaf (Success) -> Some []
    | Interior edges -> let open CCOpt.Infix in
        let intro_edge = function Edge (w, proof) ->
            let* intros = introductions_aux proof in
            let intro = w
                |> Witness.resolved_atom
                |> Introduction.of_atom
                |> CCOpt.to_list in
            Some (intro @ intros) in
        edges
            |> CCList.map intro_edge
            |> CCList.fold_left (<+>) None
    | _ -> None

let rec branches proof = proof |> branches_aux
and branches_aux = function
    | Leaf (Success) -> [[]]
    | Leaf _ -> []
    | Interior edges ->
        let branch_edge = function Edge (w, proof) -> 
            let intro = w
                |> Witness.resolved_atom
                |> Introduction.of_atom
                |> CCOpt.to_list in
            proof
                |> branches
                |> CCList.map (fun b -> intro @ b) in
        CCList.flat_map branch_edge edges

module Search = struct
    module Embedding = struct
        type t = float list

        module JSON = struct
            let encode f = f |> CCList.map JSON.Make.float |> JSON.Make.list
            let decode = JSON.Parse.(list float)
        end
    end

    module Choice = struct
        type t = {
            embedding : Embedding.t;
            context : Embedding.t list;
        }

        module JSON = struct
            let encode choice = `Assoc [
                ("type", `String "choice");
                ("embedding", choice.embedding |> Embedding.JSON.encode);
                ("context", choice.context |> CCList.map Embedding.JSON.encode |> JSON.Make.list);
            ]

            let decode json = let open CCOpt in
                let* embedding = JSON.Parse.(find Embedding.JSON.decode "embedding" json) in
                let* context = JSON.Parse.(find (list Embedding.JSON.decode) "context" json) in
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
                ("choices", history |> CCList.map Choice.JSON.encode |> JSON.Make.list);
            ]

            let decode json = let open CCOpt in
                let* choices = JSON.Parse.(find (list Choice.JSON.decode) "choices" json) in
                return choices
        end
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
end

module Zipper = struct
    type t = View of proof * context list
    and context = Context of Witness.t * edge list * edge list

    (* getters and setters *)
    let focus = function
        | View (proof, _) -> proof
    
    let set_focus proof = function
        | View (_, contexts) -> View (proof, contexts)

    (* basic movement *)
    let left = function
        | View (proof, Context (w, Edge (w', proof') :: ls, rs) :: rest) ->
            let edge = Edge (w, proof) in
            let context = Context (w', ls, edge :: rs) in
            Some (View (proof', context :: rest))
        | _ -> None

    let right = function
        | View (proof, Context (w, ls, Edge (w', proof') :: rs) :: rest) ->
            let edge = Edge (w, proof) in
            let context = Context (w', edge :: ls, rs) in
            Some (View (proof', context :: rest))
        | _ -> None

    let up = function
        | View (proof, Context (w, ls, rs) :: rest) ->
            let edge = Edge (w, proof) in
            let edges = (CCList.rev ls) @ (edge :: rs) in
            Some (View (Interior edges, rest))
        | _ -> None

    let down = function
        | View (Interior (Edge (w, proof) :: edges), contexts) ->
            let context = Context (w, [], edges) in
            Some (View (proof, context :: contexts))
        | _ -> None

    (* advanced movement *)
    let rec next z = let open CCOpt.Infix in
        (right z) <+> ((up z) >>= next)

    let preorder z = let open CCOpt.Infix in
        (down z) <+> (next z)

    let rec find pred z = let open CCOpt.Infix in
        let check = CCOpt.if_ (fun f -> pred (focus f)) z in
        check <+> ((preorder z) >>= (find pred))

    (* construction / conversion *)
    let of_proof proof = View (proof, [])

    let rec to_proof zipper = match up zipper with
        | Some zipper -> to_proof zipper
        | None -> focus zipper
end