type 'a t = ('a * Watson.Proof.t) CCRandom.t

let sample posterior = CCRandom.run posterior

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

module Feature = struct
  type t = Watson.Proof.t -> float

  let apply proof feature = feature proof

  (* pre-builts *)
  let intros proof = proof
    |> Explanation.of_proof
    |> Explanation.introductions
    |> CCList.length
    |> CCFloat.of_int

  let constrained_intros proof = proof
    |> Explanation.of_proof
    |> Explanation.introductions
    |> CCList.filter Explanation.Introduction.is_constrained
    |> CCList.length
    |> CCFloat.of_int

  let length proof = proof
    |> Watson.Proof.to_atoms
    |> CCList.length
    |> CCFloat.of_int

  let context str proof = proof
    |> Explanation.of_proof
    |> Explanation.introductions
    |> CCList.flat_map Explanation.Introduction.context
    |> CCList.filter (fun t -> Watson.Term.equal t (Watson.Term.Symbol str))
    |> CCList.length
    |> CCFloat.of_int

  let context_operator ss =
    let default = [
      intros;
      constrained_intros;
      length;
    ] in
    let contexts = ss |> CCList.map context in
      default @ contexts
end

module Parameterization = struct
  type t = float list

  module JSON = struct
    let encode p = `Assoc [
      ("type", `String "parameterization");
      ("weights", p |> CCList.map JSON.Make.float |> JSON.Make.list);
    ]
    
    let decode json = JSON.Parse.(find (list float) "weights" json)
  end
end

module Score = struct
  type t = {
    parameters : Parameterization.t;
    operator : Feature.t list;
  }

  let dot params features = {
    parameters = params;
    operator = features;
  }

  let ( @. ) params features = dot params features

  let of_assoc fs =
    let params = CCList.map fst fs in
    let features = CCList.map snd fs in
      params @. features

  module Record = struct
    type t = float list
      
    module JSON = struct
      let encode p = `Assoc [
        ("type", `String "record");
        ("values", p |> CCList.map JSON.Make.float |> JSON.Make.list);
      ]
          
      let decode json = JSON.Parse.(find (list float) "weights" json)
    end
  end

  let project score proof = score.operator
    |> CCList.map (Feature.apply proof)

  let apply score proof =
    let features = project score proof in
    let prods = CCList.map2 ( *. ) features score.parameters in
      CCList.fold_left ( +. ) 0.0 prods

  let apply_and_project score proof =
    let features = project score proof in
    let prods = CCList.map2 ( *. ) features score.parameters in
    let score = CCList.fold_left ( +. ) 0.0 prods in
      (score, features)
end

let random_proof score proofs = fun state ->
  let scores = proofs |> CCList.map (Score.apply score) in
  let index = CCRandom.run ~st:state (categorical scores) in
  let proof = proofs
    |> CCList.get_at_idx index
    |> CCOpt.get_exn_or "Invalid sampling index" in
  ((), proof)

let random_proof_and_records score proofs = fun state ->
  let scores_and_records = proofs |> CCList.map (Score.apply_and_project score) in
  let scores = scores_and_records |> CCList.map fst in
  let records = scores_and_records |> CCList.map snd in
  let index = CCRandom.run ~st:state (categorical scores) in
  let proof = proofs
    |> CCList.get_at_idx index
    |> CCOpt.get_exn_or "Invalid sampling index" in
  let record = records
    |> CCList.get_at_idx index
    |> CCOpt.get_exn_or "Invalid sampling index" in
  ((record, records), proof)