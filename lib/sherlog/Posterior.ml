type t = Watson.Proof.t CCRandom.t
type score = Watson.Proof.t -> float

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
end

module Parameterization = struct
  type t = float list
end

let dot (params : Parameterization.t) (features : Feature.t list) (proof : Watson.Proof.t) : float =
  let features = features
    |> CCList.map (Feature.apply proof) in
  let sums = CCList.map2 ( +. ) features params in
  CCList.fold_left ( *. ) 1.0 sums

let ( @. ) = dot

let random_proof (score : score) (proofs : Watson.Proof.t list) : t = fun state ->
  let weights = proofs |> CCList.map score in
  let index = CCRandom.run ~st:state (categorical weights) in
  proofs
    |> CCList.get_at_idx index
    |> CCOpt.get_exn_or "Invalid sampling index"