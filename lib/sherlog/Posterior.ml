module Feature = struct
	type t = Watson.Proof.t -> float

	let apply proof feature = feature proof

	(* builtins *)
	let intros proof = proof
		|> Watson.Proof.to_atoms
		|> CCList.filter_map Introduction.of_atom
		|> CCList.length
		|> CCFloat.of_int

	let constrained_intros proof = proof
		|> Watson.Proof.to_atoms
		|> CCList.filter_map Introduction.of_atom
		|> CCList.filter Introduction.is_constrained
		|> CCList.length
		|> CCFloat.of_int

	let length proof = proof
		|> Watson.Proof.to_atoms
		|> CCList.length
		|> CCFloat.of_int

	let context str proof = proof
		|> Watson.Proof.to_atoms
		|> CCList.filter_map Introduction.of_atom
		|> CCList.flat_map Introduction.context
		|> CCList.filter (fun t -> Watson.Term.equal t (Watson.Term.Symbol str))
		|> CCList.length
		|> CCFloat.of_int
end

module Embedding = struct
	type t = float list

	let to_json embedding = `Assoc [
		("type", `String "embedding");
		("value", `List (embedding |> CCList.map JSON.Make.float));
	]

	let of_json json = JSON.Parse.(find (list float) "value" json)
end

module Operator = struct
	type t = {
		defaults : bool;
		context_clues : string list;
	}

	let default_features operator = match operator.defaults with
		| true -> [Feature.intros; Feature.constrained_intros; Feature.length]
		| false -> []

	let context_features operator = operator.context_clues
		|> CCList.map Feature.context

	let apply operator proof =
		let features = (default_features operator) @ (context_features operator) in
		CCList.map (fun f -> f proof) features

	let default = {
		defaults = true;
		context_clues = [];
	}

	let of_context_clues clues = {
		defaults = false;
		context_clues = clues;
	}

	let join left right = {
		defaults = left.defaults && right.defaults;
		context_clues = left.context_clues @ right.context_clues;
	}

	let to_json operator = `Assoc [
		("type", `String "operator");
		("defaults", `Bool operator.defaults);
		("context-clues", `List (operator.context_clues |> CCList.map JSON.Make.string));
	]

	let of_json json = let open CCOpt in
		let* defaults = JSON.Parse.(find bool "defaults" json) in
		let* context_clues = JSON.Parse.(find (list string) "context-clues" json) in
		return {
			defaults = defaults;
			context_clues = context_clues;
		}
end

module Ensemble = struct
	type t =
		| Linear of float list

	let apply ensemble embedding = match ensemble with
		| Linear [] -> 1.0
		| Linear weights -> let prods = CCList.map2 ( *. ) weights embedding in
			CCList.fold_left ( +. ) 0.0 prods

	let to_json = function
		| Linear weights -> `Assoc [
			("type", `String "ensemble");
			("kind", `String "linear");
			("weights", `List (weights |> CCList.map JSON.Make.float));
		]

	let of_json json = let open CCOpt in
		let* kind = JSON.Parse.(find string "kind" json) in
		match kind with
			| "linear" ->
				let* weights = JSON.Parse.(find (list float) "weights" json) in
				return (Linear weights)
			| _ -> None
end

type t = {
	operator : Operator.t;
	ensemble : Ensemble.t;
}

let embed_and_score posterior proof =
	let embedding = Operator.apply posterior.operator proof in
	let score = Ensemble.apply posterior.ensemble embedding in
		(embedding, score)

let apply posterior proof = proof
	|> embed_and_score posterior
	|> snd

let make operator ensemble = {
	operator = operator;
	ensemble = ensemble;
}

let uniform = {
	operator = Operator.of_context_clues [];
	ensemble = Linear [];
}
let default = {
	operator = Operator.default;
	ensemble = Ensemble.Linear [1.0; 1.0; 1.0];
}

let operator posterior = posterior.operator
let ensemble posterior = posterior.ensemble

let to_json posterior = `Assoc [
	("type", `String "posterior");
	("operator", posterior |> operator |> Operator.to_json);
	("ensemble", posterior |> ensemble |> Ensemble.to_json);
]

let of_json json = let open CCOpt in
	let* operator = JSON.Parse.(find Operator.of_json "operator" json) in
	let* ensemble = JSON.Parse.(find Ensemble.of_json "ensemble" json) in
		return (make operator ensemble)