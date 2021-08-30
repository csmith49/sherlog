module Operator = struct
	module Feature = struct
		type t = Watson.Proof.t -> float

		let apply proof feature = feature proof

		(* pre-builts *)
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

	type t = Feature.t list

	let of_contexts ss =
		let default = [
			Feature.intros;
			Feature.constrained_intros;
			Feature.length;
		] in
		let contexts = ss |> CCList.map Feature.context in
			default @ contexts

	let apply op proof = CCList.map (fun f -> f proof) op
end

module Parameterization = struct
	type t = float list

	let linear_combination params features =
		let prods = CCList.map2 ( *. ) params features in
			CCList.fold_left ( +. ) 0.0 prods

	module JSON = struct
		let encode p = `Assoc [
			("type", `String "parameterization");
			("weights", p |> CCList.map JSON.Make.float |> JSON.Make.list);
		]

		let decode json = JSON.Parse.(find (list float) "weights" json)
	end
end

type t = Operator.t * Parameterization.t

let make op p = (op, p)

let operator = fst
let parameterization = snd