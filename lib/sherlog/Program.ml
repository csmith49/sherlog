open Watson

(* BASICS *)

type t = {
	rules : Rule.t list;
	parameters : Parameter.t list;
	evidence : Evidence.t list;
	ontology : Ontology.t;
}

let is_intro_rule rule = rule
	|> Rule.head
	|> Atom.relation
	|> CCString.equal Explanation.Introduction.Key.introduction

let rules program = program.rules
let introduction_rules program = program
	|> rules
	|> CCList.filter is_intro_rule
let non_introduction_rules program = program
	|> rules
	|> CCList.filter (fun r -> not (is_intro_rule r))

let parameters program = program.parameters
let evidence program = program.evidence
let ontology program = program.ontology

let make rules parameters evidence ontology = {
	rules = rules;
	parameters = parameters;
	evidence = evidence;
	ontology = ontology;
}

(* SEMANTICS *)

module Semantics = struct
	open Watson
	
	type t = Proof.t -> Proof.t list

	let rec one rules proof = match rules with
		| [] -> []
		| rule :: rest -> begin match Proof.resolve proof rule with
			| Some result -> [result]
			| None -> one rest proof
		end
	
	let all rules proof = rules |> CCList.filter_map (Proof.resolve proof)

	let rec fp sem proof = match sem proof with
		| [] -> [proof]
		| results -> results
			|> CCList.flat_map (fp sem)

	let seq fst snd proof = proof
		|> fst
		|> CCList.flat_map snd

	let xor fst snd proof = match fst proof with
		| [] -> snd proof
		| results -> results

	module Infix = struct
		let ( <+> ) = xor
		let ( >> ) = seq
	end
end

module Filter = struct
	type t = Watson.Proof.t -> bool

	let negate filter = fun x -> not (filter x)

	let constraint_avoiding ontology proof =
		let constraints = ontology |> Ontology.constraints in
		let atoms = proof |> Watson.Proof.to_atoms in
		constraints
			|> CCList.map (fun c -> Watson.Atom.embed_all c atoms)
			|> CCList.for_all CCList.is_empty

	let intro_consistent proof =
		let intro_tags = proof
			|> Explanation.of_proof
			|> Explanation.introductions
			|> CCList.map Explanation.Introduction.tag in
		let num_intros = CCList.length intro_tags in
		let num_unique_intros = intro_tags
			|> CCList.sort_uniq ~cmp:(CCList.compare Watson.Term.compare)
			|> CCList.length in
		CCInt.equal num_intros num_unique_intros

	let length k proof = 
		let l = proof
			|> Watson.Proof.witnesses
			|> CCList.length in
		l <= k

	let join l r = fun proof -> (l proof) && (r proof)

	let ( ++ ) = join
end

(* APPLICATION *)

let apply program =
	let intro_rules = introduction_rules program in
	let non_intro_rules = non_introduction_rules program in
	let open Semantics in let open Semantics.Infix in
		(fp (one non_intro_rules)) >> (all intro_rules)

let apply_with_dependencies program =
	let intro_rules = introduction_rules program in
	let non_intro_rules = non_introduction_rules program in
	let ontological_rules = program |> ontology |> Ontology.dependencies in
	let open Semantics in let open Semantics.Infix in
		(fp (one non_intro_rules)) >> (all intro_rules <+> all ontological_rules)

(* SEARCH *)

let linear_domain : t -> (t -> Watson.Proof.t -> Watson.Proof.t list) -> Posterior.t -> (module Search.Domain with type t = Watson.Proof.t) = fun program successor posterior ->
	(module struct
		type t = Watson.Proof.t

		let features = Posterior.Operator.apply (Posterior.operator posterior)
		let score = Posterior.Parameterization.linear_combination (Posterior.parameterization posterior)
		let accept = Watson.Proof.is_resolved
		let reject = fun _ -> false
		let successors = successor program
	end)

let initial_worklist goal cache =
	let history = Search.History.empty in
	let proof = Watson.Proof.of_atoms goal cache in
		[Search.State.make proof history]

(* MODEL BUILDING *)
let prove ?width:(width=CCInt.max_int) program posterior goal =
	(* get search domain *)
	let (module D) = linear_domain program apply_with_dependencies posterior in
	(* initialize worklist *)
	let wl = initial_worklist goal [] in
	(* search *)
	let results = Search.stochastic_beam_search (module D) width wl [] in
	(* and remove anything with a contradiction *)
	let constraint_avoiding = program
		|> ontology
		|> Filter.constraint_avoiding in
	results |> CCList.filter (Search.State.check constraint_avoiding)

let contradict ?width:(width=CCInt.max_int) program posterior proof =
	(* get search domain *)
	let (module D) = linear_domain program apply posterior in
	(* initialize worklist *)
	let proven_atoms = proof |> Watson.Proof.to_atoms in
	let constraints = program |> ontology |> Ontology.constraints in
	let wl = constraints |> CCList.flat_map (fun c -> initial_worklist c proven_atoms) in
	(* search *)
	let results = Search.stochastic_beam_search (module D) width wl [] in
		results

let models ?width:(width=CCInt.max_int) program posterior goal =
	let brooms = goal
		|> prove ~width:width program posterior
		|> CCList.map (fun state ->
				let proof = state |> Search.State.value in
				let contradictions = contradict ~width:width program posterior proof in
					(state, contradictions)) in
	brooms |> CCList.map (CCPair.merge Model.of_search_states)

(* INPUT / OUTPUT *)

let pp ppf program = let open Fmt in
	pf ppf "Parameters: %a@, Rules: %a@,%a"
		(list ~sep:comma Parameter.pp) program.parameters
		(list ~sep:comma Watson.Rule.pp) program.rules
		Ontology.pp program.ontology

module JSON = struct
	let encode program = `Assoc [
		("type", `String "program");
		("rules", `List (program |> rules |> CCList.map Watson.Rule.JSON.encode));
		("parameters", `List (program |> parameters |> CCList.map Parameter.JSON.encode));
		("evidence", `List (program |> evidence |> CCList.map Evidence.JSON.encode));
		("ontology", program |> ontology |> Ontology.JSON.encode);
	]

	let decode json = let open CCOpt in
		let* rules = JSON.Parse.(find (list Watson.Rule.JSON.decode) "rules" json) in
		let* parameters = JSON.Parse.(find (list Parameter.JSON.decode) "parameters" json) in
		let* evidence = JSON.Parse.(find (list Evidence.JSON.decode) "evidence" json) in
		let* ontology = JSON.Parse.(find Ontology.JSON.decode "ontology" json) in
			return {
				rules = rules;
				parameters = parameters;
				evidence = evidence;
				ontology = ontology;
			}
end