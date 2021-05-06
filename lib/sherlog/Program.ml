open Watson

type t = {
	rules : Rule.t list;
	parameters : Parameter.t list;
	evidence : Evidence.t list;
	ontology : Ontology.t;
}

let rules program = program.rules
let parameters program = program.parameters
let evidence program = program.evidence
let ontology program = program.ontology

let make rules parameters evidence ontology = {
	rules = rules;
	parameters = parameters;
	evidence = evidence;
	ontology = ontology;
}

let apply_rules program proof = program
	|> rules
	|> CCList.filter_map (Watson.Proof.resolve proof)

let apply_ontology program proof = program
	|> ontology
	|> Ontology.dependencies
	|> CCList.filter_map (Watson.Proof.resolve proof)

let apply program proof =
	let proofs = apply_rules program proof in
	if (CCList.length proofs) != 0 then proofs
	else apply_ontology program proof

module Filter = struct
	type t = Watson.Proof.t list -> Watson.Proof.t list

	let total proofs = proofs
	
	let rec intro_consistent proofs = CCList.filter is_intro_consistent proofs
	and is_intro_consistent proof =
		let intro_tags = proof
			|> Explanation.of_proof
			|> Explanation.introductions
			|> CCList.map Explanation.Introduction.tag in
		let num_intros = CCList.length intro_tags in
		let num_unique_intros = intro_tags
			|> CCList.sort_uniq ~cmp:(CCList.compare Watson.Term.compare)
			|> CCList.length in
		CCInt.equal num_intros num_unique_intros

	let length l proofs = 
		proofs
		|> CCList.filter (fun p -> (p |> Watson.Proof.witnesses |> CCList.length) <= l)
	

	(* width done via sampling from categorical w/ heuristic scores *)

	(* we rely on the gumbel trick *)
	let gumbel_trick weight = let open CCRandom in
		let scale u = u |> log |> log |> fun x -> 0.0 -. x in
		(float_range 0.0 1.0) >|= scale >|= CCFloat.add (log weight)

	let categorical weights : int CCRandom.t = fun state ->
		let scores = weights
			|> CCList.map gumbel_trick
			|> CCList.map (CCRandom.run ~st:state)
			|> CCList.mapi CCPair.make in
		let sort_key (li, ls) (ri, rs) = if ls >= rs then (li, ls) else (ri, rs) in
		let argmax = scores
			|> CCList.fold_left sort_key (0, -1.0) in
		fst argmax

	let score proof : float = 
		let intros = proof |> Explanation.of_proof |> Explanation.introductions in
		(* total introductions *)
		let total_intros = intros |> CCList.length in
		(* constrained introductions *)
		let constrained_intros = intros |> CCList.filter Explanation.Introduction.is_constrained |> CCList.length in
		(* proof length *)
		let proof_length = proof |> Watson.Proof.to_atoms |> CCList.length in
		(* linear combo *)
		-1.0 *. (float total_intros) +. 0.3 *. (float constrained_intros) +. 0.2 *. (float proof_length)

	let random_proof score proofs : Watson.Proof.t option CCRandom.t = fun state ->
		let weights = proofs |> CCList.map score in
		let index = CCRandom.run ~st:state (categorical weights) in
		CCList.get_at_idx index proofs

	let width w proofs =
		if CCList.length proofs <= w then proofs else
		let random_proofs = proofs
			|> random_proof score
			|> CCList.replicate w
			|> CCRandom.list_seq in
		CCRandom.run random_proofs |> CCList.keep_some

	let compose f g proofs = proofs |> f |> g
	let (>>) f g = compose f g
end

let rec explore program expand filter worklist proven =
	match worklist with
	| [] -> proven
	| proof :: rest ->
		if Watson.Proof.is_resolved proof then
			let proven = proof :: proven in
			explore program expand filter rest proven
		else
			let proofs = proof
				|> expand program
				|> CCList.append rest
				|> filter in
			explore program expand filter proofs proven

let prove program filter goal =
	let initial_proof = Watson.Proof.of_atoms goal [] in
	let worklist = [initial_proof] in
		explore program apply filter worklist []

let (|=) program goal = prove program Filter.total goal

let contradict program filter proof =
	let proven_atoms = proof
		|> Watson.Proof.to_atoms in
	let constraints = program
		|> ontology
		|> Ontology.constraints in
	let initial_proofs = constraints
		|> CCList.map (fun c -> Watson.Proof.of_atoms c proven_atoms) in
	explore program apply_rules filter initial_proofs []

let models program filter goal =
	let mk (p, cs) = match cs with
		| [] -> [Model.of_proof p]
		| _ -> CCList.map (Model.of_proof_and_contradiction p) cs in
	let proofs = goal
		|> prove program filter
		|> CCList.map (CCPair.dup_map (contradict program filter)) in
	CCList.flat_map mk proofs

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

	let decode json =
		let rules = JSON.Parse.(find (list Watson.Rule.JSON.decode) "rules" json) in
		let parameters = JSON.Parse.(find (list Parameter.JSON.decode) "parameters" json) in
		let evidence = JSON.Parse.(find (list Evidence.JSON.decode) "evidence" json) in
		let ontology = JSON.Parse.(find Ontology.JSON.decode "ontology" json) in
		match rules, parameters, evidence, ontology with
			| Some rules, Some parameters, Some evidence, Some ontology -> Some {
				rules = rules;
				parameters = parameters;
				evidence = evidence;
				ontology = ontology;
			}
			| _ -> None
end