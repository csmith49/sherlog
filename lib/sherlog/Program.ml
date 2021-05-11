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
	
	let constraint_avoiding ontology proofs =
		let constraints = ontology
			|> Ontology.constraints in
		let check proof =
			let atoms = proof |> Watson.Proof.to_atoms in
			constraints
				|> CCList.map (fun c -> Watson.Atom.embed_all c atoms)
				|> CCList.for_all CCList.is_empty in
		CCList.filter check proofs

	let rec intro_consistent proofs = CCList.filter is_intro_consistent proofs
	and is_intro_consistent proof =
		let intro_tags = proof
			|> Explanation.of_proof
			|> Explanation.introductions
			|> CCList.map Explanation.Introduction.tag in
			(* |> CCList.filter (CCList.for_all Watson.Term.is_ground) in *)
		let num_intros = CCList.length intro_tags in
		let num_unique_intros = intro_tags
			|> CCList.sort_uniq ~cmp:(CCList.compare Watson.Term.compare)
			|> CCList.length in
		let result = CCInt.equal num_intros num_unique_intros in
		result

	let length l proofs = 
		proofs
		|> CCList.filter (fun p -> (p |> Watson.Proof.witnesses |> CCList.length) <= l)
	
	let beam_width score w proofs =
		if CCList.length proofs <= w then proofs else
		let random_proofs = proofs
			|> Posterior.random_proof score
			|> CCList.replicate w
			|> CCRandom.list_seq in
		CCRandom.run random_proofs
	
	let uniform_width w proofs =
		if CCList.length proofs <= w then proofs else
		let random_proofs = proofs
			|> CCRandom.pick_list
			|> CCList.replicate w
			|> CCRandom.list_seq in
		CCRandom.run random_proofs

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

let models program pos_filter neg_filter goal =
	let brooms = goal
		|> prove program pos_filter
		|> Filter.constraint_avoiding (ontology program)
		|> CCList.map (CCPair.dup_map (contradict program neg_filter)) in
	CCList.map (fun (handle, bristles) -> Model.of_proof_and_contradictions handle bristles) brooms

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