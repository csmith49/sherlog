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

let apply program proof =
	(* try applying rules *)
	let proofs = program
		|> rules
		|> CCList.filter_map (Watson.Proof.resolve proof) in
	(* otherwise use the ontology *)
	if (CCList.length proofs) != 0 then proofs
	else let proofs = program
		|> ontology
		|> Ontology.dependencies
		|> CCList.filter_map (Watson.Proof.resolve proof) in
	proofs

module Filter = struct
	type t = Watson.Proof.t list -> Watson.Proof.t list

	let total proofs = proofs
	
	(* TODO - fix this to ensure consistency wrt params and contexts *)
	let intro_consistent proofs = proofs
	
	let length l proofs = proofs
		|> CCList.filter (fun p -> Watson.Proof.length p <= l)
	
	let width w proofs =
		if CCList.length proofs <= w then proofs else
		let random_proofs = proofs
			|> CCRandom.pick_list
			|> CCRandom.sample_without_duplicates ~cmp:Watson.Proof.compare w in
		CCRandom.run random_proofs

	let compose f g proofs = proofs |> f |> g
	let (>>) f g = compose f g
end

let rec prove program filter fact =
	let worklist = [Watson.Proof.of_fact fact] in
		explore_tr program filter worklist []
and explore_tr program filter worklist proven = match worklist with
	| [] -> proven
	| proof :: rest ->
		if Watson.Proof.is_complete proof then
			explore_tr program filter rest (proof :: proven)
		else
			let proofs = proof
				|> apply program
				|> filter in
			explore_tr program filter (proofs @ rest) proven

let (|=) program fact = prove program Filter.total fact

let contradict program filter proof =
	let state = match CCList.last_opt (Watson.Proof.resolutions proof) with
		| Some (_, state) -> state
		| None -> Watson.Proof.State.of_fact (Watson.Fact.empty) in
	let atom = Watson.Atom.make "true" [] in 
	let proofs = program
		(* get the body of the constraints from the ontology *)
		|> ontology
		|> Ontology.constraints
		(* build them into resolutions *)
		|> CCList.map (fun atoms -> Watson.Proof.State.extend atoms state)
		|> CCList.map (CCPair.make atom)
		(* and tack them on to the original proof *)
		|> CCList.map (Watson.Proof.extend proof) in
	explore_tr program filter proofs []

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