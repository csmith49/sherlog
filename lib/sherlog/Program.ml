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
	let rec intro_consistent proofs = CCList.filter is_intro_consistent proofs
	and is_intro_consistent proof =
		let intros = proof
			|> Watson.Proof.to_atoms
			|> CCList.filter_map Explanation.Introduction.of_atom in
		let size = intros |> CCList.length in
		let unique_intros = intros
			|> CCList.map intro_hash
			|> CCList.uniq ~eq:(CCList.equal Watson.Term.equal)
			|> CCList.length in
		CCInt.equal size unique_intros
	and intro_hash intro =
		let mechanism = Watson.Term.Symbol (Explanation.Introduction.mechanism intro) in
		let parameters = Explanation.Introduction.parameters intro in
		let context = Explanation.Introduction.context intro in
			mechanism :: (parameters @ context)
	
	let length l proofs = proofs
		|> CCList.filter (fun p -> (p |> Watson.Proof.witnesses |> CCList.length) <= l)
	
	let width w proofs =
		if CCList.length proofs <= w then proofs else
		let random_proofs = proofs
			|> CCRandom.pick_list
			|> CCRandom.sample_without_duplicates ~cmp:Stdlib.compare w in
		CCRandom.run random_proofs

	let compose f g proofs = proofs |> f |> g
	let (>>) f g = compose f g
end

let rec prove program filter goal =
	let worklist = [Watson.Proof.of_atoms goal] in
		explore_tr program filter worklist []
and explore_tr program filter worklist proven = match worklist with
	| [] -> proven
	| proof :: rest ->
		if Watson.Proof.is_resolved proof then
			explore_tr program filter rest (proof :: proven)
		else
			let proofs = proof
				|> apply program
				|> filter in
			explore_tr program filter (proofs @ rest) proven

let (|=) program goal = prove program Filter.total goal

let contradict program filter proof =
	let state = proof |> Watson.Proof.state in
	let proofs = program
		|> ontology
		|> Ontology.constraints
		|> CCList.map (Watson.Proof.State.reset_goal state)
		|> CCList.map (Watson.Proof.of_state) in
	explore_tr program filter proofs []

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