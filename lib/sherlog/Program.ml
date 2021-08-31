open Watson

(* BASICS *)

type t = {
	rules : Rule.t list;
	parameters : Parameter.t list;
}

let is_intro_rule rule = rule
	|> Rule.head
	|> Atom.relation
	|> CCString.equal Introduction.relation

let rules program = program.rules
let introduction_rules program = program
	|> rules
	|> CCList.filter is_intro_rule
let non_introduction_rules program = program
	|> rules
	|> CCList.filter (fun r -> not (is_intro_rule r))

let parameters program = program.parameters

let make rules parameters = {
	rules = rules;
	parameters = parameters;
}

(* SEMANTICS *)

module Semantics = struct	
	type t = Proof.t -> Proof.t list

	module M = struct
		let return = CCList.return
		let bind xs f = CCList.flat_map f xs
		let (>>=) = bind
		let zero = []
		let plus = (@)
		let (++) = plus
	end

	(* CONSTRUCTION *)
	
	let result xs = fun _ -> xs
	
	let of_rule rule proof = match Proof.resolve proof rule with
	| Some result -> M.return result
	| None -> M.zero

	(* COMBINATORS *)

	module Combinator = struct
		(* DISJUNCTIVE *)
		let try_or l r = fun p -> match l p with
			| [] -> r p
			| results -> results
		let (<|>) = try_or
		let choice ss = CCList.fold_left try_or (result M.zero) ss

		(* CONJUNCTIVE *)
		let try_and l r = fun p -> M.(l p ++ r p)
		let (<&>) = try_and
		let union ss = CCList.fold_left try_and (result M.zero) ss

		(* SEQUENCING *)
		let seq l r = fun p -> M.(l p >>= r)
		let (>>) = seq

		(* FAILURE *)
		let attempt (sem : t) : t = sem <|> M.return

		(* RECURSION *)
		let rec fixpoint sem proof = match sem proof with
			| [] -> [proof]
			| results -> M.(results >>= fixpoint sem)
	end
end

module Filter = struct
	type t = Watson.Proof.t -> bool

	let negate filter = fun x -> not (filter x)

	let intro_consistent proof =
		let intro_tags = proof
			|> Watson.Proof.to_atoms
			|> CCList.filter_map Introduction.of_atom
			|> CCList.map Introduction.introduction_consistency_tag in
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
	let classical = program
		|> non_introduction_rules
		|> CCList.map Semantics.of_rule
		|> Semantics.Combinator.union in (* choice here doesn't quite do what you want... *)
	let introduction = program
		|> introduction_rules
		|> CCList.map Semantics.of_rule
		|> Semantics.Combinator.union in
	Semantics.Combinator.(
		let fp = fixpoint classical in
		fp >> introduction >> fp
	)

(* SEARCH *)

let linear_domain : t -> (t -> Watson.Proof.t -> Watson.Proof.t list) -> Posterior.t -> (module Search.Domain with type t = Watson.Proof.t) = fun program successor posterior ->
	(module struct
		type t = Watson.Proof.t

		let features = Posterior.Operator.apply (Posterior.operator posterior)
		let score = Posterior.Ensemble.apply (Posterior.ensemble posterior)
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
	let (module D) = linear_domain program apply posterior in
	(* initialize worklist *)
	let wl = initial_worklist goal [] in
	(* search *)
	Search.stochastic_beam_search (module D) width wl []
	
let explanations ?width:(width=CCInt.max_int) program posterior goal = goal
	|> prove ~width:width program posterior
	|> CCList.map Search.State.to_tuple
	|> CCList.map (CCPair.fold Explanation.of_proof)

(* INPUT / OUTPUT *)

let pp ppf program = let open Fmt in
	pf ppf "Parameters: %a@, Rules: %a"
		(list ~sep:comma Parameter.pp) program.parameters
		(list ~sep:comma Watson.Rule.pp) program.rules


let to_json program = `Assoc [
	("type", `String "program");
	("rules", `List (program |> rules |> CCList.map Watson.Rule.JSON.encode));
	("parameters", `List (program |> parameters |> CCList.map Parameter.JSON.encode));
]

let of_json json = let open CCOpt in
	let* rules = JSON.Parse.(find (list Watson.Rule.JSON.decode) "rules" json) in
	let* parameters = JSON.Parse.(find (list Parameter.JSON.decode) "parameters" json) in
		return {
			rules = rules;
			parameters = parameters;
		}