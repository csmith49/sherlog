module Introduction = struct
	type t = {
		target : Watson.Term.t;
		mechanism : string;
		context : Watson.Term.t list;
		parameters : Watson.Term.t list;
	}

	let target intro = intro.target
	let mechanism intro = intro.mechanism
	let context intro = intro.context
	let parameters intro = intro.parameters

	let make mechanism parameters context target = {
		target = target;
		mechanism = mechanism;
		context = context;
		parameters = parameters;
	}
 
	let to_atom intro = let open Watson.Term in
		let m = Function ("mechanism", [Symbol intro.mechanism]) in
		let p = Function ("parameters", intro.parameters) in
		let c = Function ("context", intro.context) in
		let y = Function ("target", [intro.target]) in
			Watson.Atom.make "introduction" [m ; p; c; y]

	let of_atom atom = let open Watson.Term in
		(* check the relation *)
		if not (CCString.equal (Watson.Atom.relation atom) "introduction") then None else
		(* unpack the arguments *)
		match Watson.Atom.terms atom with
			| [m; p; c; y] -> let open CCOpt in
				let* mechanism = match m with
					| Function ("mechanism", [Symbol m]) -> Some m
					| _ -> None in
				let* parameters = match p with
					| Function ("parameters", p) -> Some p
					| _ -> None in
				let* context = match c with
					| Function ("context", c) -> Some c
					| _ -> None in
				let* target = match y with
					| Function ("target", [y]) -> Some y
					| _ -> None in
				(* no such thing as CCOpt.map4... *)
				return (make mechanism parameters context target)
			| _ -> None

	let to_string intro =
		let t = intro |> target |> Watson.Term.to_string in
		let f = intro |> mechanism in
		let p = intro |> parameters |> CCList.map Watson.Term.to_string |> CCString.concat ", " in
		let c = intro |> context |> CCList.map Watson.Term.to_string |> CCString.concat ", " in
			t ^ " <- " ^ f ^ "(" ^ p ^ " | " ^ c ^ ")"
end

type t = Introduction.t list

let introductions ex = ex

let empty = []
let of_proof proof = proof
	|> Watson.Proof.to_fact
	|> Watson.Fact.atoms
	|> CCList.filter_map Introduction.of_atom

let join = CCList.append

let to_string ex = ex
	|> introductions
	|> CCList.map Introduction.to_string
	|> CCString.concat ", "