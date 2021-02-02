type domain =
	| Unit
	| Positive
	| Real
	
type t = {
	name : string;
	domain : domain;
}

let name param = param.name
let domain param = param.domain

let make name domain = {
	name = name;
	domain = domain;
}

let to_string parameter =
	let domain = match domain parameter with
		| Unit -> "[0, 1]"
		| Positive -> "ℝ⁺"
		| Real -> "ℝ" in
	let name = parameter |> name in
	name ^ " : " ^ domain

let domain_of_string = function
	| "unit" -> Some Unit
	| "positive" -> Some Positive
	| "real" -> Some Real
	| _ -> None

module JSON = struct
	let encode parameter = 
		let domain = match domain parameter with
			| Unit -> `String "unit"
			| Positive -> `String "positive"
			| Real -> `String "real" in
		`Assoc [
			("type", `String "parameter");
			("name", `String (parameter |> name));
			("domain", domain)
		]

	let decode json =
		let name = JSON.Parse.(find string "name" json) in
		let domain = JSON.Parse.(find string "domain" json) in
		CCOpt.map2 make name (CCOpt.flat_map domain_of_string domain)
end