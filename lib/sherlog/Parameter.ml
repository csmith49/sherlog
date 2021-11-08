module Domain = struct
	type t =
		| Unit
		| Positive
		| Real

	let to_string = function
		| Unit -> "unit"
		| Positive -> "positive"
		| Real -> "real"

	let pp ppf = function
		| Unit -> Fmt.pf ppf "[0, 1]"
		| Positive -> Fmt.pf ppf "ℝ⁺"
		| Real -> Fmt.pf ppf "ℝ"

	module JSON = struct
		let encode = function
			| Unit -> JSON.Encode.string "unit"
			| Positive -> JSON.Encode.string "positive"
			| Real -> JSON.Encode.string "real"

		let decode json = match JSON.Parse.string json with
			| Some "unit" -> Some Unit
			| Some "positive" -> Some Positive
			| Some "real" -> Some Real
			| _ -> None
	end
end
	
type t = {
	name : string;
	domain : Domain.t;
	dimension : int;
}

module Functional = struct
	let name param = param.name
	let domain param = param.domain
	let dimension param = param.dimension

	let make name domain dimension = {
		name=name;
		domain=domain;
		dimension=dimension;
	}
end

let to_string parameter =
	let name = parameter |> Functional.name in
	let domain = parameter |> Functional.domain |> Domain.to_string in
	let dimension = parameter
		|> Functional.dimension
		|> CCInt.to_string in
	name ^ " : " ^ domain ^ "[" ^ dimension ^ "]"

let pp ppf param = let open Fmt in
	pf ppf "%s : %a^%d"
		(param |> Functional.name)
		Domain.pp (param |> Functional.domain)
		(param |> Functional.dimension)

module JSON = struct
	let encode parameter = `Assoc [
			("type", `String "parameter");
			("name", parameter |> Functional.name |> JSON.Encode.string);
			("domain", parameter |> Functional.domain |> Domain.JSON.encode);
			("dimension", parameter |> Functional.dimension |> JSON.Encode.int);
		]

	let decode json = let open CCOpt in
		let* name = JSON.Parse.(find "name" string json) in
		let* domain = JSON.Parse.(find "domain" Domain.JSON.decode json) in
		let* dimension = JSON.Parse.(find "dimension" int json) in
			return (Functional.make name domain dimension)
end