open Watson

type t = {
	dependencies : Rule.t list;
	constraints : (Atom.t list) list;
}

let empty = {
	dependencies = [];
	constraints = [];
}

let make deps cons = { dependencies = deps; constraints = cons }

let dependencies ontology = ontology.dependencies
let constraints ontology = ontology.constraints

let add_dependency dep ontology = { ontology with dependencies = dep :: (dependencies ontology) }
let add_constraint con ontology = { ontology with constraints = con :: (constraints ontology) }

module JSON = struct
	let encode_constraint atoms = `List (atoms |> CCList.map Atom.JSON.encode)

	let encode ontology = `Assoc [
		("type", `String "ontology");
		("dependencies", `List (ontology |> dependencies |> CCList.map Rule.JSON.encode));
		("constraints", `List (ontology |> constraints |> CCList.map encode_constraint));
	]

	let decode json =
		let dependencies = JSON.Parse.(find (list Rule.JSON.decode) "dependencies" json) in
		let constraints = JSON.Parse.(find (list (list Atom.JSON.decode)) "constraints" json) in
		CCOpt.map2 (fun dep -> fun con -> {dependencies = dep; constraints = con;}) dependencies constraints
end