type t = Watson.Atom.t list

let make atoms = atoms

let to_fact ev = Watson.Fact.of_atoms ev

module JSON = struct
	let encode evidence = `Assoc [
		("type", `String "evidence");
		("value", `List (evidence |> CCList.map Watson.Atom.JSON.encode));
	]

	let decode json =
		let atoms = JSON.Parse.(find (list Watson.Atom.JSON.decode) "value" json) in
		CCOpt.map make atoms
end