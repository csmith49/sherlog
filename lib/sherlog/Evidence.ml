type t = Watson.Atom.t list

let make atoms = atoms

let to_atoms ev = ev

let to_json evidence = `Assoc [
	("type", `String "evidence");
	("value", `List (evidence |> CCList.map Watson.Atom.JSON.encode));
]

let of_json json =
	let atoms = JSON.Parse.(find (list Watson.Atom.JSON.decode) "value" json) in
	CCOpt.map make atoms