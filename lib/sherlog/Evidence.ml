type t = Watson.Atom.t list

let of_atoms atoms = atoms

let to_atoms ev = ev

module JSON = struct
    let encode evidence = `Assoc [
	    ("type", `String "evidence");
	    ("value", evidence |> JSON.Encode.list Watson.Atom.JSON.encode);
    ]

    let decode json = json
        |> JSON.Parse.(find "value" (list Watson.Atom.JSON.decode))
        |> CCOpt.map of_atoms
end
