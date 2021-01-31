module ASet = CCSet.Make(Atom)

type t = ASet.t

let atoms = ASet.to_list
let of_atoms = ASet.of_list
let singleton = ASet.singleton
let empty = ASet.empty

let conjoin = ASet.union
let add = ASet.add

let apply h = ASet.map (Atom.apply h)

let to_string fact = fact
	|> atoms
	|> CCList.map Atom.to_string
	|> CCString.concat ", "

let discharge fact = match ASet.choose_opt fact with
	| Some atom -> Some (atom, ASet.remove atom fact)
	| _ -> None

let variables fact = fact
    |> atoms
    |> CCList.flat_map Atom.variables
    |> Identifier.uniq

let mem = ASet.mem