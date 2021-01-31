type t = {
	name : string;
	index : int;
}

let name id = id.name
let index id = id.index

let reindex i id = { id with index = i; }

let compare left right = match CCString.compare left.name right.name with
	| 0 -> CCInt.compare left.index right.index
	| (_ as r) -> r

let equal left right = (compare left right) == 0

let to_string id = id.name ^ "_" ^ (CCInt.to_string id.index)
let of_string name = {
	name = name;
	index = 0;
}

let uniq = CCList.uniq ~eq:equal

let avoiding_index ids =
	let rec find_unused xs target = match xs with
		| [] -> target
		| x :: xs ->
			if x > target then target else find_unused xs (target + 1) in
	let used = ids |> CCList.map index |> CCList.sort_uniq ~cmp:CCInt.compare in
	find_unused used 0