type t = string

let compare = CCString.compare
let equal = CCString.equal

let to_string id = id

let uniq = CCList.uniq ~eq:equal