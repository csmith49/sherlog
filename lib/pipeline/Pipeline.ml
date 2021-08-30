(* import the other modules to put them into the global space *)
module Value = Value
module Statement = Statement

type 'a t = 'a Statement.t list

let to_json literal_to_json pipeline = 
 `Assoc [
    ("type", `String "pipeline");
    ("statements", `List (pipeline |> CCList.map (Statement.to_json literal_to_json)));
]
