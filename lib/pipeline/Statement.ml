type 'a t = {
  target : string;
  function_id : string;
  arguments : 'a Value.t list;
}

  let to_json literal_to_json statement = `Assoc [
    ("type", `String "statement");
    ("target", `String statement.target);
    ("function", `String statement.function_id);
    ("arguments", `List (statement.arguments |> CCList.map (Value.to_json literal_to_json)));
  ]
