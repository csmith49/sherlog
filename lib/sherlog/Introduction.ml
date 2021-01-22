type t = {
    f : string;
    args : Watson.Term.t list;
    context : Watson.Term.t list;
}

let placeholder = {
    f = "placeholder";
    args = [];
    context = [];
}

let variables intro = intro.args
    |> CCList.flat_map Watson.Term.variables

let to_json intro = `Assoc [
    ("function", `String intro.f);
    ("arguments", `List (CCList.map Utility.Term.to_json intro.args));
    ("context", `List (CCList.map Utility.Term.to_json intro.context));
]