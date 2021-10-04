open Watson

module Context = struct
    type t = Context of string

    (* construction *)
    module Seed = struct
        let counter = ref 0

        let get ?salt:(salt=1237) () =
            let result = !counter in
            let _ = counter := result + 1 in
            CCInt.logxor result salt
    end
    let fresh () =
        let hash = ()
            |> Seed.get
            |> CCInt.to_string
            |> Base64.encode_exn in
        Context hash

    (* conversion *)
    let to_string = function Context str -> str
    let of_string str = Context str

    (* comparison *)
    let compare left right = match left, right with
        | Context l, Context r -> CCString.compare l r
    
    let equal left right = (compare left right) == 0
    
    let hash = function Context str -> (CCHash.pair CCHash.string CCHash.string) ("context", str)
end

type t = {
    relation : string;
    context : Context.t;
    terms : Watson.Term.t list;
    function_id : string;
    arguments : Watson.Term.t list;
    target : Watson.Term.t;
}

module Embedding = struct

    (* encode values as terms *)
    module Encode = struct
        let symbol name value = Term.Function (name, [Term.Symbol value])

        let singleton name value = Term.Function (name, [value])

        let list name values = Term.Function (name, values)
    end

    (* decode terms to values *)
    module Decode = struct
        let symbol name = function
            | Term.Function (f, [Term.Symbol value]) when CCString.equal f name -> Some value
            | _ -> None

        let singleton name = function
            | Term.Function (f, [value]) when CCString.equal f name -> Some value
            | _ -> None

        let list name = function
            | Term.Function (f, values) when CCString.equal f name -> Some values
            | _ -> None
    end

    (* keys for denoting the content of the embedded term *)
    module Keys = struct
        let atom = "sl:intro"
        let relation = "sl:relation"
        let context = "sl:context"
        let terms = "sl:terms"
        let function_id = "sl:function_id"
        let arguments = "sl:arguments"
        let target = "sl:target"
    end
end

module Functional = struct
    let make relation context terms function_id arguments target = {
        relation=relation;
        context=context;
        terms=terms;
        function_id=function_id;
        arguments=arguments;
        target=target;
    }

    let relation intro = intro.relation
    let context intro = intro.context
    let terms intro = intro.terms
    let function_id intro = intro.function_id
    let arguments intro = intro.arguments
    let target intro = intro.target
end

let to_atom intro = let open Embedding in Atom.make Keys.atom [
    intro |> Functional.relation |> Encode.symbol Keys.relation;
    intro |> Functional.context |> Context.to_string |> Encode.symbol Keys.context;
    intro |> Functional.terms |> Encode.list Keys.terms;
    intro |> Functional.function_id |> Encode.symbol Keys.function_id;
    intro |> Functional.arguments |> Encode.list Keys.arguments;
    intro |> Functional.target |> Encode.singleton Keys.target;
]

let of_atom atom = let open Embedding in
    if not (CCString.equal (Atom.relation atom) Keys.atom) then None else
    let open CCOpt in match Atom.terms atom with
        | [relation; context; terms; function_id; arguments; target] ->
            let* relation = Decode.symbol Keys.relation relation in
            let* context = Decode.symbol Keys.context context in
            let* terms = Decode.list Keys.terms terms in
            let* function_id = Decode.symbol Keys.function_id function_id in
            let* arguments = Decode.list Keys.arguments arguments in
            let* target = Decode.singleton Keys.target target in
            return (Functional.make relation (Context.of_string context) terms function_id arguments target)
        | _ -> None

(* utilities *)

let observed intro = match Functional.target intro with
    | Term.Variable _ -> false
    | _ -> true
