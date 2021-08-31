open Watson

module type ATTRIBUTE_EMBEDDING = sig
  type attribute

  val key : string
  val to_term : attribute -> Watson.Term.t
  val of_term : Watson.Term.t -> attribute option
end

module Target = struct
  type t = Term.t
  let key = "sl:target"
  let to_term target = Term.Function (key, [target])
  let of_term = function
    | Term.Function (k, [target]) when CCString.equal k key -> Some target
    | _ -> None
end

module FunctionID = struct
  type t = string
  let key = "sl:function_id"
  let to_term function_id = Term.Function (key, [Term.Symbol function_id])
  let of_term = function
    | Term.Function (k, [Term.Symbol function_id]) when CCString.equal k key -> Some function_id
    | _ -> None
end

module Arguments = struct
  type t = Term.t list
  let key = "sl:arguments"
  let to_term arguments = Term.Function (key, arguments)
  let of_term = function
    | Term.Function (k, arguments) when CCString.equal k key -> Some arguments
    | _ -> None
end

module Context = struct
  type t = Term.t list
  let key = "sl:context"
  let to_term context = Term.Function (key, context)
  let of_term = function
    | Term.Function (k, context) when CCString.equal k key -> Some context
    | _ -> None
end

type t = {
  target : Target.t;
  function_id : FunctionID.t;
  arguments : Arguments.t;
  context : Context.t;
}

let make target function_id arguments context = {
  target = target;
  function_id = function_id;
  arguments = arguments;
  context = context;
}

let target intro = intro.target
let function_id intro = intro.function_id
let arguments intro = intro.arguments
let context intro = intro.context

let relation = "sl:introduction"

let to_atom intro = Atom.make relation [
  Target.to_term intro.target;
  FunctionID.to_term intro.function_id;
  Arguments.to_term intro.arguments;
  Context.to_term intro.context;
]

let of_atom atom =
  (* check the relation is correct *)
  if not (CCString.equal (Atom.relation atom) relation) then None else
  (* unpack the arguments *)
  let open CCOpt in match Atom.terms atom with
    | [target; function_id; arguments; context] ->
      (* convert arguments, fail-through if None encountered *)
      let* target = Target.of_term target in
      let* function_id = FunctionID.of_term function_id in
      let* arguments = Arguments.of_term arguments in
      let* context = Context.of_term context in
      (* wraps result in Some *)
      return (make target function_id arguments context)
    | _ -> None

let targetless_identifier intro =
  let function_id = intro
    |> function_id in
  let arguments = intro
    |> arguments
    |> CCList.map Term.to_string
    |> CCString.concat ", " in
  let context = intro
    |> context
    |> CCList.map Term.to_string
    |> CCString.concat ", " in
  function_id ^ "[" ^ arguments ^ "|" ^ context ^ "]"

let is_constrained intro = match intro.target with
  | Watson.Term.Variable _ -> true
  | _ -> false

let introduction_consistency_tag intro =
  let r = Watson.Term.Symbol intro.function_id in
  r :: (intro.arguments @ intro.context)