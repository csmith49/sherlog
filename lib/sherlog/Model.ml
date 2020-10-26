(* module Value = struct
    type t = Watson.Logic.ETerm.t

    let of_term = Watson.Logic.ETerm.lift
    let of_eterm x = x

    let to_string = Watson.Logic.ETerm.to_string

    let to_json _ = `Null

    let equal = Watson.Logic.ETerm.equal
end

module Location = struct
    type t = {
        function_name : string;
        module_name : string;
    }

    let to_string loc = loc.function_name ^ " @ " ^ loc.module_name

    let to_json _ = `Null
end

module Distribution = struct
    type t =
        | Normal
        | Bernoulli

    let to_json _ = `Null
end

module Command = struct
    type t =
        | Assign of Location.t * Value.t list
        | Sample of Distribution.t * Value.t list

    let to_json _ = `Null
end

module Observation = struct
    type t = (string * Value.t) list
end

(* we're using strings to identify nodes for now *)
module Graph = Data.Graph.Make(CCString)

(* model itself is just a graph with nodes labeled by commands *)
type t = (Command.t, unit) Graph.t

(* this module holds all the important compilation pipeline stuff *)
module Compile = struct
    
end *)