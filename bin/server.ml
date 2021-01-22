(* reference for argument parsing *)
let port = ref 7999 (* default port, changeable via args *)

(* argument parsing *)
let spec_list = [
    ("--port", Arg.Set_int port, "Port to host local server on");
]
let usage_msg = "Server for SherLog"
let _ = Arg.parse spec_list print_endline usage_msg

(* message handling *)

let handler json = match Interface.JSON.Parse.(find string "command" json) with
    | Some "parse" ->
        Interface.JSON.Parse.(find string "program" json)
            |> CCOpt.map Sherlog.parse
            |> CCOpt.map Sherlog.Problem.to_json
    | Some "run" ->
        let query = Interface.JSON.Parse.(find identity "query" json) 
            |> CCOpt.flat_map Sherlog.Query.of_json in
        let program = Interface.JSON.Parse.(find identity "program" json)
            |> CCOpt.flat_map Sherlog.Program.of_json in
        let depth = match Interface.JSON.Parse.(find identity "depth" json) with
            | Some (`Int depth) -> depth
            | _ -> 1000 in
        begin match query, program with
            | Some query, Some program ->
                let model = query
                    |> Watson.Proof.of_query
                    |> Watson.Proof.resolve ~max_depth:depth program
                    |> Sherlog.Model.of_proof in
                Some (Sherlog.Model.to_json model)
            | _ -> None
        end
    | _ -> None

(* main *)
let _ =
    let socket = Interface.Network.socket Interface.Network.local_address !port in
    let server = Interface.Network.server handler socket in
    Interface.Network.run server