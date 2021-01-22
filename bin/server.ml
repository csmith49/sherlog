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
        begin match query, program with
            | Some query, Some program ->
                let model = query
                    |> Watson.Proof.of_query
                    |> Watson.Proof.resolve program
                    |> Sherlog.Model.of_proof in
                Some (Sherlog.Model.to_json model)
            | _ -> None
        end
    | Some "sample" ->
        (* get program and query *)
        let query = Interface.JSON.Parse.(find identity "query" json)
            |> CCOpt.flat_map Sherlog.Query.of_json in
        let program = Interface.JSON.Parse.(find identity "program" json)
            |> CCOpt.flat_map Sherlog.Program.of_json in
        (* get configuration *)
        let depth = Interface.JSON.Parse.(find int "depth" json)
            |> CCOpt.get_or ~default:CCInt.max_int in
        let width = Interface.JSON.Parse.(find int "width" json)
            |> CCOpt.get_or ~default:CCInt.max_int in
        let seeds = Interface.JSON.Parse.(find int "seeds" json)
            |> CCOpt.get_or ~default:1 in
        let configuration = {Watson.Proof.Random.default_configuration with
            depth = depth; width = width; seeds = seeds;
        } in begin match query, program with
            | Some query, Some program ->
                let model = query
                    |> Watson.Proof.Random.resolve configuration program
                    |> CCList.flat_map Watson.Proof.Solution.of_proof
                    |> Sherlog.Model.of_solutions in
                Some (Sherlog.Model.to_json model)
            | _ -> None
        end
    | _ -> None

(* main *)
let _ =
    let socket = Interface.Network.socket Interface.Network.local_address !port in
    let server = Interface.Network.server handler socket in
    Interface.Network.run server