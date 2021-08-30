(* reference for argument parsing *)
let port = ref 7999 (* default port, changeable via args *)
let timeout = ref 3600 (* doesn't do anything atm - lwt requires cooperative threading, and sherlog isn't cooperating *)

(* argument parsing *)
let spec_list = [
    ("--port", Arg.Set_int port, "Port to host local server on");
    ("--timeout", Arg.Set_int timeout, "Max time (in seconds) per query");
]
let usage_msg = "Server for SherLog"
let _ = Arg.parse spec_list print_endline usage_msg

(* utility *)
let rec shuffle xs = match xs with
    | [] -> []
    | xs ->
        (* get a random index *)
        let idx' = xs
            |> CCList.length
            |> CCRandom.int in
        let idx = CCRandom.run idx' in
        (* split xs into x and xs' *)
        let x = CCList.get_at_idx_exn idx xs in
        let xs' = CCList.remove_at_idx idx xs in
            x :: (shuffle xs')

(* message handling *)
let handler json = match JSON.Parse.(find string "command" json) with
    | Some "parse" ->
        JSON.Parse.(find string "program" json)
            |> CCOpt.map Sherlog.IO.parse
            |> CCOpt.map Sherlog.Program.to_json
    | Some "query" -> let open CCOpt in
        (* get core program and query *)
        let* program = JSON.Parse.(find Sherlog.Program.of_json "program" json) in
        let* query = JSON.Parse.(find Sherlog.Evidence.JSON.decode "query" json) 
            |> CCOpt.map Sherlog.Evidence.to_atoms in
        (* get parameters for search *)
        let search_width = JSON.Parse.(find int "width" json)
            |> CCOpt.get_or ~default:CCInt.max_int in
        let operator = JSON.Parse.(find (list string) "contexts" json)
            |> CCOpt.get_or ~default:[]
            |> Sherlog.Posterior.Operator.of_contexts in
        let parameterization = JSON.Parse.(find Sherlog.Posterior.Parameterization.JSON.decode "parameterization" json)
            |> CCOpt.get_or ~default:(CCList.replicate (CCList.length operator) 1.0) in
        (* build score function *)
        let posterior = Sherlog.Posterior.make operator parameterization in
        (* build filter from parameters *)
        let explanations = query
            |> Sherlog.Program.explanations ~width:search_width program posterior
            |> CCList.map (Sherlog.Explanation.to_json Sherlog.Explanation.Term.to_json)
            |> shuffle in
        return (`List explanations)
    | _ -> None

(* main *)
let _ =
    let socket = Network.socket Network.local_address !port in
    let server = Network.server ~timeout:(float_of_int !timeout) handler socket in
    Network.run server