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

let handler json = let open CCOpt in match JSON.Parse.(find string "type" json) with
    | Some "parse-source-request" ->
        let* source = JSON.Parse.(find string "source" json) in
        let lines = source |> Sherlog.IO.parse in
        let response = `Assoc [
            ("type", `String "parse-source-response");
            ("program", lines |> Sherlog.IO.program_of_lines |> Sherlog.Program.to_json);
            ("evidence", lines
                |> Sherlog.IO.evidence_of_lines
                |> CCList.map Sherlog.Evidence.to_json
                |> JSON.Make.list
            );
        ] in
        return response

    | Some "query-request" ->
        (* core programmatic info *)
        let* program = JSON.Parse.(find Sherlog.Program.of_json "program" json) in
        let* posterior = JSON.Parse.(find Sherlog.Posterior.of_json "posterior" json) in
        let* query = JSON.Parse.(find Sherlog.Evidence.of_json "evidence" json) in
        (* parameters for the search *)
        let search_width = JSON.Parse.(find int "search-width" json)
            |> CCOpt.get_or ~default:CCInt.max_int in
        (* get explanations *)
        let explanations = query
            |> Sherlog.Evidence.to_atoms
            |> Sherlog.Program.explanations ~width:search_width program posterior
            |> CCList.map (Sherlog.Explanation.to_json Sherlog.Explanation.Term.to_json)
            |> shuffle in
        let response = `Assoc [
            ("type", `String "query-response");
            ("explanations", `List explanations)
        ] in
        return response

    | _ -> None

(* main *)
let _ =
    let socket = Network.socket Network.local_address !port in
    let server = Network.server ~timeout:(float_of_int !timeout) handler socket in
    Network.run server