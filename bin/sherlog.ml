(* references for cmd-line parsing *)
let filepath = ref ""
let verbose  = ref false

let spec_list = [
    ("--input",   Arg.Set_string filepath, "Input SherLog (.sl) file");
    ("--verbose", Arg.Set verbose,         "Enables verbose output");
]

let usage_msg = "SherLog Program Interpreter"
let _ = Arg.parse spec_list print_endline usage_msg

(* grab the file contents *)
let contents =
    let channel = open_in !filepath in
    let length = in_channel_length channel in
        really_input_string channel length

let lines = Sherlog.Parse.parse_string contents
let problem = Sherlog.Problem.of_lines lines

(* answer each query *)
let _ = CCList.iter (fun query ->
    (* which q we working on? *)
    let query_string = query
        |> CCList.map Watson.Syntax.Atom.to_string
        |> CCString.concat ", " in
    let _ = print_endline ("Answering: " ^ query_string) in
    (* build the proof *)
    let program = problem.program in
    let _ = print_endline "Starting resolution..." in
    let proof = query
        |> Watson.Resolution.Proof.of_query
        |> Watson.Resolution.Proof.resolve program in
    let solutions = Watson.Resolution.Proof.Solution.of_proof proof in
    let _ = print_endline ("Resolution done. Found " ^ (solutions |> CCList.length |> string_of_int) ^ " solutions.") in
    CCList.iteri (fun i -> fun sol ->
        let map = Watson.Resolution.Proof.Solution.map sol in
        let solution = CCList.map (Watson.Syntax.Atom.apply map) query in
        let solution_string = solution
            |> CCList.map Watson.Syntax.Atom.to_string
            |> CCString.concat ", " in
        let intros = Watson.Resolution.Proof.Solution.introductions sol in
        let intro_string = intros
            (* |> CCList.map (fun (ob, v, p) -> let v = CCList.map (Watson.Syntax.Map.apply map) v in let p = CCList.map (Watson.Syntax.Map.apply map) p in (ob, v, p)) *)
            |> CCList.map (fun (ob, v, p) -> Watson.Syntax.Obligation.applied_representation v p ob)
            |> CCList.rev
            |> CCString.concat "; " in
        let output = if CCString.is_empty intro_string then solution_string else
            intro_string ^ " => " ^ solution_string in
        let _ = print_endline ((string_of_int i) ^ ": " ^ output) in
        ()
    ) solutions
) problem.queries