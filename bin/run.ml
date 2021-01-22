(* references for cmd-line parsing *)
let filepath = ref ""
let verbose  = ref false
let depth = ref CCInt.max_int
let width = ref CCInt.max_int
let seeds = ref 1

let spec_list = [
    ("--input",   Arg.Set_string filepath, "Input SherLog (.sl) file");
    ("--verbose", Arg.Set verbose,         "Enables verbose output");
    ("--depth",   Arg.Set_int depth,       "Sets the maximum depth of resolution");
    ("--width",   Arg.Set_int width,       "Sets the width of the resolution beam");
    ("--seeds",   Arg.Set_int seeds,       "Sets the number of resolution seeds");
]

let usage_msg = "SherLog Program Interpreter"
let _ = Arg.parse spec_list print_endline usage_msg

(* grab the file contents *)
let contents =
    let channel = open_in !filepath in
    let length = in_channel_length channel in
        really_input_string channel length

let problem = Sherlog.parse contents

let _ = print_endline "Parsed program:"
let _ = print_endline (Sherlog.Problem.to_string problem)

(* answer each query *)
let _ = CCList.iter (fun query ->
    (* which q we working on? *)
    let query_string = query
        |> CCList.map Watson.Atom.to_string
        |> CCString.concat ", " in
    let _ = print_endline ("Answering: " ^ query_string) in
    (* build the proof *)
    let program = Sherlog.Problem.program problem in
    let _ = print_endline "Starting resolution..." in
    let configuration = { Watson.Proof.Random.default_configuration with
        depth = !depth; width = !width; seeds = !seeds;
    } in
    let proofs = query
        |> Watson.Proof.Random.resolve configuration program in
    let solutions = CCList.flat_map Watson.Proof.Solution.of_proof proofs in
    let _ = print_endline ("Resolution done. Found " ^ (solutions |> CCList.length |> string_of_int) ^ " solutions.") in
    let _ = CCList.iteri (fun i -> fun sol ->
        let map = Watson.Proof.Solution.map sol in
        let solution = CCList.map (Watson.Atom.apply map) query in
        let solution_string = solution
            |> CCList.map Watson.Atom.to_string
            |> CCString.concat ", " in
        let intros = Watson.Proof.Solution.introductions sol in
        let intro_string = intros
            |> CCList.map (fun (guard, parameters, _, target) -> Watson.Guard.applied_representation target parameters guard)
            |> CCList.rev
            |> CCString.concat "; " in
        let output = if CCString.is_empty intro_string then solution_string else
            intro_string ^ " => " ^ solution_string in
        let _ = print_endline ((string_of_int i) ^ ": " ^ output) in
        ()
    ) solutions in ()
) (Sherlog.Problem.queries problem)
