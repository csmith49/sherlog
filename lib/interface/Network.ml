open Lwt

(* exposed type aliases *)
type handler = Yojson.Basic.t -> Yojson.Basic.t option
type port = int
type address = Lwt_unix.inet_addr
type socket = Lwt_unix.file_descr
type server = unit -> unit Lwt.t

(* the local address *)
let local_address = Unix.inet_addr_loopback

(* constructs a connection from a socket *)
let connection_of_socket socket =
    let ic = Lwt_io.of_fd ~mode:Lwt_io.Input socket in
    let oc = Lwt_io.of_fd ~mode:Lwt_io.Output socket in
        (ic, oc)

let safe_parse string = try Some (Yojson.Basic.from_string string) with _ -> None

let close_connection (ic, oc) () =
    return_unit >>= (fun () -> Lwt_io.close ic) >>= (fun () -> Lwt_io.close oc)

(* applies a handler to a connection *)
let handle_connection handler (ic, oc) () = Lwt_io.read_line_opt ic >>=
    (fun msg -> 
        match msg with
        | Some msg -> begin match msg |> safe_parse |> CCOpt.flat_map handler with
            | Some result ->
                let reply = Yojson.Basic.to_string result in
                Lwt_io.write_line oc reply >>= (close_connection (ic, oc))
            | None ->
                let reply = Yojson.Basic.to_string (`String "failure") in
                Lwt_io.write_line oc reply >>= (close_connection (ic, oc)) end
        | _ -> return_unit)

(* constructs a handler for the output of lwt_unix.accept *)
let handle_socket handler (socket, _) =
    let connection = connection_of_socket socket in
        handle_connection handler connection ()

(* constructs a socket *)
let socket address port = let open Lwt_unix in
    let sock = socket PF_INET SOCK_STREAM 0 in
    let addr = ADDR_INET (address, port) in
    let _ = bind sock addr in
    let _ = listen sock 10000 in
    let _ = setsockopt sock SO_REUSEADDR in (* ensures we can restart the server quickly *)
        sock

(* build the server *)
let server handler socket =
    let rec serve () =
        Lwt_unix.accept socket >>= handle_socket handler >>= serve
    in serve

let run server = Lwt_main.run (server ())