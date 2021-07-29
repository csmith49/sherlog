(** {0 Network}
TCP/IP interface for sending and receiving JSON objects. The primary interface is through the {!type:handler} type.
*)

(** {1 Basic Types} *)

(** These type aliases are wrappers around Lwt representations. *)

type port = int
(** A [port] is the TCP network port a server can communicate over. *)

type address = Lwt_unix.inet_addr
(** An [address] is a network address. *)

type socket
(** A [socket] is an endpoint for sending and receiving data. Should be constructed with the {!val:socket} function. *)

type server
(** A [server] iteratively runs a handler to process communications from a {!type:socket}. Should be constructed via the {!val:server} function. *)

(** {1 Handlers} *)

type handler = Yojson.Basic.t -> Yojson.Basic.t option
(** A [handler] consumes JSON and (optionally) produces a response. *)

(** {1 Building Servers} *)

(**
To build a {!type:socket}, we must provide an {!type:address} and a {!type:port}. The latter are simple integers, but the former require special constructors (found in Lwt or Unix). For ease of use, we provide the {!val:local_address} constructor.
*)

val local_address : address
(** [local_address] is the {!type:address} representing the local host (i.e., the IPv4 loopback address 127.0.0.1). *)

val socket : address -> port -> socket
(** [socket addr port] constructs a network endpoint for communicating at address [addr] and over port [port]. *)

(**
A server is an infinite process that handles incoming JSON objects with a {!type:handler}. Regardless of the behavior of the handler, the server will continue to process JSON until an exception is thrown or the process is terminated.
*)

val server : ?timeout:float -> handler -> socket -> server
(** [server h sock] builds a process that communicates over [sock] by processing JSON objects with [handler]. *)

(** {1 Running Servers} *)

val run : server -> unit
(** [run server] starts the server process. *)