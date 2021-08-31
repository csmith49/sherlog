# Interface

This module is `sherlog.interface`, and captures the communication with the external OCaml `sherlog-server` that handles the parsing, resolution, and pipeline construction.

## Organization

`server.py` handles the server process, including starting, stopping, and shutdown.

`socket.py` controls the actual communication, including sending and receiving JSON messags from `sherlog-server`.

`__init__.py` exposes minimal control over servers and sockets, and also provides utility functions that construct common messages and destruct their responses.