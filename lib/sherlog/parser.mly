%{

%}

%token LPARENS
%token RPARENS
%token LBRACKET
%token RBRACKET
%token SEMICOLON
%token MID
%token ARROW
%token PERIOD
%token COMMA
%token QMARK
%token PLUS
%token BANG
%token EOF

%token TRUE
%token FALSE
%token <float> FLOAT
%token <int> INTEGER
%token <string> SYMBOL
%token <string> VARIABLE

%start <CALL.line list> program

%%

// Core Abductive Logic Language (CALL)
term :
    | TRUE { `Boolean true }
    | FALSE { `Boolean false }
    | f = FLOAT { `Float f }
    | i = INTEGER { `Integer i} 
    | s = SYMBOL { `Constant s }
    | x = VARIABLE { `Variable x }
    | f = SYMBOL; LPARENS; args = separated_list(COMMA, term); RPARENS { `Function (f, args) }
    ;
terms : ts = separated_list(COMMA, term) { ts } ;

atom : s = SYMBOL; LPARENS; ts = terms; RPARENS { `Atom (s, ts) } ;
atoms : ss = separated_list(COMMA, atom) { ss } ;

clause :
    | fact = atom; PERIOD { `Clause (`True, fact, []) } // fact construction
    | head = atom; ARROW; body = atoms; PERIOD { `Clause (`True, head, body) } // rule construction
    ;

// basic CALL.tags
tag :
    | atoms = atoms; QMARK { `Query atoms }
    | PLUS; atoms = atoms; PERIOD { `Evidence atoms }
    | BANG; s = SYMBOL; PERIOD { `Parameter s }
    ;

// for Layer.Delta
delta_distribution :
    | d = SYMBOL; LBRACKET; args = terms; RBRACKET { `DeltaDistribution (d, args, []) }
    | d = SYMBOL; LBRACKET; args = terms; MID; cs = terms; RBRACKET { `DeltaDistribution (d, args, cs) }
    ;

delta_atom :
    | s = SYMBOL; LPARENS; d = delta_distribution; RPARENS { `DeltaAtom (s, [], d) }
    | s = SYMBOL; LPARENS; ts = terms; SEMICOLON; d = delta_distribution; RPARENS { `DeltaAtom (s, ts, d) }
    ;

delta_clause :
    | fact = delta_atom; PERIOD { `DeltaClause (fact, []) }
    | head = delta_atom; ARROW; body = atoms; PERIOD { `DeltaClause (head, body) }
    ;

// combines parsing of CALL and layers
line :
    | clause = clause; { `C clause }
    | tag = tag; { `T tag }
    | delta_clause = delta_clause { delta_clause |> Delta.simplify }
    ;

// entrypoint - collects lists of lines
program : cs = list(line); EOF { cs } ;