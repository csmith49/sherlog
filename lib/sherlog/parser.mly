%{
    open Watson.Syntax
    open Watson.Semantics

    open Problem.Parameter

    exception DomainError
%}

// all the tokens needed (plus the core term values) to define a pure Datalog program
%token LPARENS
%token RPARENS
%token ARROW
%token PERIOD
%token COMMA

// learning annotations are marked with a bang and delimited with colons, etc.
%token COLON
%token BANG
// with arguments provided via brackets
%token LBRACKET
%token RBRACKET

%token AT

// question marks are used for traditional queries
%token QMARK

%token SEMICOLON

// EOF of course a requirement to know when we've stopped parsing
%token EOF

// tokens for the core term values
%token TRUE
%token FALSE
%token <float> FLOAT
%token <int> INTEGER
%token <string> SYMBOL
%token <string> VARIABLE

%start <Problem.Line.t list> problem

%%

// Core Abductive Logic Language (CALL)
term :
    | TRUE { Term.Boolean true }
    | FALSE { Term.Boolean false }
    | f = FLOAT { Term.Float f }
    | i = INTEGER { Term.Integer i} 
    | s = SYMBOL { Term.Constant s }
    | x = VARIABLE { Term.Variable x }
    | f = SYMBOL; LPARENS; args = separated_list(COMMA, term); RPARENS { Term.Function (f, args) }
    ;
terms : ts = separated_list(COMMA, term) { ts } ;

atom : s = SYMBOL; LPARENS; ts = terms; RPARENS { Atom.Atom (s, ts) } ;
atoms : ss = separated_list(COMMA, atom) { ss } ;

clause :
    | fact = atom; PERIOD { Rule.Rule (fact, []) } // fact construction
    | head = atom; ARROW; body = atoms; PERIOD { Rule.Rule (head, body) } // rule construction
    ;

parameter :
    | BANG; s = SYMBOL; COLON; dom = SYMBOL; PERIOD {
        match dom with
            | "unit" -> Parameter (s, Unit)
            | "positive" | "pos" -> Parameter (s, Positive)
            | "real" ->  Parameter (s, Real)
            | _ -> raise DomainError
    }
    // | BANG; s = SYMBOL; COLON; dom = SYMBOL; LBRACKET; i = INTEGER; RBRACKET; PERIOD {
    //     match dom with
    //         | "categorical" | "cat" -> Parameter (s, Categorical i)
    //         | _ -> raise DomainError
    // }
    ;

query : atoms = atoms; QMARK { atoms } ;

location : f = SYMBOL; AT; m = SYMBOL { Problem.Location.Location (f, m) } ;
external_function : BANG; f = SYMBOL; ARROW; loc = location; PERIOD { Problem.Function.Function (f, loc) } ;

evidence : BANG; atoms = atoms; PERIOD { atoms } ;

intro_term : f = SYMBOL; LBRACKET; params = terms; RBRACKET { (f, params) } ;
intro_atom : 
    | rel = SYMBOL; LPARENS; args = terms; SEMICOLON; it = intro_term; RPARENS {
        let f, params = it in let context = (Term.Constant rel) :: (args @ params) in
        (rel, args, f, params, context)
    }
    | rel = SYMBOL; LPARENS; args = terms; SEMICOLON; it = intro_term; AT; context = terms; RPARENS {
        let f, params = it in (rel, args, f, params, context)
    }
    ;

intro_clause : ia = intro_atom; ARROW; body = atoms; PERIOD {
    let rel, args, f, params, context = ia in Problem.simplify_introduction rel args f params context body
} ;

line :
    | clause = clause;              { [Problem.Line.Rule clause] }
    | parameter = parameter;        { [Problem.Line.Parameter parameter] }
    | query = query;                { [Problem.Line.Query query] }
    | ext_f = external_function;    { [Problem.Line.Function ext_f] }
    | evidence = evidence;          { [Problem.Line.Evidence evidence] }
    | intro_clause = intro_clause;  { CCList.map Problem.Line.lift_rule intro_clause }
    // | tag = tag; { `T tag }
    // | delta_clause = delta_clause { delta_clause |> Delta.simplify }
    ;

// entrypoint - collects lists of lines
problem : cs = list(line); EOF { CCList.flatten cs } ;