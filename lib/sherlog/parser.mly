%{
    open Watson

    open Problem.Parameter

    exception DomainError
    (* exception FactError *)
%}

// EOF of course a requirement to know when we've stopped parsing
%token EOF

// all the tokens needed (plus the core term values) to define a pure Datalog program
%token LPARENS
%token RPARENS
%token ARROW
%token PERIOD
%token COMMA

// tokens for the core term values
%token TRUE
%token FALSE
%token <float> FLOAT
%token <int> INTEGER
%token <string> SYMBOL
%token <string> VARIABLE

// question marks are used for traditional queries
%token QMARK

// generative rules need brackets (for parameters)
%token LBRACKET
%token RBRACKET

// at-sign (for context)
%token AT

// and semicolons for separating the generative part from the logical
%token SEMICOLON

// for inference, we have special tokens for denoting the relevant values
%token PARAMETER
%token NAMESPACE
%token EVIDENCE

// and colons for separating arguments and the like
%token COLON
%token IN


%start <Problem.line list> problem

%%

// core logic programming
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

atom : 
    | s = SYMBOL; LPARENS; ts = terms; RPARENS { Atom.Atom (s, ts) } 
    | s = SYMBOL; { Atom.Atom (s, []) }
    ;
atoms : ss = separated_list(COMMA, atom) { ss } ;

clause :
    | fact = atom; PERIOD { Rule.Rule (fact, []) } // fact construction
    | head = atom; ARROW; body = atoms; PERIOD { Rule.Rule (head, body) } // rule construction
    ;

query : atoms = atoms; QMARK { atoms } ;

// generative logic programming
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

intro_clause : 
    | ia = intro_atom; ARROW; body = atoms; PERIOD {
        let rel, args, f, params, context = ia in Problem.simplify_introduction
            ~relation:rel ~arguments:args ~generator:f ~parameters:params ~context:context ~body:body
    } 
    | ia = intro_atom; PERIOD {
        let rel, args, f, params, context = ia in Problem.simplify_introduction
            ~relation:rel ~arguments:args ~generator:f ~parameters:params ~context:context ~body:[]
    }
    ;

// fuzzy facts
// fuzzy_fact : p = term; COLON; a = atom; PERIOD {
//     match a with
//         | Atom.Atom (r, args) -> Problem.simplify_fuzzy_fact ~probability:p ~relation:r ~arguments:args
//         | _ -> raise FactError
// }
// ;

// inference
parameter :
    | PARAMETER; s = SYMBOL; COLON; dom = SYMBOL; PERIOD {
        match dom with
            | "unit" -> Parameter (s, Unit)
            | "positive" | "pos" -> Parameter (s, Positive)
            | "real" ->  Parameter (s, Real)
            | _ -> raise DomainError
    }
    ;

namespace : NAMESPACE; name = SYMBOL; PERIOD { Problem.Namespace.Namespace name } ;

evidence:
    | EVIDENCE; atoms = atoms; PERIOD { Problem.Evidence.Evidence atoms }
    | EVIDENCE; LPARENS; bindings = separated_list(COMMA, SYMBOL); IN; source = SYMBOL; RPARENS; atoms = atoms; PERIOD {
        Problem.Evidence.ParameterizedEvidence (bindings, source, atoms)
    }
    ;


// generating lines to build program from
line :
    // core logic programming
    | clause = clause;              { [`Rule clause] }
    | query = query;                { [`Query query] }
    // generative logic programming
    | intro_clause = intro_clause;  { intro_clause }
    // fuzzy fact
    // | fuzzy_fact = fuzzy_fact       { fuzzy_fact }
    // inference
    | parameter = parameter;        { [`Parameter parameter] }
    | namespace = namespace;        { [`Namespace namespace] }
    | evidence = evidence;          { [`Evidence evidence] }
    ;

// entrypoint - collects lists of lines
problem : cs = list(line); EOF { CCList.flatten cs } ;