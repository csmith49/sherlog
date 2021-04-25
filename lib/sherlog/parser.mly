%{
    open Watson
    exception DomainError
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

// generative rules need brackets (for parameters)
%token LBRACKET
%token RBRACKET

// at-sign (for context)
%token AT

// and semicolons for separating the generative part from the logical
%token SEMICOLON

// for inference, we have special tokens for denoting the relevant values
%token PARAMETER
%token EVIDENCE

// ontologies explicitly labeled too
%token DEPENDENCY
%token CONSTRAINT

// and colons for separating arguments and the like
%token COLON
%token DOUBLECOLON
%token OR
%token BLANK


%start <Line.t list> lines

%%

// core logic programming
value :
    | TRUE { Term.Boolean true }
    | FALSE { Term.Boolean false }
    | f = FLOAT { Term.Float f }
    | i = INTEGER { Term.Integer i }
    | s = SYMBOL { Term.Symbol s }
    | x = VARIABLE { Term.Variable x }
    ;

term :
    | v = value { v }
    | LBRACKET; RBRACKET { Term.Unit }
    | v = value; DOUBLECOLON; t = term; { Term.Function ("cons", [v; t]) }
    | LBRACKET; vs = separated_list(COMMA, value); RBRACKET { 
        let cons x y = Term.Function ("cons", [x ; y]) in
        CCList.fold_right cons vs Term.Unit
    }
    | BLANK { Term.Wildcard }
    | LPARENS; t = term; RPARENS { t }
    ;
terms : ts = separated_list(COMMA, term) { ts } ;

atom : 
    | s = SYMBOL; LPARENS; ts = terms; RPARENS { Atom.make s ts } 
    | s = SYMBOL; { Atom.make s [] }
    ;
atoms : ss = separated_list(COMMA, atom) { ss } ;

clause :
    | fact = atom; PERIOD { Rule.make fact [] } // fact construction
    | head = atom; ARROW; body = atoms; PERIOD { Rule.make head body } // rule construction
    ;

// generative logic programming
intro_term : f = SYMBOL; LBRACKET; params = terms; RBRACKET { (f, params) } ;
intro_atom : 
    | rel = SYMBOL; LPARENS; args = terms; SEMICOLON; it = intro_term; RPARENS {
        let rel_term = (Term.Symbol rel) in
        let f, params = it in let context = rel_term :: (args @ params) in
        (rel, args, f, params, context)
    }
    | rel = SYMBOL; LPARENS; args = terms; SEMICOLON; it = intro_term; AT; context = terms; RPARENS {
        let f, params = it in (rel, args, f, params, context)
    }
    ;

intro_clause : 
    | ia = intro_atom; ARROW; body = atoms; PERIOD {
        let rel, args, f, params, context = ia in Line.encode_intro
            ~relation:rel ~arguments:args ~guard:f ~parameters:params ~context:context ~body:body
    } 
    | ia = intro_atom; PERIOD {
        let rel, args, f, params, context = ia in Line.encode_intro
            ~relation:rel ~arguments:args ~guard:f ~parameters:params ~context:context ~body:[]
    }
    ;

fuzzy_clause :
    | w = term; DOUBLECOLON; head = atom; ARROW; body = atoms; PERIOD {
        Line.encode_fuzzy ~head:head ~body:body ~weight:w
    }
    | w = term; DOUBLECOLON; head = atom; PERIOD {
        Line.encode_fuzzy ~head:head ~body:[] ~weight:w
    }
    ;

// inference
parameter :
    | PARAMETER; s = SYMBOL; COLON; dom = SYMBOL; PERIOD {
        match dom with
            | "unit" -> Parameter.make s Parameter.Unit
            | "positive" | "pos" -> Parameter.make s Parameter.Positive
            | "real" ->  Parameter.make s Parameter.Real
            | _ -> raise DomainError
    }
    ;

evidence :
    | EVIDENCE; atoms = atoms; PERIOD { Evidence.make atoms }
    ;

ontology_dependency :
    | DEPENDENCY; head = separated_list(OR, atom); ARROW; body = atoms; PERIOD { 
        head |> CCList.map (fun hd -> `Dependency (Rule.make hd body))
    }
    ;

ontology_constraint :
    | CONSTRAINT; atoms = atoms; PERIOD { atoms }
    ;

// generating lines to build program from
line :
    // core logic programming
    | clause = clause;              { [`Rule clause] }
    // generative logic programming
    | intro_clause = intro_clause;  { intro_clause }
    // ala problog
    | fuzzy_clause = fuzzy_clause;  { fuzzy_clause }
    // inference
    | parameter = parameter;        { [`Parameter parameter] }
    | evidence = evidence;          { [`Evidence evidence] }
    | d = ontology_dependency;      { d }
    | c = ontology_constraint;      { [`Constraint c] }
    ;

// entrypoint - collects lists of lines
lines : cs = list(line); EOF { CCList.flatten cs } ;