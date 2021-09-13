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
%token LBRACE
%token RBRACE

// and semicolons for separating the generative part from the logical
%token SEMICOLON

// for inference, we have special tokens for denoting the relevant values
%token PARAMETER
%token EVIDENCE
%token SQUIGARROW

// and colons for separating arguments and the like
%token COLON
%token DOUBLECOLON
%token BLANK


%start <Line.t list> lines

%%

// core logic programming
value :
    | TRUE; { Term.Boolean true }
    | FALSE; { Term.Boolean false }
    | f = FLOAT; { Term.Float f }
    | i = INTEGER; { Term.Integer i }
    | s = SYMBOL; { Term.Symbol s }
    | x = VARIABLE; { Term.Variable x }
    ;

term :
    | v = value; { v }
    | LBRACKET; RBRACKET; { Term.Unit }
    | v = value; DOUBLECOLON; t = term; { Term.Function ("cons", [v; t]) }
    | LBRACKET; vs = separated_nonempty_list(COMMA, value); RBRACKET; { 
        let cons x y = Term.Function ("cons", [x ; y]) in
        CCList.fold_right cons vs Term.Unit
    }
    | BLANK; { Term.Wildcard }
    | t = delimited(LPARENS, term, RPARENS); { t }
    ;

// relation OR relation(t_1, ..., t_k)
atom : relation = SYMBOL; terms = loption(delimited(LPARENS, separated_nonempty_list(COMMA, term), RPARENS)) { Atom.make relation terms } ;

// <- a_1, ..., a_k OR nothing
rule_body : atoms = loption(preceded(ARROW, separated_nonempty_list(COMMA, atom))) { atoms };

// fact. OR head <- body.
rule : head = atom; body = rule_body; PERIOD { Rule.make head body } ;

// {t_1, ..., t_k}
domain : terms = delimited(LBRACE, separated_nonempty_list(COMMA, term), RBRACE) { terms } ;

// f[t_1, ..., t_k]
function_application : f = SYMBOL; arguments = delimited(LBRACKET, separated_list(COMMA, term), RBRACKET) { (f, arguments) } ;

// GENERATIVE FORM
generation(X) : relation = SYMBOL; LPARENS; terms = separated_list(COMMA, term); SEMICOLON; x = X; RPARENS; body = rule_body; PERIOD; {
    Line.Generation.make relation terms x body
};

// f[args]
introduction : f = function_application; {
    let function_id, arguments = f in `Introduction (function_id, arguments)
};

// domain <~ f[args]
discrete_embedding : domain = domain; SQUIGARROW; f = function_application; {
    let function_id, arguments = f in `DiscreteEmbedding (domain, function_id, arguments)
};

// domain <- f[args]
softmax_embedding : domain = domain; ARROW; f = function_application; {
    let function_id, arguments = f in `SoftmaxEmbedding (domain, function_id, arguments)
};

// inference forms
parameter : PARAMETER; s = SYMBOL; COLON; dom = SYMBOL; PERIOD {
    match dom with
        | "unit" -> Parameter.make s Parameter.Unit
        | "positive" | "pos" -> Parameter.make s Parameter.Positive
        | "real" ->  Parameter.make s Parameter.Real
        | _ -> raise DomainError
};

evidence : EVIDENCE; atoms = separated_nonempty_list(COMMA, atom); PERIOD; { Evidence.make atoms };

// generating lines
line :
    // CORE
    | rule = rule; { [`Rule rule] }
    // INFERENCE
    | parameter = parameter; { [`Parameter parameter] }
    | evidence = evidence; { [`Evidence evidence] }
    // GENERATIVE FORMS
    | introduction = generation(introduction); { Line.Generation.compile introduction }
    | discrete_embedding = generation(discrete_embedding); { Line.Generation.compile discrete_embedding }
    | softmax_embedding = generation(softmax_embedding); { Line.Generation.compile softmax_embedding }
    ;

// entrypoint - collects lists of lines
lines : cs = list(line); EOF { CCList.flatten cs } ;