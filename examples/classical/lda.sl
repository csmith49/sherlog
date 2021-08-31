# K is the number of topics, V the number of words in the vocabulary

!parameter alpha : positive[K].
!parameter beta : positive[V].

topics(D; dirichlet[alpha]) <- document(D).
words(T, dirichlet[beta]) <- topic(T).

word(D, I; multinomial[W]) <- words(T, W), topic(D, I, T).
topic(D, I; multinomial[T]) <- topics(D, T).