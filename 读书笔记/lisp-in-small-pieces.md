## 书籍信息

作者: Christian Queinnec, Ecole Polytechnique
英文翻译者: Kathleen Callaway

## Chapter 1: The Basic of Interpretation

![Pasted image 20220918173232.png](Pasted%20image%2020220918173232.png)

### 1.2 Basic Evaluator

 > 
 > Within a program, we distinguish *free variables* from *bound variables*. A variable
 > is free as long as no binding form (such as lambda, let, and so forth) qualifies it;
 > otherwise, we say that a variable is bound.

 > 
 > The data structure associating
 > variables and values is known as an environment.

### 1.3 Evaluating Atoms

 > 
 > The principal con-
 > ventions of representation are that a variable is represented by a symbol (its name)
 > and that a functional application is represented by a list where the first term of
 > the list represents the function to apply and the other terms represent arguments
 > submitted to that function.

 > 
 > It is important to distinguish the program from its representation (or,
 > the message from its medium, if you will)

## Chapter 2: Lisp, 1, 2, ..., $\omega$

 > 
 > Among all the objects that an evaluator can handle, a function represents a very
 > special case. This basic type has a special creator, lambda, and at least one legal
 > operation: application. We could hardly constrain a type less without stripping
 > away all its utility. Incidentally, this fact-that it has few qualities-makes a
 > function particularly attractive for specifications or encapsulations because it is
 > opaque and thus allows only what it is programmed for. We can, for example, use
 > functions to represent objects that have fields and methods (that is, data members
 > and member functions) as in \[AR88\].

 > 
 > the process of evaluating terms in an application did not distinguish
 > the function from its arguments;

````lisp
(define (evaluate e env)
    (if (atom? e) ...
        (case (car e)
                ((lambda) (make-function (cadr e) (cddr e) env))
                (else (invoke (evaluate (car e) env)
                              (evlis (cdr e) env) )) ) ) )

@ or do this to "eval" a function:
(else (invoke (lookup (car e) env)
              (evlis (cdr e) env)))
````
