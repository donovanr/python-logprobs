# python-logprobs

It is very common to use log probabilities in statistics, machine learning, and scientific computation, for example to avoid underflowing floats when calculating likelihood functions.  Most of the time, the math is very simple, and doing things "by hand" is fine.  But sometimes the numerics are tricky, and it seems like there should be a standard(ish) way of doing these computations, where the optimization/numerics are behind a nice abstraction layer, and all you have to do is call functions that do what you want.

This module is supposed to abstract away the (simple) math needed to do arithmetic with log probabilities.  For example, If `p` and `q` are probabilities, then `x` and `y` are representations in log space:
```code
x = log(p) and y = log(q)  <==>  p = e^x and q = e^y
```
To multiply two probabilities, `r = p*q`, it's easy to stay in the log space:
```
w = log(r) = log(p*q) = log(e^x * e^y) = log(e^(x + y)) = x + y
```
So the function `multiply_log_weights` assumes you already have `x` and `y`, the log space representations of some probabilities, and returns the log space result of multiplying the probabilities, i.e. adding the log space values. To get back to normal space, just exponentiate using `convert_logweight_to_standard(w)`.  If you don't have `x` to start with, but only `p`, you gan get it using `convert_standard_to_logweight(p)`, which simply takes a log.

Doing addition and subtraction is the hard part, becasue if you do the naive thing:
```
w = log(r) = log(p + q) = log(e^x + e^y)
```
those exponentials are still a problem, and you under/overflow at the same point.  Fortunately, there's an easy fix:
```
log(e^x + e^y) = log(e^x(1 + e^(y-x))) = log(e^x) + log(1 + e^(y-x)) = x + log1p(e^(y-x))
```
where `log1p(x) = log(1+x)` is a function optimized for logs of numbers that are close to one, and we assume, without loss of generality, that `x > y`.  Now the computation only over/underflows if the difference in the magnidude of the numbers is very large, a situation that happens much less often.
