Unroll-and-jam is an effective loop optimization that not only im-  
proves cache locality and instruction level parallelism (ILP) but  
also benefits other loop optimizations such as scalar replacement.

Unroll-and-jam is a loop transformation that increases the size of an  
inner loop body by unrolling outer loops multiple times followed by  
fusing the copies of inner loops back together

**Jam is also called "Loop fusion".** 

Example:

````python
# Original program
for i in range(0, 100):
    for j in range(0, 100):
        A[i, j] = 0;
        for k in range(0, 100):
            A[i, j] += B[i, k] * C[k, j]


# Unroll the outer loop
for i in range(0, 100, 2):
    for j in range(0, 100):
        A[i, j] = 0;
        for k in range(0, 100):
            A[i, j] += B[i, k] * C[k, j]
    for j in range(0, 100):
        A[i+1, j] = 0;
        for k in range(0, 100):
            A[i+1, j] += B[i, k] * C[k, j]


# Jam the one inner loop
for i in range(0, 100, 2):
    for j in range(0, 100):
        A[i, j] = 0;
        for k in range(0, 100):
            A[i, j] += B[i, k] * C[k, j]
        A[i+1, j] = 0;
        for k in range(0, 100):
            A[i+1, j] += B[i, k] * C[k, j]


# Re-order the independence instruction
# Because they are in two loops, A[i+1, j] = 0 should never affect the instructions in first inner loop.
for i in range(0, 100, 2):
    for j in range(0, 100):
        A[i, j] = 0;
        A[i+1, j] = 0;
        for k in range(0, 100):
            A[i, j] += B[i, k] * C[k, j]
        for k in range(0, 100):
            A[i+1, j] += B[i, k] * C[k, j]


# Jam another inner loop
for i in range(0, 100, 2):
    for j in range(0, 100):
        A[i, j] = 0;
        A[i+1, j] = 0;
        for k in range(0, 100):
            A[i, j] += B[i, k] * C[k, j]
            A[i+1, j] += B[i, k] * C[k, j]
````

## Ref

* [Register Pressure Guided Unroll-and-Jam](https://www.capsl.udel.edu/conferences/open64/2008/Papers/104.pdf)
* [A catalogue of optimizing transformations](https://raw.githubusercontent.com/tpn/pdfs/master/A%20Catalogue%20of%20Optimizing%20Transformations%20(1971-allen-catalog).pdf)
