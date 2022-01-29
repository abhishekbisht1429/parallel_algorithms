### Problem
Given a full rank square matrix of dimension n * n. Write a program to find
the solution to the set of linear equations Ax = b using the Gaussian
elimination algorithm followed by back substitution.

**Solution**

- There are three main steps involved in solution of linear equations using
  Gaussian Elimination.
    1. Normalization
    2. Elimination
    3. Back Substitution

- For each row of the augmented matrix normalization and elimination are
  carried out in a sequence. The back substitution is done at the end after
  the normalization and elimination has been done for every row.

- For normalization, of row i, the element [i, i] is taken as pivot and
  every element of the row is divided by it.

- The elimination takes place for every row, say j, after i. The first
  element of the jth row is taken as the scaling factor and the new elements
  of the jth row are computed using the following formula: [i', j] = [i', j]
  \- scale * [i, j], where i' is from i+1 to n-1.

- After completing the above process we are left with an upper triangular
  matrix. The solution for the equations is obtained after applying back
  substitution on it.

- We can parallelize only the elimination part in the gaussian elimination.
  This is done by associating with each rank/process a row that is to be
  eliminated. Let i be the current row for which normalization is being carried
  out and n be the total number of processes, then i mod n gives the rank
  to which the row has been assigned. 
