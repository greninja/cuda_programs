#ifndef SOLVE_H
#define SOLVE_H

// Q (Mxd), K (Nxd), V (Nxd), output are host pointers
// M = sequence length of queries
// N = sequence length of keys/values
// d = embedding dimension
void solve(const float *Q, const float *K, const float *V, float *output, int M, int N, int d);

#endif // SOLVE_H