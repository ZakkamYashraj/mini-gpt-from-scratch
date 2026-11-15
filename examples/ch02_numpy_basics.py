"""Learning Guide, Chapter 2: NumPy basics.

Run it with:  python examples/ch02_numpy_basics.py
"""

import numpy as np

# --- 1. Arrays and shapes ---
v = np.array([1.0, 2.0, 3.0])  # a vector: 1 dimension
M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # a matrix: 2 rows, 3 columns
print("1. Arrays and shapes")
print("v =", v, " shape:", v.shape, " ndim:", v.ndim)
print("M =\n", M)
print("M.shape:", M.shape, " M.ndim:", M.ndim, " M.size:", M.size)
print("M[1, 2] =", M[1, 2], "  (row 1, column 2; counting starts at 0)")
print("M[0] =", M[0], "  (the whole first row)")
print("M[:, 1] =", M[:, 1], "  (the whole second column)")

# --- 2. Element-wise arithmetic ---
print("\n2. Element-wise arithmetic")
print("v + 10 =", v + 10)
print("v * v  =", v * v)
print("np.exp(v) =", np.round(np.exp(v), 2))
print("M.sum() =", M.sum(), "  M.mean() =", M.mean())
print("M.sum(axis=0) =", M.sum(axis=0), "  (add DOWN the rows: one total per column)")
print("M.sum(axis=1) =", M.sum(axis=1), "  (add ACROSS the columns: one total per row)")

# --- 3. Reshaping ---
print("\n3. Reshaping")
r = np.arange(6)  # [0 1 2 3 4 5]
print("r =", r, " shape:", r.shape)
print("r.reshape(2, 3) =\n", r.reshape(2, 3))
print("r.reshape(3, -1) =\n", r.reshape(3, -1), " (-1 means: work this size out for me)")
print("M.T (transpose) =\n", M.T)

# --- 4. Broadcasting ---
print("\n4. Broadcasting")
row = np.array([10.0, 20.0, 30.0])  # shape (3,)
col = np.array([[100.0], [200.0]])  # shape (2, 1)
print("M + row  ->  shape", (M + row).shape)
print(M + row)
print("M + col  ->  shape", (M + col).shape)
print(M + col)
print("row + col ->  shape", (row + col).shape)
print(row + col)

# --- 5. Matrix multiplication ---
print("\n5. Matrix multiplication")
A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # shape (3, 2)
B = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]])  # shape (2, 4)
C = A @ B  # shape (3, 4): the inner sizes (2 and 2) must match
print("A.shape", A.shape, "@ B.shape", B.shape, "-> C.shape", C.shape)
print(C)
print("C[2, 3] = row 2 of A . column 3 of B =", A[2], ".", B[:, 3], "=", A[2] @ B[:, 3])
x = np.array([1.0, -1.0])
print("A @ x =", A @ x, "  (a matrix times a vector gives a vector)")

# --- 6. Softmax: turning any numbers into probabilities ---
print("\n6. Softmax")
scores = np.array([2.0, 1.0, 0.1])
exps = np.exp(scores)  # make everything positive
probs = exps / exps.sum()  # divide by the total so they sum to 1
print("scores =", scores)
print("exp(scores) =", np.round(exps, 3))
print("probabilities =", np.round(probs, 3), " sum =", round(float(probs.sum()), 6))
