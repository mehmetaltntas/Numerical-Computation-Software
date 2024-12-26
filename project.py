import numpy as np
import matplotlib.pyplot as plt

# 1. Define a 4x4 matrix A, vector b, and compute x and A_inv
A = np.matrix([
    [4,  1,  2,  3],
    [0,  5,  1,  2],
    [1,  0,  3,  1],
    [2,  1,  1,  4]
])

b = np.matrix([
    [10],
    [12],
    [8],
    [14]
])

# Compute the inverse of A
A_inv = np.linalg.inv(A)

# Compute solution vector x
x = A_inv * b

# 2. Part 1: Computing Norms
print("\n--- Computing Norms ---")
# Norms of A
norm_A_1 = np.linalg.norm(A, 1)
norm_A_2 = np.linalg.norm(A, 2)
norm_A_inf = np.linalg.norm(A, np.inf)
norm_A_fro = np.linalg.norm(A, 'fro')
print(f"1-Norm of A: {norm_A_1}") 
print(f"2-Norm of A (Spectral Norm): {norm_A_2:.2f}")  
print(f"Infinity Norm of A: {norm_A_inf}") 
print(f"Frobenius Norm of A: {norm_A_fro:.2f}")  

# Norms of A_inv
norm_Ainv_1 = np.linalg.norm(A_inv, 1)
norm_Ainv_2 = np.linalg.norm(A_inv, 2)
norm_Ainv_inf = np.linalg.norm(A_inv, np.inf)
norm_Ainv_fro = np.linalg.norm(A_inv, 'fro')
print(f"\n1-Norm of A_inv: {norm_Ainv_1}")  
print(f"2-Norm of A_inv (Spectral Norm): {norm_Ainv_2}") 
print(f"Infinity Norm of A_inv: {norm_Ainv_inf}")  
print(f"Frobenius Norm of A_inv: {norm_Ainv_fro}") 

# Norms of b
norm_b_1 = np.linalg.norm(b, 1)
norm_b_2 = np.linalg.norm(b, 2)
norm_b_inf = np.linalg.norm(b, np.inf)
print(f"\n1-Norm of b: {norm_b_1}")  
print(f"2-Norm of b: {norm_b_2:.0f}")  
print(f"Infinity Norm of b: {norm_b_inf}")  

# Norms of x
norm_x_1 = np.linalg.norm(x, 1)
norm_x_2 = np.linalg.norm(x, 2)
norm_x_inf = np.linalg.norm(x, np.inf)
print(f"\n1-Norm of x: {norm_x_1}")  
print(f"2-Norm of x: {norm_x_2:.2f}")  
print(f"Infinity Norm of x: {norm_x_inf}")  

# 3. Part 2: Perturbation in Vector b
print("\n--- Perturbation in Vector b ---")
Delta_b = np.matrix([
    [-3],
    [3],
    [2],
    [1]
])

print("Delta b:")
print(Delta_b)

# Compute b + Delta b
b_new = b + Delta_b
print("\nb + Delta b:")
print(b_new)

# Compute Delta x = A_inv * Delta b
Delta_x = A_inv * Delta_b
print("\nDelta x:")
print(Delta_x)

# Compute new solution x + Delta x
x_new = x + Delta_x
print("\nx + Delta x:")
print(x_new)

# 4. Part 3: Relative Errors and Condition Numbers
print("\n--- Relative Errors and Condition Numbers ---")
# Relative errors for Delta b
norm_Delta_b_1 = np.linalg.norm(Delta_b, 1)
relative_change_b_1 = norm_Delta_b_1 / norm_b_1 

norm_Delta_b_2 = np.linalg.norm(Delta_b, 2)
relative_change_b_2 = norm_Delta_b_2 / norm_b_2  

norm_Delta_b_inf = np.linalg.norm(Delta_b, np.inf)
relative_change_b_inf = norm_Delta_b_inf / norm_b_inf  

print(f"||Delta b||_1 / ||b||_1 = {relative_change_b_1:.3f}")  
print(f"||Delta b||_2 / ||b||_2 = {relative_change_b_2:.3f}")  
print(f"||Delta b||_inf / ||b||_inf = {relative_change_b_inf:.3f}")  
# Relative errors for Delta x
norm_Delta_x_1 = np.linalg.norm(Delta_x, 1)
relative_change_x_1 = norm_Delta_x_1 / norm_x_1 
norm_Delta_x_2 = np.linalg.norm(Delta_x, 2)
relative_change_x_2 = norm_Delta_x_2 / norm_x_2  

norm_Delta_x_inf = np.linalg.norm(Delta_x, np.inf)
relative_change_x_inf = norm_Delta_x_inf / norm_x_inf  

print(f"||Delta x||_1 / ||x||_1 = {relative_change_x_1:.3f}")  
print(f"||Delta x||_2 / ||x||_2 = {relative_change_x_2:.3f}") 
print(f"||Delta x||_inf / ||x||_inf = {relative_change_x_inf:.3f}") 

# Compute condition numbers
c1 = norm_A_1 * norm_Ainv_1  
c2 = norm_A_2 * norm_Ainv_2 
c_inf = norm_A_inf * norm_Ainv_inf 
print(f"\nc1 = ||A||_1 * ||A^-1||_1 = {c1:.1f}")  
print(f"c2 = ||A||_2 * ||A^-1||_2 = {c2:.2f}") 
print(f"c_inf = ||A||_inf * ||A^-1||_inf = {c_inf:.1f}")  

# 5. Part 4: Verifying Inequalities for Delta b
print("\n--- Verifying Inequalities for Delta b ---")
# For 1-Norm
lhs_1 = relative_change_x_1 
rhs_1 = c1 * relative_change_b_1  
print(f"||Delta x||_1 / ||x||_1 = {lhs_1} <= {c1} * {relative_change_b_1} = {rhs_1} --> {lhs_1 <= rhs_1}")

# For 2-Norm
lhs_2 = relative_change_x_2  
rhs_2 = c2 * relative_change_b_2 
print(f"||Delta x||_2 / ||x||_2 = {lhs_2} <= {c2} * {relative_change_b_2} = {rhs_2} --> {lhs_2 <= rhs_2}")

# For Infinity Norm
lhs_inf = relative_change_x_inf 
rhs_inf = c_inf * relative_change_b_inf  
print(f"||Delta x||_inf / ||x||_inf = {lhs_inf} <= {c_inf} * {relative_change_b_inf} = {rhs_inf} --> {lhs_inf <= rhs_inf}")


# Define perturbation matrix Delta A
Delta_A = np.matrix([
    [-1, 1, -1, 0],
    [-1, 0, 1, 0],
    [1, -1, 1, 0],
    [0, 1, -1, 0]
])

print("\n--- Perturbation ΔA and compute the new solution x+Δx  ---")
print("\nDelta A:")
print(Delta_A)


# Compute A + Delta A
A_new = A + Delta_A
print("\nA + Delta A:")
print(A_new)

# Compute the inverse of (A + Delta A)
try:
    A_new_inv = np.linalg.inv(A_new)
    print("\nInverse of (A + Delta A):")
    print(A_new_inv)
except np.linalg.LinAlgError:
    print("\n(A + Delta A) is singular and cannot be inverted.")
    A_new_inv = None

if A_new_inv is not None:
    # Compute Delta x = A_new_inv * b
    Delta_x_part5 = A_new_inv * b
    print("\nDelta x (using (A + Delta A)^-1 * b):")
    print(Delta_x_part5)

    # Compute new solution x + Delta x
    x_new_part5 = x + Delta_x_part5
    print("\nx + Delta x:")
    print(x_new_part5)

print("\n--- Computing Relative Errors  ---")

if A_new_inv is not None:
    # Compute norms of Delta A and A
    norm_Delta_A_1 = np.linalg.norm(Delta_A, 1)
    norm_A_1_original = np.linalg.norm(A, 1)
    relative_change_A_1 = norm_Delta_A_1 / norm_A_1_original  

    norm_Delta_A_2 = np.linalg.norm(Delta_A, 2)
    norm_A_2_original = np.linalg.norm(A, 2)
    relative_change_A_2 = norm_Delta_A_2 / norm_A_2_original  

    norm_Delta_A_inf = np.linalg.norm(Delta_A, np.inf)
    norm_A_inf_original = np.linalg.norm(A, np.inf)
    relative_change_A_inf = norm_Delta_A_inf / norm_A_inf_original  

    print(f"\n||Delta A||_1 / ||A||_1 = {relative_change_A_1:.3f}")  
    print(f"||Delta A||_2 / ||A||_2 = {relative_change_A_2:.3f}") 
    print(f"||Delta A||_inf / ||A||_inf = {relative_change_A_inf:.3f}")  

    # Compute norms of Delta x and x + Delta x
    norm_Delta_x_part5_1 = np.linalg.norm(Delta_x_part5, 1)
    norm_x_new_part5_1 = np.linalg.norm(x_new_part5, 1)
    relative_change_x_part5_1 = norm_Delta_x_part5_1 / norm_x_new_part5_1 

    norm_Delta_x_part5_2 = np.linalg.norm(Delta_x_part5, 2)
    norm_x_new_part5_2 = np.linalg.norm(x_new_part5, 2)
    relative_change_x_part5_2 = norm_Delta_x_part5_2 / norm_x_new_part5_2 

    norm_Delta_x_part5_inf = np.linalg.norm(Delta_x_part5, np.inf)
    norm_x_new_part5_inf = np.linalg.norm(x_new_part5, np.inf)
    relative_change_x_part5_inf = norm_Delta_x_part5_inf / norm_x_new_part5_inf  

    print(f"||Delta x||_1 / ||x + Delta x||_1 = {relative_change_x_part5_1:.2f}")  
    print(f"||Delta x||_2 / ||x + Delta x||_2 = {relative_change_x_part5_2:.2f}")  
    print(f"||Delta x||_inf / ||x + Delta x||_inf = {relative_change_x_part5_inf:.2f}")  

print("\n--- Verifying the Inequality for Each Norm  ---")

if A_new_inv is not None:
    # For 1-Norm
    lhs_1_part5 = relative_change_x_part5_1  
    rhs_1_part5 = c1 * relative_change_A_1  
    print(f"\n||Delta x||_1 / ||x + Delta x||_1 = {lhs_1_part5} <= {c1} * {relative_change_A_1} = {rhs_1_part5} --> {lhs_1_part5 <= rhs_1_part5}")

    # For 2-Norm
    lhs_2_part5 = relative_change_x_part5_2 
    rhs_2_part5 = c2 * relative_change_A_2  
    print(f"||Delta x||_2 / ||x + Delta x||_2 = {lhs_2_part5} <= {c2} * {relative_change_A_2} = {rhs_2_part5:.2f} --> {lhs_2_part5 <= rhs_2_part5}")

    # For Infinity Norm
    lhs_inf_part5 = relative_change_x_part5_inf  
    rhs_inf_part5 = c_inf * relative_change_A_inf 
    print(f"||Delta x||_inf / ||x + Delta x||_inf = {lhs_inf_part5} <= {c_inf} * {relative_change_A_inf} = {rhs_inf_part5} --> {lhs_inf_part5 <= rhs_inf_part5}")


