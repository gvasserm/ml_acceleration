import numpy as np

def create_matrix(m, n):
  '''Creates a random matrix of shape mxn for testing.'''  
  M = np.random.rand(m, n)  # Generate random matrix of size m x n
  print('The original matrix M:\n', M)  
  print('M shape:', M.shape)  # Print the shape for reference
  return M

def decompose_matrix(M):
  '''Decomposes the matrix M using Singular Value Decomposition (SVD).'''
  U, S, V = np.linalg.svd(M)  # Perform SVD 
  print('U shape:', U.shape) 
  print('S shape:', S.shape)
  print('V shape:', V.shape)

  # Adjust matrix shapes to ensure U (m x n), S (n x n), V (n x n):
  U = U[:, :S.shape[0]]  # Limit U to the number of singular values 
  S = np.diag(S)  # Convert S (a vector) into a diagonal matrix
  print('Adjusted U shape:', U.shape)
  print('Adjusted S shape:', S.shape)
  print('Adjusted V shape:', V.shape)

  return U, S, V

def reconstruct_matrix(U, S, V, r):
  '''Reconstructs the matrix using the SVD components with rank r.''' 
  print('\nReconstructing matrix with rank=', r)
  R = U[:, :r] @ S[:r, :r] @ V[:r, :]  # Use rank-reduced components for approximation 
  print(R)
  print('R shape:', R.shape)  # Output shape of the approximated matrix
  return R

def compare_matrices(M, R):
  '''Calculates the difference and relative difference between the original and 
      reconstructed matrix.'''
  D = M - R  # Element-wise difference
  M_norm = np.linalg.norm(M, 'fro')  # Frobenius norm of original matrix
  D_norm = np.linalg.norm(D, 'fro')  # Frobenius norm of difference matrix
  print('Diff:\n', np.array_str(D, precision=2, suppress_small=True))  # Print differences
  return D_norm / M_norm  # Calculate the relative difference ratio

# Test the functions with sample values:
m = 10
n = 5
M = create_matrix(m, n)  # Create the matrix

U, S, V = decompose_matrix(M)  # Decompose with SVD 

R1 = reconstruct_matrix(U, S, V, n)  # Full reconstruction (using all singular values)
diff_ratio1 = compare_matrices(M, R1)
print(f'Diff ratio 1: {diff_ratio1:.2f}')

R2 = reconstruct_matrix(U, S, V, n - 1)  # Reduced-rank reconstruction
diff_ratio2 = compare_matrices(M, R2)
print(f'Diff ratio 2: {diff_ratio2:.2f}')