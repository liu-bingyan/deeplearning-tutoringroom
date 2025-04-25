# Generate two random vectors in 1024 dimensions
vector1 = np.random.rand(1024)
vector2 = np.random.rand(1024)

cos_distance = 1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))