import numpy as np
np.random.seed(7)
cities = np.array(["Kathmandu", "Pokhara", "Biratnagar", "Lalitpur", "Birgunj"])
days = np.array([f"D{i:02d}" for i in range(1, 15)])
temps = np.random.randint(14, 36, size=(5, 14)).astype(float) # °C, 5 cities × 14 days
humidity = np.random.randint(40, 95, size=(5, 14)).astype(float) # %
temps[2, 5] = np.nan # sensor outage
temps[4, 11] = np.nan


## B1 ##
print("Shape:", temps.shape)
print("Dimensions:", temps.ndim)
print("Data type:", temps.dtype)
print("Total elements:", temps.size)

arr_B1 = np.arange(70).reshape(5, 14)

arr_B2 = np.arange(70).reshape(5, -1)

print("Explicit reshape:\n", arr_B1)
print("Reshape using -1:\n", arr_B2)

print("Are they identical?",
      np.array_equal(arr_B1, arr_B2))

flat = arr_B1.flatten()

print("Flattened array:", flat)
print("Flattened shape:", flat.shape)

'''
-1 tells NumPy to calculate the missing dimension
Reshaping usually creates a view, so it does not need to copy the underlying data
'''