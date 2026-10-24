import numpy as np
np.random.seed(7)
cities = np.array(["Kathmandu", "Pokhara", "Biratnagar", "Lalitpur", "Birgunj"])
days = np.array([f"D{i:02d}" for i in range(1, 15)])
temps = np.random.randint(14, 36, size=(5, 14)).astype(float) # °C, 5 cities × 14 days
humidity = np.random.randint(40, 95, size=(5, 14)).astype(float) # %
temps[2, 5] = np.nan # sensor outage
temps[4, 11] = np.nan


## B1 ##
print("\nB1\n")
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


## B2 ##
print("\nB2\n")
result_a = temps[1, 2]
print("(a) Pokhara day-3 temperature:", result_a,
      "\nShape:", np.shape(result_a))

result_b = temps[:, -1]
print("(b) Last-day readings:", result_b,
      "\nShape:", result_b.shape)

result_c = temps[2, :]
print("(c) Biratnagar full fortnight:", result_c,
      "\nShape:", result_c.shape)

result_d = temps[:3, 3:8]
print("(d) First three cities across days 4-8:\n", result_d,
      "\nShape:", result_d.shape)

result_e = temps[3, ::2]
print("(e) Lalitpur every other day:", result_e,
      "\nShape:", result_e.shape)

result_f = temps[0, ::-1]
print("(f) Kathmandu fortnight reversed:", result_f,
      "\nShape:", result_f.shape)


## B3 ##
print("\nB3\n")
mask = temps > 30

print("(a) Boolean mask:\n", mask)

counts = np.sum(mask, axis=1)

print("(b) Readings above 30°C per city:")
print(np.column_stack((cities, counts)))

kathmandu_mask = mask[0]
hot_days = days[kathmandu_mask]

print("(c) Kathmandu days above 30°C:", hot_days)

'''
A Boolean mask selects elements where its value is True 
and since it returns only the selected elements, the result does not retain the original 2D shape
'''


## B4 ##
print("\nB4\n")