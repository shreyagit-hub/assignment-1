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
-1 tells NumPy to calculate the missing dimension.
Reshaping usually creates a view, so it does not need to copy the underlying data.
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
and since it returns only the selected elements, the result does not retain the original 2D shape.
'''


## B4 ##
print("\nB4\n")
original_arr = temps[0, 0]

view = temps[0, :3]
view[0] = 999

print("Original array after modifying the view:\n", temps)
print("First temperature is now:", temps[0, 0])

temps[0, 0] = original_arr

copied_slice = temps[0, :3].copy()
copied_slice[0] = 999

print("Modified copy:", copied_slice)
print("Original array remains unchanged:\n", temps)

'''
NumPy slicing often returns a view to avoid copying large arrays, making operations efficient.
Forgetting this can accidentally modify the original dataset.
'''


## B5 ##
print("\nB5\n")
import time

def to_fahrenheit_loop(values):
    flat = values.ravel()
    result = np.empty_like(flat)

    for i in range(flat.size):
        result[i] = flat[i] * 9 / 5 + 32

    return result.reshape(values.shape)


fahrenheit_loop = to_fahrenheit_loop(temps)

print("Fahrenheit using loop:\n", fahrenheit_loop)

fahrenheit_vectorized = temps * 9 / 5 + 32

print("Fahrenheit using vectorization:\n",
      fahrenheit_vectorized)

print("Results agree:",
      np.allclose(fahrenheit_loop, fahrenheit_vectorized,
                  equal_nan=True))

large_temps = np.random.randint(
    14, 36, size=1_000_000
).astype(float)

start = time.perf_counter()
large_f_loop = to_fahrenheit_loop(large_temps)
loop_time = time.perf_counter() - start

start = time.perf_counter()
large_f_vectorized = large_temps * 9 / 5 + 32
vector_time = time.perf_counter() - start

speedup = loop_time / vector_time

print("Loop time:", loop_time, "seconds")
print("Vectorized time:", vector_time, "seconds")
print("Speedup:", speedup, "x faster")

print("Large results agree:",
      np.allclose(large_f_loop, large_f_vectorized))

'''
A Python loop has extra overhead from interpreting each iteration,
indexing individual values, and assigning them one at a time.
Vectorized NumPy operations perform the calculations in optimized
compiled code, reducing Python-level overhead.
'''


## B6 ##
print("\nB6\n")
city_means = np.nanmean(temps, axis=1, keepdims=True)

print("City means:\n", city_means)
print("Mean shape:", city_means.shape)

anomalies = temps - city_means

print("Anomaly grid:\n", anomalies)
print("Anomaly shape:", anomalies.shape)

print("Shape remains (5, 14):",
      anomalies.shape == (5, 14))

daily_correction = np.arange(14)

corrected_temps = temps + daily_correction

print("Daily correction:", daily_correction)
print("Corrected temperatures:\n", corrected_temps)
print("Corrected shape:", corrected_temps.shape)

try:
    invalid_result = temps + np.array([1, 2, 3, 4, 5])
except ValueError as error:
    print("Broadcasting error:", error)

'''
Broadcasting aligns dimensions from the right.
(5, 14) and (14,) are compatible, so the daily correction works.
But (5, 14) and (5,) are incompatible because 14 and 5 differ.
'''


## B7 ##
print("\nB7\n")
quantities = np.array([2, 3, 1])
prices = np.array([120, 90, 250])

elementwise = quantities * prices
print("Element-wise multiplication:", elementwise)

total_cost = quantities @ prices
print("Total grocery cost:", total_cost)

'''
* multiplies corresponding elements and returns an array.
@ computes the dot product here, multiplying and summing
corresponding elements to return one total value.
'''

mean_temps_per_day = np.nanmean(temps, axis=0)
mean_humidity_per_day = np.nanmean(humidity, axis=0)

features = np.column_stack((
    mean_temps_per_day,
    mean_humidity_per_day
))

weights = np.array([0.6, -0.1])

comfort_scores = features @ weights

print("Features shape:", features.shape)
print("Weights shape:", weights.shape)
print("Comfort scores:", comfort_scores)
print("Result shape:", comfort_scores.shape)

'''
Matrix multiplication requires the number of columns
in features (2) to match the number of weights (2).
The result contains one score for each of the 14 days.
'''


### B8 ##
print("\nB8\n")
