import numpy as np
np.random.seed(7)
cities = np.array(["Kathmandu", "Pokhara", "Biratnagar", "Lalitpur", "Birgunj"])
days = np.array([f"D{i:02d}" for i in range(1, 15)])
temps = np.random.randint(14, 36, size=(5, 14)).astype(float) # °C, 5 cities × 14 days
humidity = np.random.randint(40, 95, size=(5, 14)).astype(float) # %
temps[2, 5] = np.nan # sensor outage
temps[4, 11] = np.nan