import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("repeatability_log.csv")

plt.figure()
plt.scatter(df.ee_x, df.ee_y, s=3)
plt.axis("equal")
plt.show()
