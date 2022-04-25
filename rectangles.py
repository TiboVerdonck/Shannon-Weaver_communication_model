import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
fig, ax = plt.subplots()
ax.plot([1,5,2],[2,3,4],color="cyan")
ax.add_patch(Rectangle((2, 2), 1, 3,color="yellow"))
plt.xlabel("X-AXIS")
plt.ylabel("Y-AXIS")
plt.title("PLOT-1")
plt.show()