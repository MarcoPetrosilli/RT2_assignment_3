import numpy as np
import matplotlib.pyplot as plt

# Coordinate fornite
coords = np.array([
    [-3, 7], [6, 1], [-6, -7], [1, -4], [-4, 8], [7, 2], [0, -3], [8, 3],
    [-7, 4], [-2, 3], [4, -1], [3, -6], [-6, 1], [-8, 3], [-1, -2], [2, -2],
    [6, 8], [-8, -8], [-2, -4], [2, -5], [3, 7], [0, 2], [-4, -6], [8, 6],
    [-3, -1], [7, -8], [1, 4], [-6, -1], [6, -3], [-1, 6], [-5, -8], [8, -2],
    [-8, 5], [5, -5], [-2, -8], [-7, -3], [2, 8], [-4, 5], [6, -6], [-8, -1],
    [1, -8], [3, 7], [-3, 8], [4, -5], [-6, 7], [7, -4], [0, -8], [-1, 7],
    [5, 3], [-7, 3]
])

# Punto di partenza
start = np.array([[0, 1]])

# Costruiamo il percorso completo
path = np.vstack([start, coords])

# Plot
plt.figure(figsize=(8, 8))
plt.plot(path[:, 0], path[:, 1], marker='o', linestyle='-', color='blue')
plt.scatter(start[0, 0], start[0, 1], color='red', s=100, label='Start [0, 1]')

# Annotazione dei punti
for i, (x, y) in enumerate(path):
    plt.text(x + 0.3, y + 0.3, str(i), fontsize=8)

# Impostazioni grafiche
plt.xlim(-9, 9)
plt.ylim(-9, 9)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Percorso attraverso i checkpoint")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Salvataggio in formato EPS
plt.savefig('percorso_checkpoint.eps', format='eps')

# Mostra il grafico
plt.show()
