from random import Random
import math
import matplotlib.pyplot as plt
import numpy
import numpy as np

N = 100000
r = Random()

x = [-math.log(1-r.random()) for _ in range(N)]
y = [math.sqrt((2*math.e)/math.pi)*math.e**(-xi)*r.random() for xi in x]

out = [xy[0] for xy in zip(x,y) if xy[1] < math.sqrt(2/math.pi)*math.e**((-xy[0]**2)/2)]
print(out)

fig, axs = plt.subplots(1, 4, sharey=False, tight_layout=False, figsize=(10, 3))

axs[0].hist(x, bins=100)
axs[0].set_title("x")

axs[1].hist(y, bins=100)
axs[1].set_title("y")

axs[2].hist(out, bins=100)
axs[2].set_title("out")

fx_x = np.linspace(0, 4, 1000)
fx_y = np.sqrt(2/np.pi)*np.e**(-fx_x**2/2)
axs[3].plot(fx_x, fx_y)
axs[3].set_title("f(x)")

plt.show()