import numpy as np
import matplotlib.pyplot as plt

loss1 = np.load('respre.npy')
loss2 = np.load('dilation.npy')

plt.imshow(loss2-loss1, cmap='RdYlGn', vmin=-0.6, vmax=0.6)
plt.title('Dilation - ResPre')
plt.colorbar()
plt.savefig('img/dilation_respre.png')
