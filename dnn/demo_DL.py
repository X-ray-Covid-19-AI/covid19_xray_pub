import matplotlib.pyplot as plt
import numpy as np

jitter = 0.05

plt.figure()

plt.clf()
D = np.random.rand(2,60)
jit = np.random.rand(D.shape[0], D.shape[1])*jitter
Dj = D + jit
i = np.where(np.linalg.norm(D - [[1], [0.5]], axis=0) < 0.4)[0]
ig = list(set(range(30)).difference(set(i)))
plt.plot(Dj[0,i], Dj[1,i], '.m', markersize=20, markeredgewidth=5)
plt.plot(Dj[0,ig], Dj[1,ig], '.c', markersize=20, markeredgewidth=5)



plt.figure()

plt.clf()
D = np.random.rand(2,60)
jit = np.random.rand(D.shape[0], D.shape[1])*jitter
Dj = D + jit
i = np.where(np.linalg.norm(D - [[1], [0.5]], axis=0) < 0.4)[0]
ig = list(set(range(60)).difference(set(i)))
plt.plot(Dj[0,i], Dj[1,i], '.m', markersize=20, markeredgewidth=5)
plt.plot(Dj[0,ig], Dj[1,ig], '.c', markersize=20, markeredgewidth=5)
plt.axis('equal')


for ind, dj in enumerate(Dj.T):
    for a in [[0.1 * r[0], 0.1 * r[1]] for r in np.random.rand(8,2)*2-1]:
    # for a in [[0.05 * np.sin(th), 0.05 * np.cos(th)] for th in np.arange(0, 2 * np.pi, np.pi / 4)]:
        if ind in i:
            color = 'm'
        else:
            color = 'c'

        dja = dj + a
        plt.plot(dja[0], dja[1], '.' + color, markersize=10, markeredgewidth=3)
        vs = np.vstack((dja, dj))
        plt.plot(vs[:,0], vs[:,1], 'k', linewidth=0.5)

plt.axis('equal')
