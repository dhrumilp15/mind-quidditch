import matplotlib.pyplot as plt
import numpy as np

def get_dist(radius, ref_dist = (11.7, 36, 1.57)) -> float:
    '''
        Gets the distance in inches from the object to the camera
        :param radius: The radius of the ball (in px)
        :param ref_dist: Conditions that are true, calculated with the camera's focal length
        :return: The distance from the object to the camera
    '''
    return np.prod(ref_dist) / radius

radius = np.arange(1.0, 30.0, 0.5)

y = np.array(list(map(get_dist, radius)))

fig, ax = plt.subplots()

ax.plot(radius, y)
ax.set(xlabel='radius', ylabel='Estimated distance')

plt.show()

