import numpy as np


def triangle(x):
    filtered_x = []
    for i in x:
        if i[0] + i[1] - 1 <= 0 and i[1] - i[0] - 1 <= 0 and -i[1] <= 0:
            filtered_x.append(np.array([i[0], i[1]]))
    return np.array(filtered_x)


def circle(x, r=0.50):
    filtered_x = []
    for i in x:
        if i[0]**2 + i[1]**2 <= r**2:
            filtered_x.append(np.array([i[0], i[1]]))
    return np.array(filtered_x)


def folium(x, a=1, b=1):
    filtered_x = []
    for i in x:
        if (i[0]**2 + i[1]**2)*(i[0]*(i[0]+b)+i[1]**2) - 4*a*i[0]*i[1]**2 <= 0:
            filtered_x.append(np.array([i[0], i[1]]))
    return np.array(filtered_x)


def heart(x, size=0.25, ear_size=3):
    filtered_x = []
    for i in x:
        if (i[0]**2 + i[1]**2 - size)**3 - ear_size*(i[0]**2 * i[1] ** 3) <= 0:
            filtered_x.append(np.array([i[0], i[1]]))
    return np.array(filtered_x)
