from random import random


def f(x):
    return 2*x[0]**2 + 2*x[0]*x[1] + x[1]**2


def df(x):
    dx = 4*x[0] + 2*x[1]
    dy = 2*x[0] + 2*x[1]
    return dx, dy


rho = 0.005
precision = 0.0000001
difference = 100
x = [[random() for _ in range(2)]]

while difference > precision:
    dr = df(x)
    prev_x = x
    x = [x[i] - rho * dr[i] for i in range(2)]
    difference = abs(prev_x[0]-x[0] + prev_x[1]-x[1])**2
    print("x = {}, df = {}, f(x) = {:f}".format(x, dr, f(x)))
