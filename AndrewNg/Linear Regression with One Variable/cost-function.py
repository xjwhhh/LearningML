# array1 = [1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10]
# array2 = [890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157]

array1 = [3, 1, 0, 4]
array2 = [2, 2, 1, 3]


def cost_function(a0, a1):
    cost = 0
    for i in range(0, len(array1)):
        cost += ((a0 + a1 * array1[i] - array2[i])) ** 2
    cost /= (2 * len(array1))
    print(cost)


# cost_function(-596.6, -530.9)
# cost_function(-1780.0, 530.9)
# cost_function(-596.6, 530.9)
# cost_function(-1780.0, -530.9)
cost_function(0, 1)
