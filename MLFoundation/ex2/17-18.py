import numpy as np


# generate input data with 20% flipping noise
def generate_input_data(time_seed):
    np.random.seed(time_seed)
    raw_X = np.sort(np.random.uniform(-1, 1, 20))
    # 加20%噪声
    noised_y = np.sign(raw_X) * np.where(np.random.random(raw_X.shape[0]) < 0.2, -1, 1)
    return raw_X, noised_y


def calculate_Ein(x, y):
    # calculate median of interval & negative infinite & positive infinite
    thetas = np.array([float("-inf")] + [(x[i] + x[i + 1]) / 2 for i in range(0, x.shape[0] - 1)] + [float("inf")])
    Ein = x.shape[0]
    sign = 1
    target_theta = 0.0
    # positive and negative rays
    for theta in thetas:
        y_positive = np.where(x > theta, 1, -1)
        y_negative = np.where(x < theta, 1, -1)
        error_positive = sum(y_positive != y)
        error_negative = sum(y_negative != y)
        if error_positive > error_negative:
            if Ein > error_negative:
                Ein = error_negative
                sign = -1
                target_theta = theta
        else:
            if Ein > error_positive:
                Ein = error_positive
                sign = 1
                target_theta = theta
    # two corner cases
    if target_theta == float("inf"):
        target_theta = 1.0
    if target_theta == float("-inf"):
        target_theta = -1.0
    return Ein, target_theta, sign


if __name__ == '__main__':
    T = 5000
    total_Ein = 0
    sum_Eout = 0
    for i in range(0, T):
        x, y = generate_input_data(i)
        curr_Ein, theta, sign = calculate_Ein(x, y)
        total_Ein = total_Ein + curr_Ein
        sum_Eout = sum_Eout + 0.5 + 0.3 * sign * (abs(theta) - 1)
    # 17
    print((total_Ein * 1.0) / (T * 20))
    # 18
    print((sum_Eout * 1.0) / T)
