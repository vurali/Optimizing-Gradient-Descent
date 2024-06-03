import numpy as np


def compute_error_for_given_points(b, m, points):
    """
    Calculates the total error between the predicted values and the actual values.

    Args:
        b (float): The y-intercept of the line.
        m (float): The slope of the line.
        points (numpy.ndarray): A 2D array containing the x and y coordinates of the data points.

    Returns:
        float: The total error.
    """
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    """
    Calculates the gradients for the y-intercept and slope using the current values and the learning rate.

    Args:
        b_current (float): The current y-intercept.
        m_current (float): The current slope.
        points (numpy.ndarray): A 2D array containing the x and y coordinates of the data points.
        learningRate (float): The learning rate used to update the y-intercept and slope.

    Returns:
        list: The updated y-intercept and slope.
    """
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    """
    Runs the gradient descent algorithm for the specified number of iterations.

    Args:
        points (numpy.ndarray): A 2D array containing the x and y coordinates of the data points.
        starting_b (float): The initial y-intercept.
        starting_m (float): The initial slope.
        learning_rate (float): The learning rate used to update the y-intercept and slope.
        num_iterations (int): The number of iterations to run the gradient descent algorithm.

    Returns:
        list: The final y-intercept and slope.
    """
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


def run():
    """
    Runs the main program, which loads the data from a CSV file, sets the initial parameters,
    and runs the gradient descent algorithm.
    """
    points = np.genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001  # Hyperparameter

    # y = mx + b ( slope Formula)

    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                              compute_error_for_given_points(initial_b,
                                                                                                             initial_m,
                                                                                                             points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
                                                                      compute_error_for_given_points(b, m, points)))


if __name__ == '__main__':
    run()
