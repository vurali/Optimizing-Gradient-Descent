# Gradient Descent Linear Regression

This repository contains a Python implementation of the gradient descent algorithm for linear regression. The code has been optimized for performance and readability using NumPy's vectorized operations, type hints, and efficient coding practices.

## Features

- **Vectorized Operations**: Leveraged NumPy's vectorized operations for efficient calculations.
- **Type Hinting**: Added type hints for better code readability and maintainability.
- **f-strings**: Used f-strings for concise and readable print statements.
- **Efficiency Improvements**: Streamlined calculations and removed unnecessary type conversions.

## Code Overview

The main script performs the following steps:

1. **Load Data**: Reads the data from a CSV file.
2. **Compute Initial Error**: Calculates the initial error with the starting values of intercept and slope.
3. **Gradient Descent**: Runs the gradient descent algorithm for a specified number of iterations to optimize the intercept and slope.
4. **Final Error**: Computes the error after optimization.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/vurali/Optimizing-Gradient-Descent.git
    cd Optimizing-Gradient-Descent
    ```

2. Install the required dependencies:
    ```sh
    pip install numpy
    ```

3. Ensure you have a data file named `data.csv` in the repository directory.

## Usage

To run the gradient descent algorithm, execute the `main.py` script:

```sh
python main.py
