def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

    Returns:
        float: The x-coordinate of the minimum point.
    """
    f_deriv = lambda x: 4*(x**3) - 9*(x**2)
    f_double_deriv = lambda x: 12*(x**2) - 18*x
    x = start_x
    x_old = x

    while f_deriv(x)!=0:
        x_old = x
        x = x - 0.01*f_deriv(x)
        if x == x_old:
            break

    return round(x, 2)


print(find_treasure(-1.0))