import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    
    look_ahead_param = parameter - momentum * velocity
    gradient = grad_fn(look_ahead_param)

    new_velocity = momentum * velocity + learning_rate * gradient
    new_parameter = parameter - new_velocity
    
    return new_parameter, new_velocity

# Example usage
parameter = 1.0
velocity = 0.1
grad_fn = lambda x: x

parameter, velocity = nag_optimizer(parameter, grad_fn, velocity)
print(f"Updated parameter: {parameter}, Updated velocity: {velocity}")
