#This is from chat gpt and still needs work on
import numpy as np



def particle_swarm_optimization(objective_function, num_particles, num_dimensions, bounds, max_iter):
    # Initialize the swarm
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, num_dimensions))
    velocities = np.zeros((num_particles, num_dimensions))
    personal_best_positions = positions.copy()
    personal_best_fitness = np.zeros(num_particles)
    global_best_position = None
    global_best_fitness = float('inf')

    # Perform optimization iterations
    for _ in range(max_iter):
        # Update particle velocities and positions
        # ...

        # Evaluate fitness and update personal best positions
        # ...

        # Update global best position
        # ...

    return global_best_position, global_best_fitness

# Example usage
num_particles = 20
num_dimensions = 2
bounds = [(-5, 5), (-5, 5)]
max_iter = 100

best_position, best_fitness = particle_swarm_optimization(objective_function, num_particles, num_dimensions, bounds, max_iter)
print("Best solution:", best_position)
print("Best fitness:", best_fitness)