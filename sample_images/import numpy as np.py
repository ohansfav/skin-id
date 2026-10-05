import numpy as np

# --- Configuration for the Sample Problem ---
N_PLANTS = 50          # Number of solutions (waterwheels)
DIMENSIONS = 2         # Number of variables per solution (e.g., [x, y])
MAX_ITER = 200         # Total number of iterations (generations)
LOWER_BOUND = -100     # Minimum value for variables
UPPER_BOUND = 100      # Maximum value for variables

# --- Algorithm Parameters (standard WWPA parameters) ---
F = 1.0
c = 1.0

# Define the Objective Function (Goal: Minimize this value)
def calculate_fitness(position):
    """
    Example: Sphere function for minimization.
    Minimum is 0 at position [0, 0].
    """
    return sum(x**2 for x in position)

# 1. Initialization
# Create random initial positions for all plants within bounds
positions = LOWER_BOUND + np.random.rand(N_PLANTS, DIMENSIONS) * (UPPER_BOUND - LOWER_BOUND)

# Calculate initial fitness values for each plant
fitnesses = [calculate_fitness(pos) for pos in positions]

# Find the best solution found so far
best_idx = np.argmin(fitnesses)
best_position = positions[best_idx].copy()
best_fitness = fitnesses[best_idx]

print(f"Initial Best Fitness: {best_fitness}")

# 2. Iteration Loop (Main Optimization Process)
for t in range(1, MAX_ITER + 1):
    # Update the internal parameter K which balances search types
    # K increases over time to shift from exploration to exploitation
    K = 1 + (2 * t**2) / (MAX_ITER**3 + F)

    for i in range(N_PLANTS):
        # Generate random numbers needed for stochastic decisions
        r = np.random.rand()
        r1, r2, r3 = np.random.rand(3)
        
        # Decide whether to explore globally or exploit locally
        if r < 0.5:
            # Exploration Phase (Global Search)
            # Simulate searching far away for new prey
            W = r1 * (positions[i] + 2 * K)
            new_position = positions[i] + W * (2 * K + r2)
        else:
            # Exploitation Phase (Local Search)
            # Simulate moving towards the best known food source
            W = r3 * (K * best_position + r3 * positions[i])
            new_position = positions[i] + K * W

        # Ensure new positions stay within defined boundaries
        new_position = np.clip(new_position, LOWER_BOUND, UPPER_BOUND)
        
        # Evaluate the quality of this new position
        new_fitness = calculate_fitness(new_position)
        
        # Update the individual plant's position if the new one is better
        if new_fitness < fitnesses[i]: # Use '>' here if maximizing
            positions[i] = new_position
            fitnesses[i] = new_fitness

    # Check if any of the new positions improved the overall best solution
    current_best_idx = np.argmin(fitnesses)
    if fitnesses[current_best_idx] < best_fitness: # Use '>' here if maximizing
        best_fitness = fitnesses[current_best_idx]
        best_position = positions[current_best_idx].copy()
            
    # Optional: Print progress every 50 iterations
    if t % 50 == 0:
        print(f"Iteration {t}: Best Fitness = {best_fitness}")

# 3. Termination: Display the final result
print("-" * 30)
print(f"Final Best Position: {best_position}")
print(f"Final Best Fitness (Objective Value): {best_fitness}")