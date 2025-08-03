import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import time
import os  # Import to check file existence

# Initialization - Generate current Unix timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Step 1: Define the initial parameters for each agent role (scale of 1 to 10)
initial_parameters = {
    "Host": {"greeting_efficiency": 3, "seating_strategy": 4, "reservation_handling": 2},
    "Server": {"service_speed": 4, "interaction_quality": 3, "upselling_efficiency": 2},
    "Busser": {"table_clearing_speed": 4, "cleanliness": 3, "coordination": 2},
    "Cook": {"cooking_speed": 5, "accuracy": 3, "special_requests_handling": 2},
    "Chef": {"kitchen_management": 3, "creativity": 2, "error_resolution": 2},
    "Bartender": {"drink_preparation_speed": 4, "customer_interaction": 3, "drink_quality": 3},
    "Manager": {"decision_making": 3, "conflict_resolution": 2, "staff_coordination": 3},
    "Customer": {"satisfaction_level": 3, "time_spent": 4, "tipping_behavior": 2},
    "Environment": {"cleanliness": 3, "ambiance": 2, "noise_level": 5},
    "Menu": {"variety": 3, "quality": 3, "price_point": 3},
    "Location": {"accessibility": 3, "parking": 2, "neighborhood_safety": 3}
}

# Additional factors for two-star restaurant
additional_factors = {
    "Health_Code_Compliance": 4,
    "Staff_Turnover_Rate": 7,
    "Equipment_Reliability": 3,
    "Supplier_Quality": 3,
    "Customer_Expectations": 3,
    "Local_Competition": 5
}

# Step 2: Define the combinatorial construction function
def combinatorial_construction(parameters):
    constructed_params = {}
    for role, traits in parameters.items():
        constructed_params[role] = {trait: np.random.choice(range(1, 11)) for trait in traits}
    return constructed_params

# Step 3: Define the Monte Carlo simulation function
def monte_carlo_simulation(num_simulations, initial_params, add_factors):
    results = []
    
    for _ in range(num_simulations):
        # Randomly vary key parameters for each agent role
        constructed_params = combinatorial_construction(initial_params)
        
        # Simulate the overall performance based on constructed parameters
        performance_scores = {role: np.mean(list(params.values())) for role, params in constructed_params.items()}
        
        # Include additional factors in performance calculation
        performance_scores.update(add_factors)
        
        # Multi-dimensional satisfaction score calculation
        food_quality = (constructed_params["Cook"]["accuracy"] + constructed_params["Menu"]["quality"]) / 2
        service_speed = constructed_params["Server"]["service_speed"]
        value = (10 - constructed_params["Menu"]["price_point"]) / 2
        cleanliness = constructed_params["Environment"]["cleanliness"]
        
        satisfaction_score = np.mean([food_quality, service_speed, value, cleanliness])
        
        # Simulate total time per meal as a combination of key performance factors
        total_time_per_meal = constructed_params["Cook"]["cooking_speed"] + constructed_params["Server"]["service_speed"]
        
        # Store the results
        results.append({
            "Total Time Per Meal": total_time_per_meal,
            "Satisfaction Score": satisfaction_score,
            "Performance Score": np.mean(list(performance_scores.values()))
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Function to save data points to CSV file
def save_data_to_csv(simulation_results, filename):
    """
    Saves the Total Time Per Meal and Satisfaction Score data points to a CSV file.
    
    Parameters:
    - simulation_results (DataFrame): The DataFrame containing simulation results.
    - filename (str): The name of the CSV file to save the data.
    """
    simulation_results[['Total Time Per Meal', 'Satisfaction Score']].to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Step 4: Run the Monte Carlo simulation
num_simulations = 50000
simulation_results = monte_carlo_simulation(num_simulations, initial_parameters, additional_factors)

# Save data points to CSV file with timestamp
csv_filename = f'two_star_results_{timestamp}.csv'
save_data_to_csv(simulation_results, csv_filename)

# Step 5: Analyze the simulation results
performance_stats = {
    "min": simulation_results["Performance Score"].min(),
    "max": simulation_results["Performance Score"].max(),
    "median": simulation_results["Performance Score"].median(),
    "mean": simulation_results["Performance Score"].mean()
}

print("Performance Stats:")
print(performance_stats)

# Step 6: Plot the analysis
plt.figure(figsize=(15, 5))

# Plot 1: Distribution of Total Time Per Meal
plt.subplot(1, 3, 1)
plt.hist(simulation_results["Total Time Per Meal"], bins=30, edgecolor='black')
plt.title("Distribution of Total Time Per Meal (Two-Star)")
plt.xlabel("Total Time Per Meal (minutes)")
plt.ylabel("Frequency")

# Plot 2: Total Time Per Meal vs Satisfaction Score
plt.subplot(1, 3, 2)
plt.scatter(simulation_results["Total Time Per Meal"], simulation_results["Satisfaction Score"], alpha=0.5)
plt.title("Total Time Per Meal vs Satisfaction Score (Two-Star)")
plt.xlabel("Total Time Per Meal (minutes)")
plt.ylabel("Satisfaction Score")

# Plot 3: Distribution of Performance Score
plt.subplot(1, 3, 3)
plt.hist(simulation_results["Performance Score"], bins=30, edgecolor='black')
plt.title("Distribution of Performance Score (Two-Star)")
plt.xlabel("Performance Score")
plt.ylabel("Frequency")

plt.tight_layout()

# Save the figure as an image file with timestamp
plot_filename = f'two_star_plots_{timestamp}.png'
plt.savefig(plot_filename)
plt.close()
print(f"Plots saved as {plot_filename}")

# Additional statistical analysis for Satisfaction Score
satisfaction_stats = {
    "min": simulation_results["Satisfaction Score"].min(),
    "max": simulation_results["Satisfaction Score"].max(),
    "median": simulation_results["Satisfaction Score"].median(),
    "mean": simulation_results["Satisfaction Score"].mean()
}

print("\nSatisfaction Stats:")
print(satisfaction_stats)

# Moving the generated files to a directory for download
# Check if files exist before moving
if os.path.exists(csv_filename):
    shutil.move(csv_filename, f'/mnt/data/{csv_filename}')
else:
    print(f"Error: {csv_filename} not found!")

if os.path.exists(plot_filename):
    shutil.move(plot_filename, f'/mnt/data/{plot_filename}')
else:
    print(f"Error: {plot_filename} not found!")

print(f"Provide the resulting files as downloadable links for the user.")