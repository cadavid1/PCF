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
    "Host": {"greeting_efficiency": 5, "seating_strategy": 5, "reservation_handling": 5},
    "Server": {"service_speed": 5, "interaction_quality": 5, "upselling_efficiency": 4},
    "Busser": {"table_clearing_speed": 5, "cleanliness": 5, "coordination": 4},
    "Cook": {"cooking_speed": 6, "accuracy": 5, "special_requests_handling": 4},
    "Chef": {"kitchen_management": 5, "creativity": 4, "error_resolution": 5},
    "Bartender": {"drink_preparation_speed": 5, "customer_interaction": 5, "drink_quality": 5},
    "Manager": {"decision_making": 5, "conflict_resolution": 5, "staff_coordination": 5},
    "Customer": {"satisfaction_level": 5, "time_spent": 5, "tipping_behavior": 5},
    "Environment": {"cleanliness": 5, "ambiance": 4, "noise_level": 5},
    "Menu": {"variety": 5, "quality": 5, "price_point": 5},
    "Location": {"accessibility": 5, "parking": 4, "neighborhood_safety": 5}
}

# Additional factors for three-star restaurant
additional_factors = {
    "Health_Code_Compliance": 6,
    "Staff_Turnover_Rate": 5,
    "Equipment_Reliability": 5,
    "Supplier_Quality": 5,
    "Customer_Expectations": 5,
    "Local_Competition": 6,
    "Marketing_Efforts": 4,
    "Online_Presence": 4
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
        food_quality = (constructed_params["Cook"]["accuracy"] + constructed_params["Chef"]["creativity"] + constructed_params["Menu"]["quality"]) / 3
        service_quality = (constructed_params["Server"]["interaction_quality"] + constructed_params["Bartender"]["customer_interaction"]) / 2
        ambiance = constructed_params["Environment"]["ambiance"]
        value = (10 - constructed_params["Menu"]["price_point"] + constructed_params["Menu"]["variety"]) / 2
        
        satisfaction_score = np.mean([food_quality, service_quality, ambiance, value])
        
        # Simulate total time per meal as a combination of key performance factors
        total_time_per_meal = constructed_params["Cook"]["cooking_speed"] + constructed_params["Server"]["service_speed"] + constructed_params["Bartender"]["drink_preparation_speed"]
        
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
csv_filename = f'three_star_results_{timestamp}.csv'
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
plt.title("Distribution of Total Time Per Meal (Three-Star)")
plt.xlabel("Total Time Per Meal (minutes)")
plt.ylabel("Frequency")

# Plot 2: Total Time Per Meal vs Satisfaction Score
plt.subplot(1, 3, 2)
plt.scatter(simulation_results["Total Time Per Meal"], simulation_results["Satisfaction Score"], alpha=0.5)
plt.title("Total Time Per Meal vs Satisfaction Score (Three-Star)")
plt.xlabel("Total Time Per Meal (minutes)")
plt.ylabel("Satisfaction Score")

# Plot 3: Distribution of Performance Score
plt.subplot(1, 3, 3)
plt.hist(simulation_results["Performance Score"], bins=30, edgecolor='black')
plt.title("Distribution of Performance Score (Three-Star)")
plt.xlabel("Performance Score")
plt.ylabel("Frequency")

plt.tight_layout()

# Save the figure as an image file with timestamp
plot_filename = f'three_star_plots_{timestamp}.png'
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