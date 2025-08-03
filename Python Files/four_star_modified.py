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
    "Host": {"greeting_efficiency": 7, "seating_strategy": 8, "reservation_handling": 7},
    "Server": {"service_speed": 7, "interaction_quality": 8, "upselling_efficiency": 7},
    "Busser": {"table_clearing_speed": 7, "cleanliness": 8, "coordination": 7},
    "Cook": {"cooking_speed": 8, "accuracy": 8, "special_requests_handling": 7},
    "Chef": {"kitchen_management": 8, "creativity": 7, "error_resolution": 8},
    "Bartender": {"drink_preparation_speed": 8, "customer_interaction": 8, "drink_quality": 8},
    "Manager": {"decision_making": 8, "conflict_resolution": 7, "staff_coordination": 8},
    "Customer": {"satisfaction_level": 8, "time_spent": 7, "tipping_behavior": 8},
    "Environment": {"cleanliness": 8, "ambiance": 7, "noise_level": 3},
    "Menu": {"variety": 7, "quality": 8, "price_point": 7},
    "Location": {"accessibility": 7, "parking": 6, "neighborhood_safety": 8},
    "Sommelier": {"wine_knowledge": 8, "pairing_suggestions": 7, "customer_interaction": 8}
}

# Additional factors for four-star restaurant
additional_factors = {
    "Health_Code_Compliance": 8,
    "Staff_Turnover_Rate": 3,
    "Equipment_Reliability": 8,
    "Supplier_Quality": 8,
    "Customer_Expectations": 8,
    "Local_Competition": 7,
    "Marketing_Efforts": 7,
    "Online_Presence": 7,
    "Staff_Training": 8,
    "Sustainability_Efforts": 6,
    "Private_Dining_Options": 7
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
        quality = (constructed_params["Cook"]["accuracy"] + constructed_params["Chef"]["creativity"] + constructed_params["Menu"]["quality"]) / 3
        ambiance = (constructed_params["Environment"]["ambiance"] + constructed_params["Environment"]["cleanliness"]) / 2
        service = (constructed_params["Server"]["interaction_quality"] + constructed_params["Sommelier"]["customer_interaction"] + constructed_params["Bartender"]["customer_interaction"]) / 3
        value = (10 - constructed_params["Menu"]["price_point"] + constructed_params["Menu"]["variety"]) / 2
        
        satisfaction_score = np.mean([quality, ambiance, service, value])
        
        # Simulate total time per meal as a combination of key performance factors
        total_time_per_meal = (
            constructed_params["Cook"]["cooking_speed"] + 
            constructed_params["Server"]["service_speed"] + 
            constructed_params["Bartender"]["drink_preparation_speed"]
        )
        
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
csv_filename = f'four_star_results_{timestamp}.csv'
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
plt.title("Distribution of Total Time Per Meal (Four-Star)")
plt.xlabel("Total Time Per Meal (minutes)")
plt.ylabel("Frequency")

# Plot 2: Total Time Per Meal vs Satisfaction Score
plt.subplot(1, 3, 2)
plt.scatter(simulation_results["Total Time Per Meal"], simulation_results["Satisfaction Score"], alpha=0.5)
plt.title("Total Time Per Meal vs Satisfaction Score (Four-Star)")
plt.xlabel("Total Time Per Meal (minutes)")
plt.ylabel("Satisfaction Score")

# Plot 3: Distribution of Performance Score
plt.subplot(1, 3, 3)
plt.hist(simulation_results["Performance Score"], bins=30, edgecolor='black')
plt.title("Distribution of Performance Score (Four-Star)")
plt.xlabel("Performance Score")
plt.ylabel("Frequency")

plt.tight_layout()

# Save the figure as an image file with timestamp
plot_filename = f'four_star_plots_{timestamp}.png'
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