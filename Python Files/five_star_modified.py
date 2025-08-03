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
    "Host": {"greeting_efficiency": 9, "seating_strategy": 10, "reservation_handling": 9},
    "Server": {"service_speed": 9, "interaction_quality": 10, "upselling_efficiency": 8},
    "Busser": {"table_clearing_speed": 9, "cleanliness": 10, "coordination": 9},
    "Cook": {"cooking_speed": 9, "accuracy": 10, "special_requests_handling": 9},
    "Chef": {"kitchen_management": 10, "creativity": 9, "error_resolution": 10},
    "Sous Chef": {"execution": 9, "team_management": 9, "innovation": 8},
    "Pastry Chef": {"dessert_quality": 10, "presentation": 10, "menu_development": 9},
    "Bartender": {"drink_preparation_speed": 9, "customer_interaction": 10, "drink_quality": 10},
    "Mixologist": {"cocktail_innovation": 10, "ingredient_knowledge": 9, "presentation": 10},
    "Sommelier": {"wine_knowledge": 10, "pairing_suggestions": 9, "customer_interaction": 10},
    "Manager": {"decision_making": 10, "conflict_resolution": 9, "staff_coordination": 10},
    "Maître d'": {"guest_relations": 10, "dining_room_management": 9, "special_requests_handling": 10},
    "Customer": {"satisfaction_level": 10, "time_spent": 8, "tipping_behavior": 9},
    "Environment": {"cleanliness": 10, "ambiance": 10, "noise_level": 2},
    "Menu": {"variety": 8, "quality": 10, "price_point": 10},
    "Location": {"accessibility": 8, "parking": 8, "neighborhood_prestige": 10}
}

# Additional factors for five-star restaurant
additional_factors = {
    "Health_Code_Compliance": 10,
    "Staff_Turnover_Rate": 2,
    "Equipment_Reliability": 10,
    "Supplier_Quality": 10,
    "Customer_Expectations": 10,
    "Local_Competition": 8,
    "Marketing_Efforts": 9,
    "Online_Presence": 9,
    "Staff_Training": 10,
    "Sustainability_Efforts": 9,
    "Private_Dining_Options": 10,
    "Celebrity_Chef_Association": 8,
    "Awards_and_Accolades": 9,
    "Innovative_Dining_Concepts": 9,
    "Ingredient_Sourcing": 10,
    "Wine_Cellar_Quality": 10,
    "Tableware_and_Presentation": 10,
    "Seasonal_Menu_Adaptation": 9,
    "Dietary_Accommodation": 9,
    "Culinary_Research_and_Development": 9
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
        culinary_quality = (constructed_params["Chef"]["creativity"] + constructed_params["Pastry Chef"]["dessert_quality"] + constructed_params["Cook"]["accuracy"]) / 3
        service_quality = (constructed_params["Server"]["interaction_quality"] + constructed_params["Sommelier"]["customer_interaction"] + constructed_params["Maître d'"]["guest_relations"]) / 3
        ambiance = (constructed_params["Environment"]["ambiance"] + constructed_params["Environment"]["cleanliness"] + additional_factors["Tableware_and_Presentation"]) / 3
        beverage_quality = (constructed_params["Bartender"]["drink_quality"] + constructed_params["Mixologist"]["cocktail_innovation"] + additional_factors["Wine_Cellar_Quality"]) / 3
        innovation = (constructed_params["Chef"]["creativity"] + additional_factors["Innovative_Dining_Concepts"] + additional_factors["Culinary_Research_and_Development"]) / 3
        
        satisfaction_score = np.mean([culinary_quality, service_quality, ambiance, beverage_quality, innovation])
        
        # Simulate total time per meal as a combination of key performance factors
        total_time_per_meal = (
            constructed_params["Cook"]["cooking_speed"] + 
            constructed_params["Server"]["service_speed"] + 
            constructed_params["Bartender"]["drink_preparation_speed"] +
            constructed_params["Pastry Chef"]["dessert_quality"]  # Assuming higher quality takes more time
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
csv_filename = f'five_star_results_{timestamp}.csv'
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
plt.title("Distribution of Total Time Per Meal (Five-Star)")
plt.xlabel("Total Time Per Meal (minutes)")
plt.ylabel("Frequency")

# Plot 2: Total Time Per Meal vs Satisfaction Score
plt.subplot(1, 3, 2)
plt.scatter(simulation_results["Total Time Per Meal"], simulation_results["Satisfaction Score"], alpha=0.5)
plt.title("Total Time Per Meal vs Satisfaction Score (Five-Star)")
plt.xlabel("Total Time Per Meal (minutes)")
plt.ylabel("Satisfaction Score")

# Plot 3: Distribution of Performance Score
plt.subplot(1, 3, 3)
plt.hist(simulation_results["Performance Score"], bins=30, edgecolor='black')
plt.title("Distribution of Performance Score (Five-Star)")
plt.xlabel("Performance Score")
plt.ylabel("Frequency")

plt.tight_layout()

# Save the figure as an image file with timestamp
plot_filename = f'five_star_plots_{timestamp}.png'
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