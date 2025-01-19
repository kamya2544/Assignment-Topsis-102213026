import sys
import pandas as pd
import numpy as np

# Validating weights and impacts
def validate_weights_impacts(weights, impacts, num_columns):
    if len(weights) != num_columns:
        raise ValueError("Number of weights must match the number of criteria (columns 2 to end).")
    if len(impacts) != num_columns:
        raise ValueError("Number of impacts must match the number of criteria (columns 2 to end).")
    if not all(impact in ['+', '-'] for impact in impacts):
        raise ValueError("Impacts must be either '+' or '-'.")

# Clculating TOPSIS score
def calculate_topsis(data, weights, impacts):
    norm_data = data / np.sqrt((data**2).sum(axis=0))  # Normalize the data
    weighted_data = norm_data * weights  # Apply weights

    ideal_best = []
    ideal_worst = []

    for i, impact in enumerate(impacts):
        if impact == '+':
            ideal_best.append(weighted_data.iloc[:, i].max())
            ideal_worst.append(weighted_data.iloc[:, i].min())
        else:
            ideal_best.append(weighted_data.iloc[:, i].min())
            ideal_worst.append(weighted_data.iloc[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    dist_best = np.sqrt(((weighted_data - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_data - ideal_worst)**2).sum(axis=1))

    scores = dist_worst / (dist_best + dist_worst)
    return scores

# Main function
def main():
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

    input_file = sys.argv[1]
    weights = list(map(float, sys.argv[2].split(',')))
    impacts = sys.argv[3].split(',')
    result_file = sys.argv[4]

    try:
        data = pd.read_excel(input_file) if input_file.endswith('.xlsx') else pd.read_csv(input_file)
        data.to_csv(result_file.replace("-result.csv", "-data.csv"), index=False)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    if data.shape[1] < 3:
        print("Error: Input file must contain at least three columns.")
        sys.exit(1)

    try:
        numeric_data = data.iloc[:, 1:].apply(pd.to_numeric)
    except ValueError:
        print("Error: Non-numeric values found in columns 2 to end.")
        sys.exit(1)

    try:
        validate_weights_impacts(weights, impacts, numeric_data.shape[1])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    scores = calculate_topsis(numeric_data, np.array(weights), impacts)
    data['Topsis Score'] = scores
    data['Rank'] = scores.rank(ascending=False).astype(int)

    data.to_csv(result_file, index=False)
    print(f"Results written to {result_file}")

if __name__ == "__main__":
    main()

