# Standard Libraries
import os
import csv
from collections import defaultdict
# Data Handling
import pandas as pd
import numpy as np
# Machine Learning & NLP
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
# Deep Learning
from tensorflow.keras.models import load_model
# Statistical Analysis
from scipy.stats import binom
# Utilities
from tqdm import tqdm
# Custom Imports
from Codes.import_data import reformat_cgc


############################################
#initialization
###############################################
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Subfinder input and output descriptionx.")

# Add the arguments
parser.add_argument("-i", "--input", type=str, default="cgc_standard.out", help="Input data file name (default: 'cgc_standard.out')")
parser.add_argument("-o", "--output", type=str, default="predict_summary.csv", help="Model prediction results summary (default: 'predict_summary.csv')")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
input_file = args.input
output_file_location = args.output


# Example: Read input_file and write results to output_file
print(f"Input data file: {input_file}")
print(f"Output summary file: {output_file_location}")
############################################
###tranform cgc input to required input on predict model
########################################
output_file = 'reformat'
reformat_cgc(input_file, output_file)
    

# Define input and output file paths
input_file_path = output_file  # 
output_file_path = 'output.csv'  

# Open the input file for reading and the output file for writing
with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
    reader = infile.readlines()
    writer = csv.writer(outfile)
    
    # Write the header to the CSV file
    writer.writerow(['cgc_id', 'sequence'])
    
    # Process each line from the input file
    for line in reader:
        parts = line.strip().split('\t')  # Split the line into two parts
        writer.writerow(parts)  # Write the parts to the CSV file
#remove files        
os.remove('reformat')
############################################
#Import csv file 
############################################
data = pd.read_csv(output_file_path)
test_seqs = np.array([test_item.replace("|", ",").replace(",", " ") for test_item in data["sequence"].values])
label = data['cgc_id']
os.remove(output_file_path)


# def custom_tokenizer(x):
#     return str(x).replace("|", ",").split(',')
# # import training data
# data_train = pd.read_csv('Data/Train_data.csv')

# # Define parameters directly
# n_estimators = 100
# class_weight = "balanced"
# sampling_strategy = "all"
# replacement = True
# bootstrap = False

# # Get order of classes
# order = list(data_train["high_level_substr"].value_counts().index)

# # Create the classifier with direct parameter assignment
# balanced_rf = BalancedRandomForestClassifier(
#     n_estimators=n_estimators,
#     class_weight=class_weight,
#     sampling_strategy=sampling_strategy,
#     replacement=replacement,
#     bootstrap=bootstrap,
#     n_jobs=-1
# )

# # Create and train pipeline
# clf_one_vs_rest = Pipeline([
#     ('vectorizer', CountVectorizer(
#         tokenizer=custom_tokenizer, 
#         lowercase=False
#     )), 
#     ('vr', OneVsRestClassifier(balanced_rf))
# ])

# # Fit the model directly
# clf_one_vs_rest.fit(data_train['sig_gene_seq'].values, data_train["high_level_substr"].values)

# ## To get easy to understand p values
# ## we need to change the model a bit

# class_order = clf_one_vs_rest.classes_

# # get the predictions
# preds_df = pd.DataFrame()
# counter = 0
# outer_catch = []


# # we will store the normalized probabilities here
# normalized_probs = np.zeros((len(test_seqs), len(class_order)))

# how_many = 15
# for inner in tqdm(range(0, how_many)):
#     balanced_rf = BalancedRandomForestClassifier(n_jobs=7, class_weight="balanced", random_state=inner)
    
#     model = Pipeline([
#         ('vectorizer', CountVectorizer(tokenizer=custom_tokenizer, lowercase=False)), 
#         ('vr', OneVsRestClassifier(balanced_rf))
#     ])
    
#     X_train, X_val, y_train, y_val = train_test_split(
#         data_train['sig_gene_seq'].values, 
#         data_train["high_level_substr"].values, 
#         test_size=0.2, random_state=inner
#     )
    
#     model.fit(X_train, y_train)
    
#     preds_proba = model.predict_proba(test_seqs)
#     normalized_probs += preds_proba  
#     ## grab all the separate one vs all estimators
#     ova_ests = model.named_steps['vr'].estimators_
#     inner_counter = 0
#     inner_catch = []
    
#     for ests in ova_ests:
#         # We need to use the vectorizer first
#         vectorized_test_seqs = model.named_steps['vectorizer'].transform(test_seqs)
#         # Now predict on the transformed data
#         preds = ests.predict_proba(vectorized_test_seqs)
#         preds_df1 = pd.DataFrame(preds)
#         preds_df1 = preds_df1[[1]]
#         preds_df1 = pd.DataFrame(preds_df1)
#         preds_df1.columns = ["unnormalized_prob_" + class_order[inner_counter] + "_" + str(counter)]
#         inner_catch.append(preds_df1)
#         inner_counter += 1
    
#     inner_catch_df = pd.concat(inner_catch, axis=1)
#     outer_catch.append(inner_catch_df)
#     counter += 1

# # Combine stored probability data into a single DataFrame
# outer_catch_df = pd.concat(outer_catch, axis=1)

# # Ensure column names are sorted
# outer_catch_df = outer_catch_df[np.sort(outer_catch_df.columns)]

# # Assign sequences to DataFrame
# outer_catch_df["sequence"] = test_seqs

# # Reorder columns to move 'sequence' to the front
# cols = ["sequence"] + [col for col in outer_catch_df.columns if col != "sequence"]
# outer_catch_df = outer_catch_df[cols]

# # Normalize probabilities
# normalized_probs /= how_many
# normalized_probs = pd.DataFrame(normalized_probs, columns=["normalized_prob_" + item for item in class_order])

# # Assign sequence identifiers
# normalized_probs["sequence"] = test_seqs

# # Reorder columns to move 'sequence' to the front
# cols = ["sequence"] + [col for col in normalized_probs.columns if col != "sequence"]
# normalized_probs = normalized_probs[cols]

# # Determine the most probable class
# normalized_probs["predicted_substrate"] = normalized_probs.iloc[:, 1:].idxmax(axis=1)
# normalized_probs["predicted_substrate"] = normalized_probs["predicted_substrate"].apply(lambda x: x.split("_")[-1])

# # Reshape data for further analysis
# outer_catch_df = outer_catch_df.melt(id_vars=["sequence"]).reset_index(drop=True)

# # Extract relevant information from variable names
# outer_catch_df["substrate"] = outer_catch_df["variable"].apply(lambda x: x.split("_")[-2])
# outer_catch_df["order"] = outer_catch_df["variable"].apply(lambda x: x.split("_")[-1])

# # Sort values for consistency
# outer_catch_df = outer_catch_df.sort_values(["sequence", "substrate"], ascending=[True, True])

# # Classify probabilities into binary predictions
# outer_catch_df["yes_or_no"] = (outer_catch_df["value"] > 0.5).astype(int)

# # Aggregate probability and binary classification results
# outer_catch_df_summary = (
#     outer_catch_df
#     .groupby(["sequence", "substrate"])
#     .agg(probability_score=("value", "mean"), successes=("yes_or_no", "sum"))
#     .reset_index()
# )

# # Define statistical function for p-value computation
# def p_value_function(successes, trials=how_many, prob=0.5):
#     return 1 - binom.cdf(successes, trials, prob)

# # Compute p-values and clean up DataFrame
# outer_catch_df_summary["p_value"] = outer_catch_df_summary["successes"].apply(p_value_function)
# outer_catch_df_summary = outer_catch_df_summary.drop(columns=["successes"])

# # Final sorting and resetting index
# outer_catch_df_summary = outer_catch_df_summary.sort_values(["sequence", "p_value"]).reset_index(drop=True)
# outer_catch_df_summary.to_csv(output_file_location, index = False)
# normalized_probs["predicted_substrate"].value_counts()
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
def custom_tokenizer(x):
    """Split input by commas, replacing pipes with commas first."""
    return str(x).replace("|", ",").split(',')

def create_model(random_state=None):
    """Create a pipeline with balanced random forest classifier."""
    balanced_rf = BalancedRandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state
    )
    
    return Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=custom_tokenizer, lowercase=False)), 
        ('vr', OneVsRestClassifier(balanced_rf))
    ])

def load_data(train_file='Data/Train_data.csv'):
    """Load training data and return feature/target columns."""
    data_train = pd.read_csv(train_file)
    return data_train['sig_gene_seq'].values, data_train["high_level_substr"].values

def predict_with_ensemble(X_train, y_train, test_seqs, num_models=15):
    """
    Train multiple models and aggregate their predictions.
    Returns normalized probabilities and class-specific probability details.
    """
    # Get unique class labels from training data
    class_order = np.unique(y_train)
    
    # Initialize storage for probabilities
    normalized_probs = np.zeros((len(test_seqs), len(class_order)))
    all_model_predictions = []
    
    # Train multiple models with different random seeds
    for i in tqdm(range(num_models)):
        # Create and train model
        model = create_model(random_state=i)
        
        # Split data for this iteration
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=i
        )
        
        # Train the model
        model.fit(X_train_split, y_train_split)
        
        # Get overall predictions and add to normalized probabilities
        preds_proba = model.predict_proba(test_seqs)
        normalized_probs += preds_proba
        
        # Get individual class predictions
        vectorized_test_seqs = model.named_steps['vectorizer'].transform(test_seqs)
        class_predictions = {}
        
        for idx, estimator in enumerate(model.named_steps['vr'].estimators_):
            class_name = class_order[idx]
            probs = estimator.predict_proba(vectorized_test_seqs)[:, 1]  # Get positive class probabilities
            class_predictions[f"unnormalized_prob_{class_name}_{i}"] = probs
        
        # Store this model's predictions
        model_df = pd.DataFrame(class_predictions)
        all_model_predictions.append(model_df)
    
    # Combine all predictions
    combined_predictions = pd.concat(all_model_predictions, axis=1)
    combined_predictions["sequence"] = test_seqs
    
    # Process normalized probabilities
    normalized_probs /= num_models
    normalized_probs_df = pd.DataFrame(
        normalized_probs, 
        columns=[f"normalized_prob_{item}" for item in class_order]
    )
    normalized_probs_df["sequence"] = test_seqs
    
    return combined_predictions, normalized_probs_df, class_order

def analyze_results(combined_predictions, normalized_probs_df, class_order, num_models=15):
    """
    Analyze the ensemble results to get:
    1. Predicted substrate for each sequence
    2. P-values for substrate assignments
    """
    # Find most probable class for each sequence
    idx_cols = [col for col in normalized_probs_df.columns if col != "sequence"]
    normalized_probs_df["predicted_substrate"] = normalized_probs_df[idx_cols].idxmax(axis=1)
    normalized_probs_df["predicted_substrate"] = normalized_probs_df["predicted_substrate"].apply(
        lambda x: x.split("_")[-1]
    )
    
    # Melt the dataframe for analysis
    id_vars = ["sequence"]
    combined_predictions_melted = combined_predictions.melt(id_vars=id_vars).reset_index(drop=True)
    
    # Extract substrate and model order from variable names
    combined_predictions_melted["substrate"] = combined_predictions_melted["variable"].apply(
        lambda x: x.split("_")[-2]
    )
    combined_predictions_melted["order"] = combined_predictions_melted["variable"].apply(
        lambda x: x.split("_")[-1]
    )
    
    # Sort values
    combined_predictions_melted = combined_predictions_melted.sort_values(
        ["sequence", "substrate"], 
        ascending=[True, True]
    )
    
    # Create binary classifications (probability > 0.5)
    combined_predictions_melted["yes_or_no"] = (combined_predictions_melted["value"] > 0.5).astype(int)
    
    # Aggregate results by sequence and substrate
    summary = (
        combined_predictions_melted
        .groupby(["sequence", "substrate"])
        .agg(
            probability_score=("value", "mean"), 
            successes=("yes_or_no", "sum")
        )
        .reset_index()
    )
    
    # Define p-value function
    def calc_p_value(successes, trials=num_models, prob=0.5):
        return 1 - binom.cdf(successes, trials, prob)
    
    # Calculate p-values
    summary["p_value"] = summary["successes"].apply(calc_p_value)
    summary = summary.drop(columns=["successes"])
    
    # Sort by p-value
    summary = summary.sort_values(["sequence", "p_value"]).reset_index(drop=True)
    
    return summary, normalized_probs_df

def main(train_file='Data/Train_data.csv', output_file= output_file_location, test_sequences= test_seqs):
    """Main function to run the entire pipeline."""
    # Load training data
    X_train, y_train = load_data(train_file)
    
    # Set number of models in ensemble
    num_models = 15
    
    # Train ensemble and get predictions
    combined_predictions, normalized_probs_df, class_order = predict_with_ensemble(
        X_train, y_train, test_sequences, num_models
    )
    
    # Analyze results
    summary, normalized_probs_df = analyze_results(
        combined_predictions, normalized_probs_df, class_order, num_models
    )
    
    # Save results
    summary.to_csv(output_file, index=False)
    
    # Return counts of predicted substrates
    return normalized_probs_df["predicted_substrate"].value_counts()

if __name__ == "__main__":
    # Example usage
    results = main()
    print(results)