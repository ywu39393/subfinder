# library imports
import pandas as pd
import gensim
from Codes.Supervised_Trainer_tran import run_end_to_end
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
import plotly.express as px
import os
import tensorflow as tf

##########
#parameter
##########
import argparse
# Initialize the parser
# parser = argparse.ArgumentParser(description="Training model inputs.")

# # Add the arguments
# parser.add_argument("-n", "--number", type=int, default=12, help="Number of substrate class  (default: 7)")
# parser.add_argument("-m", "--model", type=str, default="transformer", help="Select model (default: 'transformer')")

# # Parse the arguments
# args = parser.parse_args()

# # Access the arguments
# range_start = args.number
# range_end = range_start+1
# model_types = [args.model]  # Ensure the model is always a list

# # Your code using n1, n2, and model
# print(f"Number of substrate class: {range_start}")
# print(f"Selected model: {model_types}")



# ############################################
# # # Define the range for top_k using tqdm
range_start = 12
range_end = 13
# # model_types = ["transformer"]
# # model_types = ["lstm_with_attention", "just_attention", "transformer"]
# #full list mode all models
model_types = ["lstm_with_attention_d2v_dbow", 'lstm_with_attention_d2v_dm', 'lstm_with_attention_w2v_cbow', 'lstm_with_attention_w2v_sg', 'lstm_with_attention_ft_sg', 'lstm_with_attention_ft_cbow',
               "just_attention_d2v_dbow", 'just_attention_d2v_dm', 'just_attention_w2v_cbow', 'just_attention_w2v_sg', 'just_attention_ft_sg', 'just_attention_ft_cbow',
               "transformer_d2v_dbow", 'transformer_d2v_dm', 'transformer_w2v_cbow', 'transformer_w2v_sg', 'transformer_ft_sg', 'transformer_ft_cbow',
               'vanilla_lstm_d2v_dbow', 'vanilla_lstm_d2v_dm', 'vanilla_lstm_w2v_cbow', 'vanilla_lstm_w2v_sg', 'vanilla_lstm_ft_sg', 'vanilla_lstm_ft_cbow',
               "countvectorizer", "doc2vec_dbow", "doc2vec_dm", "word2vec_cbow", "word2vec_sg", "fasttext_sg", "fasttext_cbow"]            
# model_types = ["transformer"]       
gpu_index = 7  # Change this to select a different GPU     

# # model_types = ["transformer"]
############################################
#Import all csv file under Train_data folder
############################################

folder_path = "Train_data"
# List to hold DataFrames
dataframes = []
# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

combined_data = pd.concat(dataframes, ignore_index=True)
combined_data.to_csv('combined_train_data.csv', index=False)
file_path = os.path.join('combined_train_data.csv')
data = pd.read_csv(file_path)



############################
#import word embedding model
############################
K = 5
known_unknown = False

# Define the base path for the models
base_path = "Embedding_Models"

# Use os.path.join to construct the file paths
model_dm_path = os.path.join(base_path, "doc2vec_dm")
model_dbow_path = os.path.join(base_path, "doc2vec_dbow")
model_cbow_path = os.path.join(base_path, "word2vec_cbow")
model_sg_path = os.path.join(base_path, "word2vec_sg")
model_fasttext_sg_path = os.path.join(base_path, "fasttext_sg")
model_fasttext_cbow_path = os.path.join(base_path, "fasttext_cbow")

# Load the models
model_dm = gensim.models.doc2vec.Doc2Vec.load(model_dm_path)
model_dbow = gensim.models.doc2vec.Doc2Vec.load(model_dbow_path)
model_cbow = gensim.models.word2vec.Word2Vec.load(model_cbow_path)
model_sg = gensim.models.word2vec.Word2Vec.load(model_sg_path)
model_fasttext_sg = gensim.models.word2vec.Word2Vec.load(model_fasttext_sg_path)
model_fasttext_cbow = gensim.models.word2vec.Word2Vec.load(model_fasttext_cbow_path)



####################
#Training all models
####################

# Create empty DataFrame with desired columns before the loop
columns = ["num_substrates", "feature_method", "avg_accuracy", "avg_classwise_acc",
           "std_err_avg_acc", "std_err_avg_classwise_acc", "avg_precision", "avg_recall", "avg_f1_score"]
overall_catch = pd.DataFrame(columns=columns)
overall_catch.to_csv('Result summary.csv', index=False)


# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices: ", tf.config.list_physical_devices('GPU'))


# Memory growth configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set visible devices to use only the specified GPU
        tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
        
        # Configure memory growth for the selected GPU
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        
        # Check logical devices after configuration
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Using GPU {gpu_index} out of {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    except IndexError:
        print(f"Error: GPU index {gpu_index} is out of range. You only have {len(gpus)} GPUs available.")


for top_k in tqdm(range(range_start,range_end)):
    for featurizer in model_types:
        print("Currently running for featurizer "+ featurizer + " with " + str(top_k) + " number of classes.")
        if featurizer == "countvectorizer":
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, None)
        elif featurizer == "doc2vec_dbow":
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dbow)
        elif featurizer == "doc2vec_dm":
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dm)
        elif featurizer == "word2vec_cbow":
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_cbow)            
        elif featurizer == "word2vec_sg":
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_sg)
        elif featurizer == "fasttext_sg":
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_sg)      
        elif featurizer == "fasttext_cbow":
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_cbow)  
            
        elif featurizer == "lstm_with_attention_d2v_dbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dbow)  
        elif featurizer == "lstm_with_attention_d2v_dm": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dm)   
        elif featurizer == "lstm_with_attention_w2v_cbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_cbow)        
        elif featurizer == "lstm_with_attention_w2v_sg": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_sg)   
        elif featurizer == "lstm_with_attention_ft_sg": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_sg)
        elif featurizer == "lstm_with_attention_ft_cbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_cbow)   

        
        elif featurizer == "just_attention_d2v_dbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dbow)        
        elif featurizer == "just_attention_d2v_dm": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dm)        
        elif featurizer == "just_attention_w2v_cbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_cbow)        
        elif featurizer == "just_attention_w2v_sg": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_sg)        
        elif featurizer == "just_attention_ft_sg": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_sg)        
        elif featurizer == "just_attention_ft_cbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_cbow)        



        
        elif featurizer == "vanilla_lstm_d2v_dbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dbow)        
        elif featurizer == "vanilla_lstm_d2v_dm": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dm)        
        elif featurizer == "vanilla_lstm_w2v_cbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_cbow)        
        elif featurizer == "vanilla_lstm_w2v_sg": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_sg)        
        elif featurizer == "vanilla_lstm_ft_sg": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_sg)        
        elif featurizer == "vanilla_lstm_ft_cbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_cbow)        


        
        elif featurizer == "transformer_d2v_dbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dbow)        
        elif featurizer == "transformer_d2v_dm": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_dm)        
        elif featurizer == "transformer_w2v_cbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_cbow)        
        elif featurizer == "transformer_w2v_sg": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_sg)        
        elif featurizer == "transformer_ft_sg": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_sg)        
        elif featurizer == "transformer_ft_cbow": 
            avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, trained_model, params_best, fig, fig2, fig3 = run_end_to_end(top_k, data, featurizer, K, known_unknown, model_fasttext_cbow)        
        
        else:
            pass
        
        
        # Save fig1
        fig.savefig(f'plot/figure1_{featurizer}.png', dpi=300, bbox_inches='tight')

        # Save fig2 
        fig2.savefig(f'plot/figure2_{featurizer}.png', dpi=300, bbox_inches='tight')

        # Save fig3
        fig3.savefig(f'plot/figure3_{featurizer}.png', dpi=300, bbox_inches='tight')
        print('training figure saved')

        # Create a new row of results
        new_row = pd.DataFrame([[top_k, featurizer, avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc,
                                overall_report["precision"], overall_report["recall"], overall_report["f1-score"]]],
                              columns=columns)
        
        # Append to CSV file directly
        new_row.to_csv('Result summary.csv', mode='a', header=False, index=False)
        
        print([top_k, featurizer, avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc,
               overall_report["precision"], overall_report["recall"], overall_report["f1-score"]])
        plt.close('all')

# Read the final results for plotting
overall_catch = pd.read_csv('Result summary.csv')

# #create a comparsion plot
plt.figure(figsize = (12,8))
# filter_condn = [ "countvectorizer"]
sns.lineplot(data=overall_catch,  x="num_substrates", y="avg_accuracy", hue="feature_method",  marker="o")
plt.title("Plot of Average Accuracy by number of substrates for different feature extraction methods.", fontsize = 20 ,weight = "bold")
plt.xlabel("Number of Substrates",  weight = "bold", fontsize = 20)
plt.ylabel("Accuracy Value", weight = "bold", fontsize = 20)
plt.xticks(range(7,15), weight = "bold", fontsize = 15)
plt.yticks(weight = "bold", fontsize = 15, rotation = 0)
plt.savefig('Accuracy plot.png')
print('trained model and label encoder were saved')
print('Training complete')