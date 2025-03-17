from Codes.Unsupervised_Preparer import read_the_data
import os
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import gensim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



'''
This is the code for training word embedding model. 
Before starts, please make the the work directory is consistant.
'''
# os.chdir('path/to/your/work/dict')
#########################
#import unsupervised file
#########################
# Path to the folder containing the CSV files
input_folder = os.path.join("Data_word_embedding", "Unsupervised_Sequences")
output_file = os.path.join("Data_word_embedding", "Output", "all_unsupervised.csv")
read_the_data(input_folder, output_file)
print("Combined unsupervised CSV saved to:", output_file)
output_file_path = os.path.join("Data_word_embedding", "Output", "all_unsupervised.csv")
updated_data_unsupervised = pd.read_csv(output_file_path)
updated_data_unsupervised = updated_data_unsupervised.sample(frac = 1.0).reset_index(drop = True)
updated_data_unsupervised = updated_data_unsupervised.drop_duplicates()
## prepare the unsupervised data as gensim expects
gene_list = [str(seq).replace("|", ",").split(",") for seq in updated_data_unsupervised["sequence"]]
gene_list_tagged = [gensim.models.doc2vec.TaggedDocument(seq_list, [i]) for i, seq_list in enumerate(gene_list)] 

#######################
#import supervised file
#######################

# Path to the folder containing the CSV files
input_folder_s = os.path.join("Data_word_embedding", "Supervised_Sequences")
output_file_s = os.path.join("Data_word_embedding", "Output", "all_supervised.csv")
read_the_data(input_folder_s, output_file_s)
print("Combined supervised CSV saved to:", output_file_s)
output_file_path = os.path.join("Data_word_embedding", "Output", "all_supervised.csv")
supervised_data = pd.read_csv(output_file_path)
supervised_data.columns = ["PULID", "sequence"]
supervised_with_unsupervised_seqs = pd.DataFrame(pd.concat([updated_data_unsupervised["sequence"], supervised_data["sequence"]], ignore_index = True))
supervised_with_unsupervised_seqs.columns = ["sequence"]
supervised_with_unsupervised_seqs["sequence"] = [seq.replace("|", ",").replace(",", " ") for seq in supervised_with_unsupervised_seqs["sequence"]]
# supervised_with_unsupervised_seqs["sequence"].to_csv(r"Data//Output//Unsupervised_10_12//all_unsupervised_text.txt", header=None, index=None, sep=' ', mode='a')
output_file_path_txt = os.path.join("Data_word_embedding", "Output", "all_unsupervised_text.txt")
np.savetxt(output_file_path_txt, supervised_with_unsupervised_seqs.values, fmt='%s')
## count number of unique genes in the supervised dataset
vec_size = len(np.unique([gene for seq in supervised_data["sequence"] for gene in seq.replace("|", ",").split(",")]))
vec_size = np.min((300, vec_size))



##########################
#train the embedding model
##########################
#create a new folder
os.makedirs('Embedding_Models', exist_ok=True)
## doc2vec_dbow
doc2vec_dbow = gensim.models.doc2vec.Doc2Vec(corpus_file=output_file_path_txt, 
                                           vector_size=vec_size, min_count=5, epochs=60, workers = 7, dm = 0, 
                                      dbow_words = 0, window = 7)
model_path_dbow = os.path.join("Embedding_Models", "doc2vec_dbow")
doc2vec_dbow.save(model_path_dbow)


## word2vec_cbow
word2vec_cbow = gensim.models.Word2Vec(corpus_file=output_file_path_txt, 
                                           vector_size=vec_size, window = 7, min_count = 5, max_vocab_size = None,
                                           sg = 0, workers = 7, epochs=60)

model_path_cbow = os.path.join("Embedding_Models", "word2vec_cbow")
word2vec_cbow.save(model_path_cbow)


## word2vec_sg
word2vec_sg = gensim.models.Word2Vec(corpus_file=output_file_path_txt, 
                                           vector_size=vec_size, window = 7, min_count = 5, max_vocab_size = None, sg = 1,
                                           workers = 7, epochs=60)
model_path_sg = os.path.join("Embedding_Models", "word2vec_sg")
word2vec_sg.save(model_path_sg)

## fasttext_cbow
fasttext_cbow = gensim.models.fasttext.FastText(corpus_file=output_file_path_txt, 
                                           vector_size=vec_size, window = 7, min_count = 5, max_vocab_size = None, sg = 0,
                                           workers = 6, epochs=60)
model_path_fcbow = os.path.join("Embedding_Models", "fasttext_cbow")
fasttext_cbow.save(model_path_fcbow)

## fasttext_sg
fasttext_sg = gensim.models.fasttext.FastText(corpus_file=output_file_path_txt, 
                                           vector_size=vec_size, window = 7, min_count = 5, max_vocab_size = None, sg = 1,
                                           workers = 6, epochs=60)
model_path_fsg = os.path.join("Embedding_Models", "fasttext_sg")
fasttext_sg.save(model_path_fsg)


## doc2vec_dbow
doc2vec_dm = gensim.models.doc2vec.Doc2Vec(corpus_file=output_file_path_txt, 
                                           vector_size=vec_size, min_count=5, epochs=60, workers = 7, dm = 1, 
                                      dbow_words = 0, window = 7)
model_path_dm = os.path.join("Embedding_Models", "doc2vec_dm")
doc2vec_dm.save(model_path_dm)

print("all word embedding model saved under Embedding_Models folder")