# Guide(draft updated-04-18)

There are four Python scripts listed below. The first three are used for training a new word embedding model \& best_k_model. The last one is used as a prediction tool for unlabeled data. Detailed descriptions are provided below:

## 01_train_word_embedding_model.py

This script uses supervised and unsupervised sequences to train the word embedding model, which will be used in the next script. If there are new supervised/unsupervised sequences available in the future, you can simply place them into the specific folder under the `Data_word_embedding` folder and rerun this script. The data format needs to follow the same structure, where the first column is the CGC ID and the second column is the sequence. The column name does not matter. After running this script, the word embedding model will be saved in the `Embedding_Models` folder. You don't need to create this folder ahead of time; the script will create it for you.



## 02_find_best_k_model.py
This script is used to train the subfinder model. Before training, please place all labeled sequence data into the folder named `Train_data`. It has to follow the same structure as before, where the first column is the CGC ID and the second column is the sequence. The script will read all the CSV files under this folder and combine them together.

This script also allows you to choose the exact model and the number range of substrates you want to train. To do that, first uncomment row 24-43 and comment row 49-59. Use 

```bash
python 02_find_best_k_model.py --help
```

for more detailed instructions. For example, if you want to train a subfinder to identify the top 8 substrate classes using a transformer model:

```bash
python 02_find_best_k_model.py -n 7 -m transformer
```

## 02_train_best_model.py
This script is used for train the best model-convectorizer.


## 03_predict.py
This script uses the pretrained subfinder model to predict the substrate class based on the output of dbCAN. You need to specify the input file name and the output summary result name. For example, if you have the output from dbCAN named 'cgc_standard.out' and you want the output file named 'predict_summary.csv':

```bash
python 03_predict.py -i cgc_standard.out -o predict_summary.csv
```

Where:
- `-i` represents the input file
- `-o` represents the output file

The results will contain three columns:
1. CGC ID
2. Name of the fiber substrate
3. Score value (ranges from 0 to 1)

The higher the score value, the higher the possibility that this CGC ID belongs to this class.


