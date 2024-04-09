# Import packages
# Generic
import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "1" # Set the GPU ID
device ="cuda"
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings, json
warnings.filterwarnings('ignore')

# TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Transformer Models
from transformers import BertTokenizer, TFAutoModel

# SKLearn Library
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef # Compute the Matthews correlation coefficient (MCC)

# data aug libraries
import nlpaug.augmenter.word as naw
import random


# data augmentation: sentences swapping
def augment_text_swapping(train_samples):
    new_examples = []
    ## data augmentation loop for the positive class
    train_samples_p = []
    for ele in train_samples:
        if ele[2] == 1:
            train_samples_p.append(ele)
    for sample in train_samples_p:
            text1 = sample[0]
            text2 = sample[1]
            label_id = sample[2]
            new_examples.append([text2, text1,label_id])
    augmented_train_samples = train_samples+new_examples
    # random shuffling
    random.shuffle(augmented_train_samples)
    df = pd.DataFrame (augmented_train_samples, columns = ['sentence1','sentence2','gold_label'])
    return df


# data augmentation: back translation
def augment_text_back_translation(train_samples):
    # models used for translation 
    to_model_dir2 = "facebook/wmt19-en-de"
    from_model_dir2 = "facebook/wmt19-de-en"
    from_model_dir1 = "Helsinki-NLP/opus-mt-de-ar"
    to_model_dir1 = "Helsinki-NLP/opus-mt-ar-de"
    back_translation_aug1 = naw.BackTranslationAug(from_model_name=from_model_dir1, to_model_name=to_model_dir1)
    back_translation_aug2 = naw.BackTranslationAug(from_model_name=from_model_dir2, to_model_name=to_model_dir2)
    new_examples = []
    ## data augmentation loop for the positive class
    train_samples_p = []
    for ele in train_samples:
        if ele[2] == 1:
            train_samples_p.append(ele)
    for sample in train_samples_p:
            text1 = sample[0]
            text2 = sample[1]
            augmented_text1 = back_translation_aug1.augment(text1)[0]
            augmented_text2 = back_translation_aug1.augment(text2)[0]
            label_id = sample[2]
            new_examples.append([augmented_text2, augmented_text1, label_id])
    for sample in train_samples_p:
            text1 = sample[0]
            text2 = sample[1]
            augmented_text3 = back_translation_aug2.augment(text1)[0]
            augmented_text4 = back_translation_aug2.augment(text2)[0]
            label_id = sample[2]
            new_examples.append([augmented_text4, augmented_text3, label_id])
    augmented_train_samples = train_samples+new_examples
    # random shuffling
    random.shuffle(augmented_train_samples)
    df = pd.DataFrame (augmented_train_samples, columns = ['sentence1','sentence2','gold_label'])
    return df


# augmentation: synonym replacement
def augment_text_synonym_replacement(train_samples):
    aug1 = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased', action="substitute")  # substitute
    new_examples = []
    ## data augmentation loop only for positive class
    train_samples_p = []
    for ele in train_samples:
        if ele[2] == 1:
            train_samples_p.append(ele)
    for sample in train_samples_p:
            text1 = sample[0]
            text2 = sample[1]
            augmented_text1 = aug1.augment(text1)[0]
            augmented_text2 = aug1.augment(text2)[0]
            label_id = sample[2]
            new_examples.append([augmented_text2, augmented_text1,label_id])
    augmented_train_samples = train_samples+new_examples
    # random shuffling
    random.shuffle(augmented_train_samples)
    df = pd.DataFrame (augmented_train_samples, columns = ['sentence1','sentence2','gold_label'])
    return df


# augmentation: all types of augmentation together
def augment_all(train_samples):
    aug1 = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased', action="substitute")  # substitute
    to_model_dir2 = "facebook/wmt19-en-de"
    from_model_dir2 = "facebook/wmt19-de-en"
    from_model_dir1 = "Helsinki-NLP/opus-mt-de-ar"
    to_model_dir1 = "Helsinki-NLP/opus-mt-ar-de"
    back_translation_aug1 = naw.BackTranslationAug(from_model_name=from_model_dir1, to_model_name=to_model_dir1)
    back_translation_aug2 = naw.BackTranslationAug(from_model_name=from_model_dir2, to_model_name=to_model_dir2)
    new_examples = []
    ## data augmentation loop
    train_samples_p = []
    for ele in train_samples:
        if ele[2] == 1:
            train_samples_p.append(ele)
    for sample in train_samples_p:
            text1 = sample[0]
            text2 = sample[1]
            augmented_text1 = aug1.augment(text1)[0]
            augmented_text2 = aug1.augment(text2)[0]
            label_id = sample[2]
            new_examples.append([augmented_text1, augmented_text2,label_id])
    for sample in train_samples_p:
            text1 = sample[0]
            text2 = sample[1]
            augmented_text1 = back_translation_aug1.augment(text1)[0]
            augmented_text2 = back_translation_aug1.augment(text2)[0]
            label_id = sample[2]
            new_examples.append([augmented_text2, augmented_text1, label_id])
    for sample in train_samples_p:
            text1 = sample[0]
            text2 = sample[1]
            augmented_text3 = back_translation_aug2.augment(text1)[0]
            augmented_text4 = back_translation_aug2.augment(text2)[0]
            label_id = sample[2]
            new_examples.append([augmented_text3, augmented_text4, label_id])
    augmented_train_samples = train_samples+new_examples
    # random shuffling
    random.shuffle(augmented_train_samples)
    df = pd.DataFrame (augmented_train_samples, columns = ['sentence1','sentence2','gold_label'])
    return df


# Build the complete model with tensorflow (language model + classification head on top)
def build_model(transformer, max_len):
        transformer_encoder = TFAutoModel.from_pretrained(transformer)  #Pretrained language Transformer Model
        input_layer = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")
        sequence_output = transformer_encoder(input_layer)[0]
        cls_token = sequence_output[:, 0, :]
        output_layer = Dense(1, activation='sigmoid')(cls_token)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            Adam(lr=1e-5), 
            loss='binary_crossentropy', 
            metrics=[tf.keras.metrics.Recall()] 
        )
        return model
    
    
    
    
def main():

    # Transformer Model Name that are going to be used
    #transformer_model = 'bert-base-multilingual-cased'
    #transformer_model = 'bert-base-multilingual-uncased'
    #transformer_model = 'distilbert-base-multilingual-cased'
    #transformer_model = "distilbert-base-german-cased"
    #transformer_model = "uklfr/gottbert-base"
    #transformer_model = "deepset/gelectra-base-germanquad"
    #transformer_model = "deepset/gelectra-base"
    #transformer_model = "bert-base-german-cased"
    transformer_model = "deepset/gbert-base"   # Electra is peforming the best, it is the main model

    # Define Tokenizer
    tokenizer = BertTokenizer.from_pretrained(transformer_model)
    # Define Max Length
    max_len = 80   # << change if you wish
    # Batch size and epochs
    AUTO = tf.data.experimental.AUTOTUNE
    batch_size = 16
    epochs = 4  
    # Input files
    csv_file = "antithesis_phrases_annotated.csv"
    # Load Training Data
    dataset1 = pd.read_csv(csv_file)
    # Train-test split
    train_data, test_data = train_test_split(dataset1, test_size=0.2, random_state=42)
    train = train_data[['sentence1','sentence2']].values.tolist()
    test = test_data[['sentence1','sentence2']].values.tolist()

    # Augment data
    # train_samples = train_data[['sentence1','sentence2', "gold_label"]].values.tolist()
    # train_data_augmented = augment_all(train_samples)
    # train = train_data_augmented[['sentence1','sentence2']].values.tolist()
    # test = test_data[['sentence1','sentence2']].values.tolist()

    # Encode the training & test data 
    train_encode = tokenizer.batch_encode_plus(train, pad_to_max_length=True, max_length=max_len)
    test_encode = tokenizer.batch_encode_plus(test, pad_to_max_length=True, max_length=max_len)
    # Split the Training Data into Training (90%) & Validation (10%)
    test_size = 0.1  # << change if you wish
    x_train, x_val, y_train, y_val = train_test_split(train_encode['input_ids'], train_data.gold_label.values, test_size=test_size)
    # Test Data
    x_test = test_encode['input_ids']

    # Loading Data Into TensorFlow Dataset
    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(3072).batch(batch_size).prefetch(AUTO))
    val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(AUTO))
    test_ds = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))

    # Compute weights of the two classes
    train_classes = train_data[["gold_label"]].to_numpy()[:,0]
    class_weights = compute_class_weight( class_weight = "balanced", classes = np.unique(train_classes), y = train_classes)
    class_weights_final = dict(zip(np.unique(train_classes), class_weights))

    # Applying the build model function
    model = build_model(transformer_model, max_len)

    # load weights to  models, when model is pretrained with snli for contradiction detection,
    # checkpoint_path = "saved_model/cp.ckpt"
    # model.load_weights(checkpoint_path)

    # Train model
    n_steps = len(train_data) // batch_size 
    model.fit(train_ds, 
              steps_per_epoch = n_steps, 
              class_weight = class_weights_final,
              validation_data = val_ds,
              epochs = epochs)
    
    # Predictions
    prediction = model.predict(test_ds, verbose=0)
    prediction = prediction>0.5
    y_true = test_data[["gold_label"]]

    # Compute metrics
    f1_sco = f1_score(y_true, prediction) # average = 'macro' 'weighted'
    f1_sco_weighted = f1_score(y_true, prediction, average='weighted') # average = 'macro' 'weighted'
    precision = precision_score(y_true, prediction)
    recall= recall_score(y_true, prediction)
    acc = accuracy_score(y_true, prediction)
    avgp = average_precision_score(y_true, prediction)
    conf_mat = confusion_matrix(y_true, prediction, labels=[1, 0])

    # Print results
    print("F1:                 {:.2f}".format(f1_sco * 100))
    print("F1_weighetd:                 {:.2f}".format(f1_sco_weighted * 100))
    print("Precision:          {:.2f}".format(precision * 100))
    print("Recall:             {:.2f}".format(recall * 100))
    print("Accuracy:           {:.2f}".format(acc * 100))
    print("Average precision:           {:.2f}".format(avgp * 100))
    print("confMatrix")
    print(conf_mat)

    # Save results in json
    diction = {}
    diction["f1"] = round(f1_sco*100, 2)
    diction["precision"] = round(precision*100,2)
    diction["recall"] = round(recall*100,2)
    diction["avgp"] = round(avgp*100,2) 
    diction["accuracy"] = round(acc*100,2)
    diction["conf_mat"] = conf_mat.tolist()
    mcc = matthews_corrcoef(y_true, prediction)
    diction["MCC"] = mcc
    with open('Metricsdata.json', 'w') as fp:
        json.dump(diction, fp)
    
    
if __name__ == "__main__":
    main()
    