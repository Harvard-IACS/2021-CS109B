import os
import requests
import zipfile
import tarfile
import shutil
import json
import time
import numpy as np
import collections
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.layer_utils import count_params
from transformers import TFBertForSequenceClassification
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.ticker as mticker

def download_file(packet_url, base_path="", extract=False, headers=None):
    if base_path != "":
        if not os.path.exists(base_path):
            os.mkdir(base_path)
    packet_file = os.path.basename(packet_url)
    with requests.get(packet_url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(os.path.join(base_path, packet_file), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    if extract:
        if packet_file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(base_path, packet_file)) as zfile:
                zfile.extractall(base_path)
        else:
            packet_name = packet_file.split('.')[0]
            with tarfile.open(os.path.join(base_path, packet_file)) as tfile:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tfile, base_path)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def save_model(model, path="models", model_name="model01"):
    # Ensure path exists
    if not os.path.exists(path):
        os.mkdir(path)

    if isinstance(model, TFBertForSequenceClassification):
        # model.save_pretrained(path)
        model.save_weights(os.path.join(path, model_name + ".h5"))
    else:
        # Save the enitire model (structure + weights)
        model.save(os.path.join(path, model_name + ".hdf5"))

        # Save only the weights
        model.save_weights(os.path.join(path, model_name + ".h5"))

        # Save the structure only
        model_json = model.to_json()
        with open(os.path.join(path, model_name + ".json"), "w") as json_file:
            json_file.write(model_json)


def get_model_size(path="models", model_name="model01"):
    model_size = os.stat(os.path.join(path, model_name + ".h5")).st_size
    return model_size


def evaluate_save_model(model, test_data, training_results, execution_time, learning_rate, epochs, save=True):
    # Get the model train history
    model_train_history = training_results.history
    # Get the number of epochs the training was run for
    num_epochs = len(model_train_history["loss"])

    # Plot training results
    fig = plt.figure(figsize=(20, 5))
    axs = fig.add_subplot(1, 2, 1)
    axs.set_title('Loss')
    # Plot all metrics

    for metric in ["loss", "val_loss"]:        
        axs.plot(np.arange(0, num_epochs,1), model_train_history[metric], label=metric)
        axs.xaxis.set_major_locator(mticker.MaxNLocator(num_epochs))
        ticks_loc = axs.get_xticks().tolist()
        axs.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        label_format = '{:,.0f}'
        axs.set_xticklabels([label_format.format(x) for x in ticks_loc]);
        
    axs.legend()

    axs = fig.add_subplot(1, 2, 2)
    axs.set_title('Accuracy')
    # Plot all metrics
    for metric in ["accuracy", "val_accuracy"]:
        axs.plot(np.arange(0, num_epochs,1), model_train_history[metric], label=metric)
        axs.xaxis.set_major_locator(mticker.MaxNLocator(num_epochs))
        ticks_loc = axs.get_xticks().tolist()
        axs.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        label_format = '{:,.0f}'
        axs.set_xticklabels([label_format.format(x) for x in ticks_loc]);
    axs.legend()

    plt.show()

    # Evaluate on test data    
    evaluation_results = model.evaluate(test_data)
    ytrue = np.concatenate([y for x, y in test_data], axis=0)    
    if isinstance(model, TFBertForSequenceClassification):        
        preds = model.predict(test_data)
        preds = np.argmax(preds['logits'],axis=1)
        evaluation_results.append(f1_score(ytrue,preds))
    else:        
        preds = model.predict(test_data).flatten()
        evaluation_results.append(f1_score(ytrue,preds>0.5))
        
    print(evaluation_results)

    if save:
        # Save model
        save_model(model, model_name=model.name)
        model_size = get_model_size(model_name=model.name)

        # Save model history
        with open(os.path.join("models", model.name + "_train_history.json"), "w") as json_file:
            json_file.write(json.dumps(model_train_history, cls=JsonEncoder))

        trainable_parameters = count_params(model.trainable_weights)
        non_trainable_parameters = count_params(model.non_trainable_weights)

        # Save model metrics
        metrics = {
            "trainable_parameters": trainable_parameters,
            "execution_time": execution_time,
            "loss": evaluation_results[0],
            "accuracy": evaluation_results[1],
            "f1_score": evaluation_results[2],
            "model_size": model_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "name": model.name,
            "id": int(time.time())
        }
        with open(os.path.join("models", model.name + "_metrics.json"), "w") as json_file:
            json_file.write(json.dumps(metrics, cls=JsonEncoder))

# Util function to build dataset
def build_dataset(words, vocab_size):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def find_similar(words, dictionary, reverse_dictionary, embedding_layer_weights, topn=5):
    subset_word_index  = []
    for word in words:
        subset_word_index.append(dictionary[word])

    cs_op = cosine_similarity(embedding_layer_weights[subset_word_index], embedding_layer_weights)

    similar_embeddings = []
    similar_labels = []
    for idx in range(len(words)):
        top = cs_op[idx].argsort()[-topn:][::-1]
        for i,t in enumerate(top):
            similar_embeddings.append(embedding_layer_weights[t])
            similar_labels.append(reverse_dictionary[t])

    return similar_embeddings, similar_labels