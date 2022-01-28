import numpy as np
import os
import glob
import pandas as pd
import re    # To match regular expression for extracting labels
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from unet import get_model

import tensorflow as tf
print("Tensorflow version: ", tf.__version__)


np.random.seed(1111)

def create_random_csv_files(fault_classes, number_of_files_in_each_class):
    os.mkdir("./random_data/")  # Make a directory to save created files.
    for fault_class in fault_classes:
        for i in range(number_of_files_in_each_class):
            data = np.random.rand(1024,)
            file_name = "./random_data/" + eval("fault_class") + "_" + "{0:03}".format(i+1) + ".csv" # This creates file_name
            np.savetxt(eval("file_name"), data, delimiter = ",", header = "V1", comments = "")
        print(str(eval("number_of_files_in_each_class")) + " " + eval("fault_class") + " files"  + " created.")

create_random_csv_files(["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"], number_of_files_in_each_class = 100)

files = glob.glob("./random_data/*")
print("Total number of files: ", len(files))
print("Showing first 10 files...")
print(files[:10])

print(files[0])

print(files[0][14:21])

class CustomSequence(tf.keras.utils.Sequence):  # It inherits from `tf.keras.utils.Sequence` class
  def __init__(self, filenames, batch_size):  # Two input arguments to the class.
        self.filenames= filenames
        self.batch_size = batch_size

  def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

  def __getitem__(self, idx):  # idx is index that runs from 0 to length of sequence
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size] # Select a chunk of file names
        data = []
        labels = []
        label_classes = ["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"]

        for file in batch_x:   # In this loop read the files in the chunk that was selected previously
            temp = pd.read_csv(open(file,'r')) # Change this line to read any other type of file
            data.append(temp.values.reshape(32,32,1)) # Convert column data to matrix like data with one channel
            pattern = "^" + eval("file[14:21]")      # Pattern extracted from file_name
            for j in range(len(label_classes)):
                if re.match(pattern, label_classes[j]): # Pattern is matched against different label_classes
                    labels.append(j)  
        data = np.asarray(data).reshape(-1,32,32,1)
        labels = np.asarray(labels)
        return data, labels


sequence = CustomSequence(filenames = files, batch_size = 10)

print("Sequence lenght ", sequence.__len__())

for num, (data, labels) in enumerate(sequence):
    print(data.shape, labels.shape)
    print(labels)
    if num > 5: break

fault_folders = ["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"]
for folder_name in fault_folders:
    os.mkdir(os.path.join("./random_data", folder_name))

for file in files:
    pattern = "^" + eval("file[14:21]")
    for j in range(len(fault_folders)):
        if re.match(pattern, fault_folders[j]):
            dest = os.path.join("./random_data/",eval("fault_folders[j]"))
            shutil.move(file, dest)

fault_1_files = glob.glob("./random_data/Fault_1/*")
fault_2_files = glob.glob("./random_data/Fault_2/*")
fault_3_files = glob.glob("./random_data/Fault_3/*")
fault_4_files = glob.glob("./random_data/Fault_4/*")
fault_5_files = glob.glob("./random_data/Fault_5/*")


fault_1_train, fault_1_test = train_test_split(fault_1_files, test_size = 20, random_state = 5)
fault_2_train, fault_2_test = train_test_split(fault_2_files, test_size = 20, random_state = 54)
fault_3_train, fault_3_test = train_test_split(fault_3_files, test_size = 20, random_state = 543)
fault_4_train, fault_4_test = train_test_split(fault_4_files, test_size = 20, random_state = 5432)
fault_5_train, fault_5_test = train_test_split(fault_5_files, test_size = 20, random_state = 54321)

fault_1_train, fault_1_val = train_test_split(fault_1_train, test_size = 10, random_state = 1)
fault_2_train, fault_2_val = train_test_split(fault_2_train, test_size = 10, random_state = 12)
fault_3_train, fault_3_val = train_test_split(fault_3_train, test_size = 10, random_state = 123)
fault_4_train, fault_4_val = train_test_split(fault_4_train, test_size = 10, random_state = 1234)
fault_5_train, fault_5_val = train_test_split(fault_5_train, test_size = 10, random_state = 12345)

train_file_names = fault_1_train + fault_2_train + fault_3_train + fault_4_train + fault_5_train
validation_file_names = fault_1_val + fault_2_val + fault_3_val + fault_4_val + fault_5_val
test_file_names = fault_1_test + fault_2_test + fault_3_test + fault_4_test + fault_5_test

# Shuffle training files (We don't need to shuffle validation and test data)
np.random.shuffle(train_file_names)

print("Number of train_files:" ,len(train_file_names))
print("Number of validation_files:" ,len(validation_file_names))
print("Number of test_files:" ,len(test_file_names))

batch_size = 10
train_sequence = CustomSequence(filenames = train_file_names, batch_size = batch_size)
val_sequence = CustomSequence(filenames = validation_file_names, batch_size = batch_size)
test_sequence = CustomSequence(filenames = test_file_names, batch_size = batch_size)


#model = tf.keras.Sequential([
#    layers.Conv2D(16, 3, activation = "relu", input_shape = (32,32,1)),
#    layers.MaxPool2D(2),
#    layers.Conv2D(32, 3, activation = "relu"),
#    layers.MaxPool2D(2),
#    layers.Flatten(),
#    layers.Dense(16, activation = "relu"),
#    layers.Dense(5, activation = "softmax")
#])

img_size = (32, 32)
num_classes = 5
model = get_model(img_size, num_classes)


model.summary()

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(train_sequence, validation_data = val_sequence, epochs = 10)

test_loss, test_accuracy = model.evaluate(test_sequence, steps = 10)

print("Test loss: ", test_loss)
print("Test accuracy:", test_accuracy)


