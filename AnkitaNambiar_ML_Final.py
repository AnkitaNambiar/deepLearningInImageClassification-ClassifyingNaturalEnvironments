#!/usr/bin/env python
# coding: utf-8

# ## Ankita Nambiar
# ## ML Final: Classifying Environments: Using Deep Learning in Image Classification

# In[82]:


import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from tqdm import tqdm


# ### Load Data
# ### Initial Preprocessing: Image Size + Color + NP Array Convert

# In[83]:


class_names = ['mountain', 'street', 'glacier', 'building', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}


# In[84]:


def loading_data():
    
    loadings = ['/Users/ankita7/Desktop/ML_data/seg_train', '/Users/ankita7/Desktop/ML_data/seg_test']
    data = []
    
    for loading in loadings:
        
        images = []
        labels = []
        print("Loading {}".format(loading))
        
        for folder in os.listdir(loading):
            if folder in class_names_label:
                label = class_names_label[folder]
            else:
                print(f"Skipped folder: {folder} (not found in class_names_label)")
                continue
            
            for file in tqdm(os.listdir(os.path.join(loading, folder))):
                img_path = os.path.join(os.path.join(loading, folder), file)
                
                image = cv2.imread(img_path)
                # Image Color 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Image Size 
                IMAGE_SIZE = (150, 150)
                image = cv2.resize(image, IMAGE_SIZE) 
                
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')   
        
        data.append((images, labels))

    return data


# In[85]:


(test_images, test_labels), (train_images, train_labels) = loading_data()


# In[86]:


train_images, train_labels = shuffle(train_images, train_labels, random_state=25)


# ### Scale Images

# In[87]:


train_images = train_images / 255.0 
test_images = test_images / 255.0


# ### Initial Data Analysis

# In[88]:


np.unique(train_labels)


# In[89]:


#train_images, train_labels = shuffle(train_images, train_labels, random_state=25)


# In[90]:


#n_train = train_labels.shape[0]
#n_test = test_labels.shape[0]

#print ("Number of training examples: {}".format(n_train))
#print ("Number of testing examples: {}".format(n_test))
#print ("Each image is of size: {}".format(IMAGE_SIZE))


# In[91]:


train_labels.shape


# In[92]:


test_labels.shape


# In[93]:


train_images.shape


# In[94]:


test_images.shape


# ### Check Class Balance: Balanced

# In[95]:


_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)

fig, ax = plt.subplots(figsize=(8, 5))

x = range(len(class_names))

ax.bar(x, train_counts, label='Train')
ax.bar(x, test_counts, label='Test', alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')

ax.set_ylabel('Counts')

ax.set_title('Class Counts in Train and Test Sets')
ax.legend()

plt.show()


# ### Check Visual Data

# In[96]:


def display_examples(class_names, images, labels, num_per_class=5):

    unique_labels = np.unique(labels)  

    fig, axes = plt.subplots(len(unique_labels), num_per_class, figsize=(12, 12))

    for i, label in enumerate(unique_labels):
        class_indices = np.where(labels == label)[0]  
        np.random.shuffle(class_indices) 
        class_indices = class_indices[:num_per_class]  

        for j, idx in enumerate(class_indices):
            ax = axes[i, j]
            ax.imshow(images[idx], cmap=plt.cm.binary)
            ax.grid(False)
            ax.axis('off')
            ax.set_title(class_names[label])

    plt.subplots_adjust(wspace=0.2, hspace=0.4) 

    plt.show()

display_examples(class_names, train_images, train_labels, num_per_class=5)


# ### Modeling: Testing Different Ones

# #### Simple CNN (NO Transfer Learning)

# In[101]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation= 'softmax')
])


# In[102]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[103]:


model.summary()


# In[104]:


history = model.fit(train_images, train_labels, batch_size=128, epochs=5, validation_split = 0.2)


# In[105]:


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(8, 5))
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Simple CNN: Model Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.show()


# In[106]:


train_loss, train_accuracy = model.evaluate(train_images, train_labels)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# #### Transfer Learning with VGG16

# In[107]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers

# Load pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


# In[108]:


v_model = models.Sequential()
v_model.add(base_model)
v_model.add(layers.Flatten()) 
v_model.add(layers.Dense(256, activation='relu'))
v_model.add(layers.Dropout(0.5))
v_model.add(layers.Dense(6, activation='softmax'))


# In[109]:


v_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[110]:


v_model.summary()


# In[111]:


history = v_model.fit(train_images, train_labels, batch_size=128, epochs=5, validation_split = 0.2)


# In[112]:


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(8, 5))
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('VGG16: Model Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.show()


# In[113]:


train_loss, train_accuracy = v_model.evaluate(train_images, train_labels)
test_loss, test_accuracy = v_model.evaluate(test_images, test_labels)

print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# #### Transfer Learning with InceptionResNetV2

# In[114]:


base_model3 = tf.keras.applications.InceptionResNetV2(
                     include_top=False,
                     weights='imagenet',
                     input_shape=(150, 150, 3)
                     )
 
base_model3.trainable=False


# In[138]:


inc_model = tf.keras.Sequential([
        base_model3,  
        tf.keras.layers.BatchNormalization(renorm=True),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])


# In[139]:


inc_model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[140]:


inc_model.summary()


# In[141]:


history= inc_model.fit(train_images, train_labels, batch_size=128, epochs=10, validation_split = 0.2)


# In[143]:


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(8, 5))
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('InceptionResNetV2: Model Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.legend()
plt.show()


# In[144]:


train_loss, train_accuracy = inc_model.evaluate(train_images, train_labels)
test_loss, test_accuracy = inc_model.evaluate(test_images, test_labels)

print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# #### Transfer Learning with ResNet50

# In[121]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


# In[122]:


base_model_res = ResNet50(include_top=False, 
                          weights='imagenet', input_shape=(150, 150, 3))


# In[123]:


base_model_res.trainable = False


# In[124]:


r_model = Sequential()
r_model.add(base_model_res)
r_model.add(Flatten())

r_model.add(Dense(256, activation='relu'))
r_model.add(Dropout(0.5))
r_model.add(Dense(6, activation='softmax'))  


# In[125]:


r_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[126]:


r_model.summary()


# In[127]:


history = r_model.fit(train_images, train_labels, batch_size=128, epochs=5, validation_split = 0.2)


# In[128]:


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(8, 5))
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('ResNet50: Model Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.legend()
plt.show()


# In[129]:


train_loss, train_accuracy = r_model.evaluate(train_images, train_labels)
test_loss, test_accuracy = r_model.evaluate(test_images, test_labels)

print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ### Hyperparameter Tuning

# #### Chosen Model: InceptionResNetV2 because of highest accuracy

# In[130]:


get_ipython().system('pip install keras_tuner')


# In[145]:


import keras_tuner
from keras_tuner.tuners import RandomSearch


# In[146]:


def build_model(hp):
    base_model = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(150, 150, 3)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,  
        tf.keras.layers.BatchNormalization(renorm=True),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu'),
        tf.keras.layers.Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.01, 0.1]))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# In[147]:


random_tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1
)


# In[154]:


random_tuner.search(train_images, train_labels, epochs=10, validation_split=0.2)


# In[155]:


best_model = random_tuner.get_best_models(num_models=1)[0]


# In[156]:


x = random_tuner.get_best_hyperparameters()[0]
x.values


# In[157]:


history = best_model.fit(train_images, train_labels, epochs=10, validation_split=0.2)


# In[158]:


train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(train_loss, 'b', label='Training Loss')
plt.plot(val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(train_accuracy, 'b', label='Training Accuracy')
plt.plot(val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.show()


# In[159]:


train_loss, train_accuracy = best_model.evaluate(train_images, train_labels)
test_loss, test_accuracy = best_model.evaluate(test_images, test_labels)

print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

