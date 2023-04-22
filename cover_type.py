import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

import os

#%%

 
def get_X_y(feats):
    X = data[ feats ].values
    y = data['target'].values

    return X, y

def draw_learning_curve(data_type, history, key='accuracy'):
 
  fig, ax = plt.subplots(1, 2, figsize=(12,6))
  ax[0].plot(history.history[key])
  ax[0].plot(history.history['val_'+ key] )
  ax[0].set_ylabel(key.title())
  ax[0].set_xlabel('Epoch')
  ax[0].legend(['train', 'val'])

  ax[1].plot(history.history['loss'])
  ax[1].plot(history.history['val_loss'] )
  ax[1].set_ylim([0,1])
  ax[1].set_ylabel('loss'.title())
  ax[1].set_xlabel('Epoch')
  ax[1].legend(['train', 'val'])
  fig.suptitle('Learning curve')
  plt.savefig("./models/fig_learning_curve_" + data_type + "_" + "neural_network.png",dpi = 200)
  return
  
def get_confusion_matrix(model_type, data_type, model, X_test, y_test):
  title_string = 'model: ' + model_type + ' | ' + 'data type: ' + data_type
  print(title_string)
  y_pred = model.predict(X_test)
  
  if(model_type== 'Neural Network'):
    pred_labels=[]
    for idx in range(len(y_pred)):
      pred_label = np.argmax(y_pred[idx])
      pred_labels.append(pred_label)
  else:
      pred_labels = y_pred
  true_labels = y_test
  score = accuracy_score(true_labels, pred_labels)

  ConfusionMatrixDisplay.from_predictions(true_labels, pred_labels, normalize='true')
  plt.savefig("./models/fig_confusion_matrix_" + data_type + "_" + model_type + ".png", dpi = 200)
  return score 

#%%

def train_model(X_train, y_train, X_val, y_val, hyperparameters):
  epochs, batch_size, learning_rate = hyperparameters
  input_size = 54
  model = Sequential([
    Dense(2*input_size, input_dim=input_size, activation='relu'),
    Dense(3*input_size, activation='relu'),
    Dense(4*input_size, activation='relu'),
    Dropout(0.1),
    Dense(num_classes, activation='softmax')
  ])
  
  opt = keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics='accuracy')
  callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
  history = model.fit(X_train, y_train,
          batch_size=batch_size, epochs=epochs, callbacks=[callback], verbose=1,
          validation_data=(X_val, y_val))
  
  best_epoch = np.argmax(history.history['val_accuracy'])
  best_val_accuracy = history.history['val_accuracy'][best_epoch]
  best_val_loss = history.history['val_loss'][best_epoch]
  train_accuracy = history.history['accuracy'][best_epoch]
  train_loss = history.history['loss'][best_epoch]


  return model, history, best_epoch, train_accuracy, train_loss, best_val_accuracy, best_val_loss


def hyperparameters_grid_search(data_type, X_train, y_train, X_val, y_val, batch_sizes, learning_rates, mach_epochs):
  path = "./models/"
  hyperparameters_data = []
  new_val_accuracy = 0
  for idx_lr, lr in enumerate(learning_rates):
    for idx_batch_size, batch_size in enumerate(batch_sizes):
      hyperparameters = [max_epochs, batch_size, lr]
      print("="*10)
      print("="*10)
      print("learning_rate  = " + str(lr) + " | " + str(idx_lr+1) + " / " + str(len(learning_rates)) )
      print("batch_size = " + str(batch_size) + " | " + str(idx_batch_size+1) + " / " + str(len(batch_sizes)) )
      model, training_history, best_epoch, train_accuracy, train_loss, best_val_accuracy, best_val_loss = train_model(X_train, y_train, X_val, y_val, hyperparameters)
      
      if(best_val_accuracy > new_val_accuracy):
        new_val_accuracy = best_val_accuracy
        best_plot, best_lr, best_batch_size = training_history, lr, batch_size
        best_NN_model = model
        if(data_type == 'balanced'):
          best_NN_model.save(path+'NN_model_balanced_batch_size{}_lr{}.h5'.format(best_batch_size, best_lr))
        elif(data_type == 'unbalanced'):
          best_NN_model.save(path+'NN_model_batch_size{}_lr{}.h5'.format(best_batch_size, best_lr))
        else:
          print('DATA TYPE ERROR! You can choose: balanced or unbalanced')


        hyperparameters_dict = {
          'learning_rate' : lr,
          'batch_size' : batch_size,
          'best_val_accuracy': best_val_accuracy,
          'best_val_loss': best_val_loss,
          'train_accuracy': train_accuracy,
          'train_loss': train_loss,
          'best_epoch': best_epoch,
          'max_epochs' : max_epochs,
          'training_history': training_history
        }
        hyperparameters_data.append(hyperparameters_dict)
      
      else:
        hyperparameters_dict = {
          'learning_rate' : lr,
          'batch_size' : batch_size,
          'best_val_accuracy': best_val_accuracy,
          'best_val_loss': best_val_loss,
          'train_accuracy': train_accuracy,
          'train_loss': train_loss,
          'best_epoch': best_epoch,
          'max_epochs' : max_epochs,
          'training_history': training_history
        }
        hyperparameters_data.append(hyperparameters_dict)

  dataframe_training = pd.DataFrame(hyperparameters_data)
  return best_NN_model, dataframe_training, best_plot, best_lr, best_batch_size  



def get_models():
    
    models = [('dummy', DummyClassifier(strategy='stratified')),
        ('decision_tree', DecisionTreeClassifier(max_depth=10)),
        ('random_forest', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)),
    ]
    
    return models

def plot_results(plot_title, result, scoring):

    result = sorted(result, key=lambda x: x[1])

    ys = [i[1] for i in result]
    xs_labels = [i[0] for i in result]
    xs = range(len(xs_labels))
    
    plt.figure(figsize=(15, 5))
    plt.title('best model={}, {}={}'.format(xs_labels[-1], scoring, ys[-1] ), fontsize=14)
    plt.xlabel('models')
    plt.ylabel(scoring)
    plt.bar(xs, ys)
    plt.xticks(xs, xs_labels, rotation=90)
    plt.savefig("./models/fig_" + plot_title + ".png",dpi=200)
    return

def run_models(data_type, X_train, y_train, X_test, y_test, scoring, plot_result=True):
    path = "./models/"
    result = []

    for it, (model_name, model) in enumerate(get_models()):
        
        model.fit(X_train, y_train)       
        test_score = get_confusion_matrix(model_name, data_type, model, X_test, y_test)

        print("model={}, {}: {}".format(model_name, scoring, test_score))

        if(data_type == 'balanced'):
          joblib.dump(model, path + '{}_balanced.joblib'.format(model_name))
        elif(data_type == 'unbalanced'):
          joblib.dump(model, path + '{}.joblib'.format(model_name))
        else:
          print('DATA TYPE ERROR! You can choose: balanced or unbalanced')
        
        result.append((model_name, test_score))
         
    if plot_result:
        plot_title = data_type + "_ML_models"
        plot_results(plot_title, result, scoring)
  
    return result
  
#%%
if(os.path.isdir("models")==False):
    os.system("mkdir models") # create folder for saved models


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',names=['Elevation', 'Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points','Wilderness_Area_1','Wilderness_Area_2','Wilderness_Area_3','Wilderness_Area_4','Soil_Type_1','Soil_Type_2','Soil_Type_3','Soil_Type_4','Soil_Type_5','Soil_Type_6','Soil_Type_7','Soil_Type_8','Soil_Type_9','Soil_Type_10','Soil_Type_11','Soil_Type_12','Soil_Type_13','Soil_Type_14','Soil_Type_15','Soil_Type_16','Soil_Type_17','Soil_Type_18','Soil_Type_19','Soil_Type_20','Soil_Type_21','Soil_Type_22','Soil_Type_23','Soil_Type_24','Soil_Type_25','Soil_Type_26','Soil_Type_27','Soil_Type_28','Soil_Type_29','Soil_Type_30','Soil_Type_31','Soil_Type_32','Soil_Type_33','Soil_Type_34','Soil_Type_35','Soil_Type_36','Soil_Type_37','Soil_Type_38','Soil_Type_39','Soil_Type_40','Cover_Type' ])

data.head()
data.info()
data['Cover_Type'].unique()
#data['Cover_Type'].hist()

data['target'] = data['Cover_Type'].factorize()[0]

feats = ['Elevation',
         'Aspect',
         'Slope',
         'Horizontal_Distance_To_Hydrology',
         'Vertical_Distance_To_Hydrology',
         'Horizontal_Distance_To_Roadways',
         'Hillshade_9am',
         'Hillshade_Noon',
         'Hillshade_3pm', 
         'Horizontal_Distance_To_Fire_Points',
         'Wilderness_Area_1',
         'Wilderness_Area_2',
         'Wilderness_Area_3',
         'Wilderness_Area_4',
         'Soil_Type_1',
         'Soil_Type_2',
         'Soil_Type_3',
         'Soil_Type_4',
         'Soil_Type_5',
         'Soil_Type_6',
         'Soil_Type_7',
         'Soil_Type_8',
         'Soil_Type_9',
         'Soil_Type_10',
         'Soil_Type_11',
         'Soil_Type_12',
         'Soil_Type_13',
         'Soil_Type_14',
         'Soil_Type_15',
         'Soil_Type_16',
         'Soil_Type_17',
         'Soil_Type_18',
         'Soil_Type_19',
         'Soil_Type_20',
         'Soil_Type_21',
         'Soil_Type_22',
         'Soil_Type_23',
         'Soil_Type_24',
         'Soil_Type_25',
         'Soil_Type_26',
         'Soil_Type_27',
         'Soil_Type_28',
         'Soil_Type_29',
         'Soil_Type_30',
         'Soil_Type_31',
         'Soil_Type_32',
         'Soil_Type_33',
         'Soil_Type_34',
         'Soil_Type_35',
         'Soil_Type_36',
         'Soil_Type_37',
         'Soil_Type_38',
         'Soil_Type_39',
         'Soil_Type_40']

X, y = get_X_y(feats)


labels = data['target'].unique()
labels_amount = []
for label in labels:
  idx_label = np.where(label == y)[0]
  labels_amount.append(idx_label.shape[0])

idx_class_0 = np.where(label == 0)[0]

amount = 2000

X_balanced = np.zeros((1,54))
y_balanced = np.zeros((1,1))
for label in labels:
  idx_class = np.where(label == y)[0]
  idx_class_balanced = np.random.choice(idx_class, amount)
  X_tmp = X[idx_class_balanced,:]
  y_tmp = y[idx_class_balanced].reshape((amount,1))
 
  X_balanced = np.vstack((X_balanced, X_tmp))
  y_balanced = np.vstack((y_balanced, y_tmp))

y_balanced = y_balanced.reshape((y_balanced.shape[0],))

X_balanced = np.delete(X_balanced, 0, 0)
y_balanced = np.delete(y_balanced, 0, 0)


idx_shuffled = np.arange(1, y_balanced.shape[0])
np.random.shuffle(idx_shuffled)


X_balanced = X_balanced[idx_shuffled, :]
y_balanced = y_balanced[idx_shuffled]

#plt.hist(y_balanced, bins='auto')
#plt.show()


X_balanced_train, X_balanced_val, y_balanced_train, y_balanced_val = train_test_split(X_balanced, y_balanced, test_size=0.2)
X_balanced_val, X_balanced_test, y_balanced_val, y_balanced_test = train_test_split(X_balanced_val, y_balanced_val, test_size=0.5)

y_balanced_cat_train = to_categorical(y_balanced_train)
y_balanced_cat_val = to_categorical(y_balanced_val)
y_balanced_cat_test = to_categorical(y_balanced_test)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5)

y_cat_train = to_categorical(y_train)
y_cat_val = to_categorical(y_val)
y_cat_test = to_categorical(y_test)


num_classes = len(data['target'].unique())



# Train classical ML models

result = run_models('unbalanced', X_train, y_train, X_test, y_test, scoring='accuracy')
result_balanced = run_models('balanced', X_balanced_train, y_balanced_train, X_balanced_test, y_balanced_test,scoring='accuracy', plot_result=True)

#%%

os.system("cp ./models/dummy.joblib ./models/dummy_best.joblib")
os.system("cp ./models/decision_tree_balanced.joblib ./models/decision_tree_best.joblib")
os.system("cp ./models/random_forest_balanced.joblib ./models/random_forest_best.joblib")

#%%

#% Train Neural Network
batch_sizes = [256, 512, 1024]
learning_rates = [0.001, 0.0005, 0.0001]
max_epochs = 100

best_NN_model, dataframe_training, best_plot, best_lr, best_batch_size = hyperparameters_grid_search('unbalanced', X_train, y_cat_train, X_val, y_cat_val, batch_sizes, learning_rates, max_epochs)

dataframe_training.to_pickle("./models/history_traininig_unbalanced_data.pkl");
dataframe_training.head()

print('HYPERPARAMETERS | Bach_size: {} Learning_rate: {} '.format(best_batch_size, best_lr))
draw_learning_curve("unbalanced", best_plot)

NN_score = get_confusion_matrix('Neural Network', 'unbalanced', best_NN_model, X_test, y_test)



best_NN_model_balanced, dataframe_training_balanced, best_plot_balanced, best_lr_balanced, best_batch_size_balanced = hyperparameters_grid_search('balanced', X_balanced_train, y_balanced_cat_train, X_balanced_val, y_balanced_cat_val, batch_sizes, learning_rates, max_epochs)

dataframe_training_balanced.to_pickle("./models/history_traininig_balanced_data.pkl")
dataframe_training_balanced.head()
print('HYPERPARAMETERS | Bach_size_balanced: {} Learning_rate_balanced: {}'.format(best_batch_size_balanced, best_lr_balanced))
draw_learning_curve("balanced", best_plot_balanced)
NN_score_balanced = get_confusion_matrix('Neural Network', 'balanced', best_NN_model_balanced, X_balanced_test, y_balanced_test)
best_NN_model.save('./models/neural_network_best.h5')

# Models comparision
models_score = [["Dummy_model", result[0][1]] ,
                ['Decision_tree', result[1][1]],
                ['Random_forest', result[2][1]],
                ['NN_model', NN_score]] 

plot_results("unbalanced_all_models", models_score, 'accuracy')

models_score_balanced = [["Dummy_model", result_balanced[0][1]] ,
                ['Decision_tree', result_balanced[1][1]],
                ['Random_forest', result_balanced[2][1]],
                ['NN_model', NN_score_balanced]] 

plot_results("balanced_all_models",models_score_balanced, 'accuracy')

