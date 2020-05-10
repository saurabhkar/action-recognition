import pandas as pd
import numpy as np
import os
import time
import cv2
import random
import keras
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
from skimage.transform import resize
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Convolution2D , MaxPooling2D, Flatten, Dense , Dropout
from keras.layers import Activation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.regularizers import l2



def make_dataframe():
    path = 'rawframes'
    row = 64
    col = 64
    column_names = ['pixel'+str(i) for i in range(row*col)]
    column_names_with_label = column_names.copy()
    column_names_with_label.append('label')
    numclass=0
    numfol=0
    files=0
    filec=0
    #df = pd.DataFrame(columns=column_names_with_label)
    twod = []
    count = 0

    for class_name in os.listdir(path):
        classcount=0
        if(classcount==35):break
        if(class_name!='.DS_Store'):
            numfol=0
            for fol in os.listdir(path+'/'+class_name):
                if(fol[-9:]!='.DS_Store'):
                    numfol+=1
                    if(numfol==20):break
                    numfil=0
                    for filename in os.listdir(path+'/'+class_name+'/'+fol):
                        numfil+=1
                        if(filename[-9:] != '.DS_Store' and numfil%10==0):
                            
                            img = Image.open(path+'/'+class_name+'/'+fol+'/'+filename).convert('L')
                            img = img.resize((col,row))
                            img = np.array(img).flatten()
                                
                            dict_ = {column_names[i]: img[i] for i in range(len(column_names))}
                            dict_['label'] = class_name
                            twod.append(list(dict_.values()))
                            count+=1
                            print(count)          
           
    df = pd.DataFrame(twod,columns=column_names_with_label)
    df.to_csv('data.csv',index=None)

def hog_data():
    data = pd.read_csv('data.csv')
    hogarr=[]
    
    for index,row in data.iterrows():
        img = np.reshape(np.array(row[:-1],dtype=int),(64,64)).astype('uint8')
        hog_feature, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),block_norm= 'L2', visualize=True, multichannel=False)
        hogarr.append(list(hog_feature))    
    hogdf = pd.DataFrame(hogarr, columns=['pixel'+str(i) for i in range(len(hog_feature))])
    
    extra_column = data['label']
    hogdf = pd.concat([hogdf,extra_column], axis = 1)
   
    hogdf.to_csv('HOGdata.csv',index=None)
  
    hog_y = pd.DataFrame(hogdf , columns =['label'])
   
    hog_le = LabelEncoder()
    hog_y = hog_le.fit_transform(hog_y)
    
    hog_train =hogdf
    hog_train = hog_train.drop('label', axis=1).values
        
    x_tr_svm, x_ts_svm, y_tr_svm, y_ts_svm = train_test_split(hog_train, hog_y, test_size=0.2, shuffle = True)
    x_tr_knn, x_ts_knn, y_tr_knn, y_ts_knn = train_test_split(hog_train, hog_y, test_size=0.2, shuffle = True)
    
    scaler = StandardScaler()

    normalized_x_train_svm = pd.DataFrame(scaler.fit_transform(x_tr_svm))
    normalized_x_test_svm = pd.DataFrame(scaler.fit_transform(x_ts_svm))
   
    normalized_x_train_knn = pd.DataFrame(scaler.fit_transform(x_tr_knn))
    normalized_x_test_knn = pd.DataFrame(scaler.fit_transform(x_ts_knn))
   
    svm = LinearSVC()
    svm.fit(normalized_x_train_svm,y_tr_svm)
    svm_preds = svm.predict(normalized_x_test_svm)
    svm_acc_hog = accuracy_score(y_ts_svm,svm_preds)

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(normalized_x_train_knn,y_tr_knn)
    knn_preds = knn.predict(normalized_x_test_knn)
    knn_acc_hog = accuracy_score(y_ts_knn,knn_preds)

    # print('accuracies of SVM-HOG:{}'.format(svm_acc))
    # print('accuracies of KNN-HOG:{}'.format(knn_acc))
    return(knn_acc_hog,svm_acc_hog)
    
def SIFT_data():
    data = pd.read_csv('data.csv')
    siftarr=[]
    mindescriptors=1000

    for index,row in data.iterrows():
        img = np.reshape(np.array(row[:-1],dtype=int),(64,64)).astype('uint8')
        sift = cv2.xfeatures2d.SURF_create(10)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if(descriptors is not None):
            mindescriptors = min(mindescriptors,descriptors.shape[0])

    for index,row in data.iterrows():
        img = np.reshape(np.array(row[:-1],dtype=int),(64,64)).astype('uint8')
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=mindescriptors)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if(descriptors is not None):
            descriptors=descriptors[:mindescriptors]
            sift_feature = np.array(descriptors).flatten()
            siftarr.append(list(sift_feature))

    siftdf = pd.DataFrame(siftarr,columns=['pixel'+str(i) for i in range(len(sift_feature))])
    
    extra_column = data['label']
    siftdf = pd.concat([siftdf,extra_column], axis = 1)
    siftdf.fillna(siftdf.median(), inplace=True)
    
    siftdf.to_csv('SIFTdata.csv',index=None)
    
    
    sift_y = pd.DataFrame(siftdf , columns =['label'])
   
    sift_le = LabelEncoder()
    sift_y = sift_le.fit_transform(sift_y)
    
    sift_train =siftdf
    sift_train = sift_train.drop('label', axis=1).values
        
    
    
    
    x_tr_svm_sift, x_ts_svm_sift, y_tr_svm_sift, y_ts_svm_sift = train_test_split(sift_train, sift_y, test_size=0.2, shuffle = True)
    x_tr_knn_sift, x_ts_knn_sift, y_tr_knn_sift, y_ts_knn_sift = train_test_split(sift_train, sift_y, test_size=0.2, shuffle = True)
   
    
    scaler = StandardScaler()

    normalized_x_train_sift = pd.DataFrame(scaler.fit_transform(x_tr_svm_sift))
    normalized_x_test_sift = pd.DataFrame(scaler.fit_transform(x_ts_svm_sift))
     
     
    normalized_x_train_knn_sift = pd.DataFrame(scaler.fit_transform(x_tr_knn_sift))
    normalized_x_test_knn_sift = pd.DataFrame(scaler.fit_transform(x_ts_knn_sift))
   
        
    
    
    svm = LinearSVC()
    svm.fit(normalized_x_train_sift,y_tr_svm_sift)
    svm_preds_sift = svm.predict(normalized_x_test_sift)
    svm_acc_sift = accuracy_score(y_ts_svm_sift,svm_preds_sift)

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(normalized_x_train_knn_sift,y_tr_knn_sift)
    knn_preds_sift = knn.predict(normalized_x_test_knn_sift)
    knn_acc_sift = accuracy_score(y_ts_knn_sift,knn_preds_sift)

    # print('accuracies of SVM-SIFT:{}'.format(svm_acc_sift))
    # print('accuracies of KNN-SIFT:{}'.format(knn_acc_sift))
    return (knn_acc_sift, svm_acc_sift)

def cnn_data():
    class myCallback(tf.keras.callbacks.Callback) :
        def on_epoch_end(self, epochs, logs={}) :
            if(logs.get('val_acc') is not None and logs.get('val_acc') >= 0.99) :
                print('\nReached 99% accuracy so cancelling training!')
                self.model.stop_training = True
    callbacks = myCallback()

    # Importing the dataset
    dataset = pd.read_csv('data.csv')
    data_encoded = dataset
    row = len(dataset)
    col = len(dataset.columns)

    number_classes=  data_encoded['label'].value_counts()
    

    le = LabelEncoder()
    data_encoded['label'] = le.fit_transform(data_encoded['label'])
    X = dataset.iloc[:, 0:col-1].values
    y = dataset.iloc[:, col-1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, shuffle=True)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    XX_train = X_train[:,0:].reshape(X_train.shape[0],64,64,1).astype('float32')
        
    XX_train = XX_train / 255.0
    
    
    XX_test = X_test[:,0:].reshape(X_test.shape[0],64,64,1).astype('float32')
    XX_test = XX_test / 255.0
    
    
    classifier = Sequential()
    classifier.add(Convolution2D(64,(3,3),input_shape = (64,64,1)))
    classifier.add(Activation('relu'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(64,(3,3),input_shape = (64,64,1)))
    classifier.add(Activation('relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Dropout(0.2))    
   
    classifier.add(Convolution2D(128,(3,3)))
    classifier.add(Activation('relu'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(128,(3,3)))
    classifier.add(Activation('relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Dropout(0.2))    
    
    
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256))
    classifier.add(Activation('relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.2))    
    
    classifier.add(Dense(output_dim = number_classes.shape[0] , activation= 'softmax'))
    
    classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
    
    classifier.summary()
    classifier.fit(XX_train,y_train,
                   validation_data= (XX_test,y_test),
                   batch_size=32, epochs = 40 , verbose=1, callbacks=[callbacks])

    training_loss,training_acc = classifier.evaluate(XX_train,y_train,verbose=2)
    
    test_loss,test_acc = classifier.evaluate(XX_test,y_test,verbose=2)
  
    return(training_acc,test_acc)


def hog_and_sift_combined():
    ho = pd.read_csv("HOGdata.csv")
    si = pd.read_csv("SIFTdata.csv")
    sift_y = pd.DataFrame(ho , columns =['label'])
    del si['label']
    del ho['label']
    sift_train = pd.concat([ho,si] , axis=1)

    x_tr_svm_sift, x_ts_svm_sift, y_tr_svm_sift, y_ts_svm_sift = train_test_split(sift_train, sift_y, test_size=0.2, shuffle = True)
    x_tr_knn_sift, x_ts_knn_sift, y_tr_knn_sift, y_ts_knn_sift = train_test_split(sift_train, sift_y, test_size=0.2, shuffle = True)
    
    scaler = StandardScaler()
    
    normalized_x_train_sift = pd.DataFrame(scaler.fit_transform(x_tr_svm_sift))
    normalized_x_test_sift = pd.DataFrame(scaler.fit_transform(x_ts_svm_sift))
     
    normalized_x_train_knn_sift = pd.DataFrame(scaler.fit_transform(x_tr_knn_sift))
    normalized_x_test_knn_sift = pd.DataFrame(scaler.fit_transform(x_ts_knn_sift))
    
    svm = LinearSVC()
    svm.fit(normalized_x_train_sift,y_tr_svm_sift)
    svm_preds_sift = svm.predict(normalized_x_test_sift)
    svm_acc_sift_hog = accuracy_score(y_ts_svm_sift,svm_preds_sift)

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(normalized_x_train_knn_sift,y_tr_knn_sift)
    knn_preds_sift = knn.predict(normalized_x_test_knn_sift)
    knn_acc_sift_hog = accuracy_score(y_ts_knn_sift,knn_preds_sift)
    
    return (knn_acc_sift, svm_acc_sift)
    


print('*****************************************************')
print('making dataframe')
make_dataframe()

print('*****************************************************')
print('SIFT keypoints and descriptors extraction')
knn_acc_sift, svm_acc_sift = SIFT_data()

print('*****************************************************')
print('HOG feature extraction')
knn_acc_hog, svm_acc_hog = hog_data()    

print('*****************************************************')
print('SIFT keypoints and descriptors extraction')
knn_acc_sift_hog, svm_acc_sift_hog = hog_and_sift_combined()

print('*****************************************************')
print('CNN')
training_acc,test_acc = cnn_data()



print('accuracies of SVM-SIFT: {:.2f} '.format(svm_acc_sift*100) + '% ')
print('accuracies of KNN-SIFT:{:.2f} '.format(knn_acc_sift*100) + '% ')

print('accuracies of SVM-HOG:{:.2f} '.format(svm_acc_hog*100) + '% ')
print('accuracies of KNN-HOG:{:.2f} '.format(knn_acc_hog*100) + '% ')

print('accuracies of SVM-HOG-SIFT combined:{:.2f} '.format(svm_acc_sift_hog*100) + '% ')
print('accuracies of KNN-HOG-SIFT combined:{:.2f} '.format(knn_acc_sift_hog*100) + '% ')
          

print('Testing accuracy of CNN:{:.2f}'.format(test_acc*100) + '% ')
        
    
    
    
    
#print('Training accuracy of CNN:{}'.format(training_acc))   
    
    