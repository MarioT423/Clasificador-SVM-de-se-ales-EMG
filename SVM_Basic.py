# importar librerias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# leer datos del archivo .csv
data = pd.read_csv('EMG_Signals.csv')

# separando los datos en datos de entrenamiento y de prueba
training_set,test_set = train_test_split(data,test_size=0.2,random_state=1)
#print("train:",training_set)
#print("test:",test_set)

# preaparando datos para aplicar el svm
x_train = training_set.iloc[:,0:2].values  # data
y_train = training_set.iloc[:,2].values  # target
x_test = test_set.iloc[:,0:2].values  # data
y_test = test_set.iloc[:,2].values  # target

# modelo de entrenamiento
classifier = SVC(kernel='rbf',random_state=1,C=1,gamma='auto')
classifier.fit(x_train,y_train)

# realizando prediccion
y_pred = classifier.predict(x_test)

# creando matriz de confusion y calculando la presición
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy = float(cm.diagonal().sum())/len(y_test)
print('model accuracy is:',accuracy*100,'%')
x1_test = [[0.02,0.8]] # para nuevos datos de prueba