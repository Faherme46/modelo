import cv2
import os
import numpy as np





batch_size = 32
img_height = 150
img_width = 150


dataPath = './data' #Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)


labels = []
facesData = []
label = 0

def preprocess_image(image):
    # Histogram equalization
    equalized_image = cv2.equalizeHist(image)
    return equalized_image



for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		image=cv2.imread(personPath+'/'+fileName,0)
		if image is None:
			print(f"Error: Image {fileName} not loaded properly")
		image = preprocess_image(image)
		try:
			resized_image = cv2.resize(image, (img_width, img_height),interpolation= cv2.INTER_CUBIC)
			facesData.append(resized_image)
		except cv2.error as e:
			print(f"Error during resizing {fileName}: {e}")
		
		#image = cv2.imread(personPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1


#print('labels= ',labels)
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

# Métodos para entrenar el reconocedor
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.FisherFaceRecognizer_create()
radius = 1
neighbors = 120
#face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius, neighbors)

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels),)

# Almacenando el modelo obtenido
face_recognizer.write('modeloEigenFace.xml')
face_recognizer.write('modeloFisherFace.xml')
#face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")