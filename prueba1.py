import streamlit as st
import numpy as np
import cv2
import os
import imutils
from PIL import Image


dataPath = './data' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)

print('imagePaths=',imagePaths)
if 'my_list' not in st.session_state:
    st.session_state['my_list'] = []

def success(name):
    LISTA=st.session_state.my_list
    if name not in LISTA:
        st.success('Ingresó '+name)
        LISTA.append(name)

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    # Convertir RGB a BGR
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image

def capture_from_file(image_file):
    try:
        pil_image = Image.open(image_file)
        frame = pil_to_cv2(pil_image)
        ret = True
    except Exception as e:
        ret = False
        frame = None
        st.error(f"Error al cargar la imagen: {e}")
    return ret, frame    

def trainModel():
    dataPath = './data' #Cambia a la ruta donde hayas almacenado Data
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)
    labels = []
    facesData = []
    label = 0
    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imágenes')

        for fileName in os.listdir(personPath):
            print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            
        label = label + 1

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrenando el reconocedor de rostros
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels),)
    face_recognizer.write('modeloLBPHFace.xml')
    st.success('Modelo entrenado')
    print("Modelo almacenado...")

def load_model():
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modeloLBPHFace.xml')
    return face_recognizer

def load_image(image_file):
    img = Image.open(image_file)
    ret, frame = capture_from_file(image_file)
    return img

def detect_faces(image):
    # Convert the image to a numpy array
    image_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Load OpenCV pre-trained face detector model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def capture_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def draw_faces(image, faces,frame, only=False):

    # Convert the image to a numpy array
    image_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    

    for (x, y, w, h) in faces:
            
        if only:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:    
            
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(image_np,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

            if result[1] < 70:
                cv2.putText(image_np,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(image_np, (x,y),(x+w,y+h),(0,255,0),2)
                success('{}'.format(imagePaths[result[0]]))
            else:
                cv2.putText(image_np,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(image_np, (x,y),(x+w,y+h),(0,0,255),2)



    return image_np

st.set_page_config(page_title="Reconocimiento Facial en Vivo", page_icon=":camera:", layout="wide")
st.title("📸 Vamos a tomar asistencia")

st.sidebar.title("Opciones")
image_source = st.sidebar.selectbox("Seleccione la fuente de la imagen", ["Cámara en Vivo",'Nuevo Integrante',"Subir Imagen", ])

image = None
stop_camera = False

if "stop_camera" not in st.session_state:
    st.session_state["stop_camera"] = False

if "reset" not in st.session_state:
    st.session_state["reset"] = False

if image_source == "Subir Imagen":
    face_recognizer = load_model()
    st.write("Por favor, cargue una imagen para detectar rostros.")
    image_file = st.file_uploader("Cargue una imagen", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        image = load_image(image_file)
        ret, image = capture_from_file(image_file)
        
    else:
        st.write('O capture una imagen')
        take_photo=st.button('Tomar Foto 📷')
        if take_photo:
            image_cap=capture_photo()
            img_rgb = cv2.cvtColor(image_cap, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb)
    
    if image is not None:
           
            # Detect faces
            faces = detect_faces(image)
            st.write(f"### Se detectaron {len(faces)} rostro(s) en la imagen.")

            # Draw faces
            if len(faces) > 0:
                result_image = draw_faces(image, faces,image_cap)
                st.image(result_image, caption='Imagen con rostros detectados', use_column_width=True)
    else:
        st.error('Error al capturar la imagen')


elif image_source== "Nuevo Integrante":
    personName=st.text_input('Nombre del integrante','',20)
    start_capture=st.button('Iniciar Captura')
    train_model=st.button('Entrenar modelo')
    if train_model:
        trainModel()
    if start_capture:
        if personName=='':
            st.error('Debe ingresar un nombre')
        else:
            print('Lanzando metodo')                       
            
            dataPath = './data' #Cambia a la ruta donde hayas almacenado Data
            personPath = dataPath + '/' + personName

            if not os.path.exists(personPath):
                print('Carpeta creada: ',personPath)
                os.makedirs(personPath)

            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            frame_window = st.image([])
            #cap = cv2.VideoCapture('Video.mp4')

            faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
            count = 0

            while True:

                ret, frame = cap.read()
                if ret == False: break
                frame =  imutils.resize(frame, width=640)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                auxFrame = frame.copy()
                
                faces = faceClassif.detectMultiScale(gray,1.3,5)

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                    rostro = auxFrame[y:y+h,x:x+w]
                    rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro)
                    count = count + 1
                
                # Draw faces
                frame_with_faces = draw_faces(Image.fromarray(frame), faces,frame,True)
                
                # Update the Streamlit image element
                frame_window.image(frame_with_faces, channels="RGB")
                k =  cv2.waitKey(1)
                if k == 27 or count >= 300:
                    break

            cap.release()
            cv2.destroyAllWindows()
            st.success('Captura completada')

else:
    
    st.write(" La cámara en vivo está activada. Por favor, espere un momento para que se muestre el video.")
    col1,col2 = st.columns(2)
    with col1:
        stop_button = st.button("Detener Cámara")
        face_recognizer = load_model()
        if stop_button:
            st.session_state["stop_camera"] = True
        
            st.write('Ingresaron las siguientes personas:')
            st.write(st.session_state.my_list)
    with col2:
        reset_button = st.button("Activar camara")
        if reset_button:
            st.session_state["reset"] = True
            st.session_state["stop_camera"] = False
            st.experimental_rerun()

    cap = cv2.VideoCapture(0)
    frame_window = st.image([])

    while not st.session_state["stop_camera"]:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo acceder a la cámara")
            break

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detect_faces(Image.fromarray(frame))
        
        # Draw faces
        frame_with_faces = draw_faces(Image.fromarray(frame), faces,frame)
        
        # Update the Streamlit image element
        frame_window.image(frame_with_faces, channels="RGB")

    cap.release()

    
    