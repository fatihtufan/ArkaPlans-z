from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras import layers, models
from keras.constraints import maxnorm
from keras.utils import to_categorical
from keras.optimizers import SGD

from skimage import io
import numpy as np
import cv2
import os


width_height = 300
dirs = ["bir","iki","uc","dort","bes"]
encoder = LabelEncoder()
y = encoder.fit_transform(dirs)
global start

def train():
    global y
    x = None

    if not os.path.exists('dataset_siyahBeyaz.npy'):
        
        index = 1
        imgs = []
        for dir in dirs:
            images = os.listdir(os.path.join("veriseti", dir))
            for image in images:
                if image.endswith(".png"):
                    
                    tmp_img = prepare_image(os.path.join("veriseti", dir, image))
                    tmp_img = resize(width_height, width_height, tmp_img)
                    
                    tmp_img = tmp_img.reshape((tmp_img.shape[0] * tmp_img.shape[1],))
                    
                    b = np.zeros((tmp_img.shape[0] + 1,))
                    b[0] = y[index - 1]
                    b[1:] = tmp_img
                    
                    imgs.append(b)
            index += 1
        
        x = np.array(imgs)
        x = shuffle(x)
        
        np.save("dataset_siyahBeyaz", x)
    else:
        x = np.load("dataset_siyahBeyaz.npy")
    
    y = x[:, [0]]
    y = y.reshape(y.shape[0], )
    x = x[:, 1:]

    total = len(dirs)
    x = x.astype('float32')
    x = x.reshape(x.shape[0], width_height, width_height, 1)
    y = to_categorical(y, total)
    model = models.Sequential()

    model.add(layers.Conv2D(32, (5, 5), padding='same', kernel_constraint=maxnorm(3), input_shape=(width_height, width_height, 1)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (5, 5), padding='same', kernel_constraint=maxnorm(3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(total, activation='softmax'))
    epochs = 50
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    #adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.3)
    model.fit(x_train, y_train, validation_split=0.3, epochs=epochs, batch_size=50)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Acc : " + str(accuracy))
    print("Test Loss : " + str(loss))

    model.save_weights('face_siyahBeyaz.h5')
    with open('face_siyah_beyaz.json', 'w') as f:
        f.write(model.to_json())


def resize(max_height: int, max_width: int, frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]

    if max_height < height or max_width < width:
        if width < height:
            scaling_factor = max_width / float(width)
        else:
            scaling_factor = max_height / float(height)

        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame
    

def crop_center(frame: np.ndarray) -> np.ndarray:
    short_edge = min(frame.shape[:2])
    yy = int((frame.shape[0] - short_edge) / 2)
    xx = int((frame.shape[1] - short_edge) / 2)
    crop_img = frame[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

    start = int((frame.shape[1] - frame.shape[0]) / 2)
    end = int(frame.shape[1] - (frame.shape[1] - frame.shape[0]) / 2)
    return frame[:, start:end]


def detect_my_face():
    # capture frames from a camera
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    model = models.Sequential()

    with open('face_siyah_beyaz.json', 'r') as f:
        model = models.model_from_json(f.read())

    model.load_weights('face_siyahBeyaz.h5')
    
    while (1):
        ret, real_frame = cap.read()
        real_frame=cv2.flip(real_frame,1)
        frame = cv2.cvtColor(real_frame, cv2.COLOR_BGR2GRAY)
        frame = fgbg.apply(frame)
        frame = resize(width_height, width_height, frame)
        frame = crop_center(frame)
        cv2.imshow("farame", frame)

        predict = frame/ 300
        predict = predict.astype('float32')
        predict = predict.reshape(width_height, width_height, 1)
       
        
       
        prediction = model.predict(np.array([predict])).tolist()
        
        print("prediction: ",prediction)
        prediction_result = np.max(prediction)
        print("prediction_result: ",prediction_result)
        start=0
        end=0
        if prediction_result > 0.5:
            text = dirs[np.argmax(prediction)]
            print("text: ",np.argmax(prediction))
            start = int((real_frame.shape[1] - real_frame.shape[0]) / 2)
            end = int(real_frame.shape[1] - (real_frame.shape[1] - real_frame.shape[0]) / 2)

            cv2.putText(real_frame, text, (0, 125), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,0,255))
            cv2.putText(real_frame, "%.2f" % prediction_result, (0, 55), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,0,255))
        else:
            cv2.putText(real_frame, "TANIMSIZ", (0, 125), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,0,255))
                
        lineThickness = 2
        cv2.line(real_frame, (start, 0), (start, real_frame.shape[0]), (0, 0, 255), lineThickness)
        cv2.line(real_frame, (end, 0), (end, real_frame.shape[0]), (0, 0, 255), lineThickness)
        
        real_frame = resize(300, 300, real_frame)
        cv2.imshow("Camera Screen", real_frame)

        k = cv2.waitKey(5) & 0xFF 
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()





def prepare_image(image_path: str) -> np.ndarray:
    tmp_img = io.imread(image_path)
    #tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    tmp_img = resize(width_height, width_height, tmp_img)
    tmp_img = crop_center(tmp_img)
    tmp_img = tmp_img / 300
    tmp_img = tmp_img.astype('float32')
    tmp_img = tmp_img.reshape(width_height, width_height, 1)
    return tmp_img
    





#train()
detect_my_face()