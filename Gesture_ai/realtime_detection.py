from gtts import gTTS
from tensorflow.keras.models import load_model

import cv2
loaded_model=load_model('D:/PycharmProjects/code/hand_gesture1.h5')
webcam = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)

# Category dictionary

categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}



while True:

    _, frame = cap.read()

    # Simulating mirror image

    frame = cv2.flip(frame, 1)
    print(frame)

    # Got this from collect-data.py

    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])

    y1 = 10

    x2 = frame.shape[1] - 10

    y2 = int(0.5 * frame.shape[1])

    # Drawing the ROI

    # The increment/decrement by 1 is to compensate for the bounding box

    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

    # Extracting the ROI

    roi = frame[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction

    roi = cv2.resize(roi, (64, 64))

    roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)

    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("test", test_image)

    # Batch of 1

    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    #print(result)

    prediction = {'Zero': result[0][0],
                  'One': result[0][1],
                  'Two': result[0][2],
                  'Three': result[0][3],
                  'Four': result[0][4],
                  'Five': result[0][5],
                  'Six': result[0][6],
                  'Seven': result[0][7],
                  'Eight': result[0][8],
                  'Nine': result[0][9],
                  'Space': result[0][10],
                  'A': result[0][11],
                  'B': result[0][12],
                  'C': result[0][13],
                  'D': result[0][14],
                  'E': result[0][15],
                  'F': result[0][16],
                  'G': result[0][17],
                  'H': result[0][18],
                  'I': result[0][19],
                  'J': result[0][20],
                  'K': result[0][21],
                  'L': result[0][22],
                  'M': result[0][23],
                  'N': result[0][24],
                  'O': result[0][25],
                  'P': result[0][26],
                  'Q': result[0][27],
                  'R': result[0][28],
                  'S': result[0][29],
                  'T': result[0][30],
                  'U': result[0][31],
                  'V': result[0][32],
                  'W': result[0][33],
                  'X': result[0][34],
                  'Y': result[0][35],
                  'Z': result[0][36],


                  }

    max_key = max(prediction, key=prediction.get)

    #cv2.putText(test_image, max_key, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    print(max_key)
    mytext = max_key
    cv2.putText(frame, mytext,(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)






    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(2)

    if interrupt & 0xFF == 27:  # esc key

        break

cap.release()

cv2.destroyAllWindows()
import os

os.system('signtovoice.mp3')
language = 'en'

my = gTTS(text=mytext, lang=language, slow=False)

my.save('signtovoice.mp3')



