import dlib
import numpy as np
import cv2

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords



class landmark(object):
    def __init__(self):
        self.p = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.p)
        print('iniciando')
    def play_video(self):
        cap = cv2.VideoCapture(0)
        print('capvideo')
        while True:
            _, image = cap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)

            for (i, rect) in enumerate(rects):
                shape = self.predictor(gray, rect)
                shape = shape_to_np(shape)

                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            cv2.imshow("Output", image)


            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        cap.release()

    def encode_frame(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, image = cap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)

            for (i, rect) in enumerate(rects):
                shape = self.predictor(gray, rect)
                shape = shape_to_np(shape)

                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()

    def boca(self):

        cap = cv2.VideoCapture(0)
        #print('capvideo')
        while True:
            _, image = cap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            if rects:
                for (i, rect) in enumerate(rects):
                    shape = self.predictor(gray, rect)
                    shape = shape_to_np(shape)

                    shape_boca = shape[48:68]
                    shape_mand = shape[0:17]
                    shape_nariz = shape[27:35]

                    for (x, y) in shape_boca:
                        #cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                        cv2.putText(image, 'SEM MASCARA', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)

                        (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #roi = image[y:y + h, x:x + w]

                    #for (x, y) in shape:
                        #cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            else:
                cv2.putText(image, 'COM MASCARA', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            cv2.imshow("Output", image)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break



a = landmark()
#a.play_video() #Faz a leitura do video e o display no PC
a.boca()

#a.encode_frame() Retorna o Frame encodado

