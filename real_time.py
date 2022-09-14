import cv2
import time
from ocr import ocr_in_image
from detect import detect_object_in_image

FIND_CAR = False

weights_carro = 'yolo/yolov4_tiny_carro.weights'
cfg_carro = 'yolo/yolov4_tiny_carro.cfg'
labels_carro = ['veiculo']
weights_placa = 'yolo/yolov4_tiny_placa.weights'
cfg_placa = 'yolo/yolov4_tiny_placa.cfg'
labels_placa = ['brasileira','mercosul']


def show_fps(prev_frame_time, img):
  new_frame_time = time.time() 
  sec = (new_frame_time-prev_frame_time)
  fps = 1/sec
  text = "time: {:.2f}s fps: {:.2f}".format(sec, fps)
  cv2.putText(img, text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
  return new_frame_time

def ler_placa(carro, img, x = 0, y = 0):
  if FIND_CAR:
    placas, _, _, _ = detect_object_in_image(carro, weights_placa, cfg_placa, labels_placa, size=(320, 320))
  else:
    placas, _, x, y = detect_object_in_image(carro, weights_placa, cfg_placa, labels_placa)
  for placa in placas:
    caracteresOCR = ocr_in_image(placa)
    cv2.putText(img, caracteresOCR, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    

def main():
  prev_frame_time = 0
  cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
  cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  while cam.isOpened():
    _, img = cam.read()
    prev_frame_time = show_fps(prev_frame_time, img)    
    if FIND_CAR:
      carros, _, x, y = detect_object_in_image(img, weights_carro, cfg_carro, labels_carro)
      for carro in carros:
        ler_placa(carro, img, x, y)
    else:
      ler_placa(img, img)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) == ord('q'):
      break
  cam.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()