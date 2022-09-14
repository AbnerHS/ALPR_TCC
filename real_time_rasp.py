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
  print(text)
  return new_frame_time

def ler_placa(carro, img, x = 0, y = 0):
  if FIND_CAR:
    placas, _, _, _ = detect_object_in_image(carro, weights_placa, cfg_placa, labels_placa, (256, 256))
  else:
    placas, _, x, y = detect_object_in_image(carro, weights_placa, cfg_placa, labels_placa, (256, 256))
  for placa in placas:
    caracteresOCR = ocr_in_image(placa)
    cv2.putText(img, caracteresOCR, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    

def main():
  prev_frame_time = 0
  counter = 5
  cam = cv2.VideoCapture(0)
  cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
  cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  while cam.isOpened():
    _, img = cam.read()
    if counter % 5 == 0:
      prev_frame_time = show_fps(prev_frame_time, img)    
      if FIND_CAR:
        carros, _, x, y = detect_object_in_image(img, weights_carro, cfg_carro, labels_carro, (256, 256))
        for carro in carros:
          ler_placa(carro, img, x, y)
      else:
        ler_placa(img, img)
      cv2.imshow("Video", img)
      if cv2.waitKey(10) == ord('q'):
        break
    counter += 1
  cam.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()