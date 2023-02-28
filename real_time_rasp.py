import cv2
import time
from detect import detect_object_in_image

FIND_CAR = False

weightsCarro = 'yolo/yolov4_tiny_carro.weights'
cfgCarro = 'yolo/yolov4_tiny_carro.cfg'
classesCarro = open('yolo/yolov4_tiny_carro.names', 'r').read().splitlines()

weightsPlaca = 'yolo/yolov4_tiny_placa.weights'
cfgPlaca = 'yolo/yolov4_tiny_placa.cfg'
classesPlaca = open('yolo/yolov4_tiny_placa.names', 'r').read().splitlines()

weightsOcr = "yolo/yolov4_tiny_ocr.weights"
cfgOcr = "yolo/yolov4_tiny_ocr.cfg"
classesOcr = open('yolo/yolov4_tiny_ocr.names', 'r').read().splitlines()

netCarro = cv2.dnn.readNet(weightsCarro, cfgCarro)

netPlaca = cv2.dnn.readNet(weightsPlaca, cfgPlaca)

netOCR = cv2.dnn.readNet(weightsOcr, cfgOcr)


def ler_placa(carro, img, x = [], y = []):
  if FIND_CAR:
    placaDetected, _, _ = detect_object_in_image(netPlaca, carro, classesPlaca, size=(320, 320))
  else:
    placaDetected, x, y = detect_object_in_image(netPlaca, carro, classesPlaca, size=(320, 320))
  i = 0
  for placa in placaDetected:
    caracteresOCR = detect_object_in_image(netOCR, placa["img"], classesOcr, size=(352, 128), ocr=True)
    cv2.putText(img, caracteresOCR, (x[i], y[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    i+=1

def calculate_fps(prev_frame_time):
  new_frame_time = time.time() 
  sec = (new_frame_time-prev_frame_time)
  fps = 1/sec
  return new_frame_time, fps, sec


def main():
  prev_frame_time = 0
  counter = 5
  cam = cv2.VideoCapture(0)
  cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
  cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  totalFps = 0  
  frameCountTotal = 0
  
  while cam.isOpened():
    _, img = cam.read()
    if counter % 5 == 0:
      prev_frame_time, fps, sec = calculate_fps(prev_frame_time)   
      textConsole = ""
      textScreen = ""
      if frameCountTotal > 0:  #pular primeiro frame (ignorar delay de abrir a camera)
        totalFps += fps
        mediaFps = totalFps / frameCountTotal
        textConsole += "FPS: {:.2f} time: {:.2f} ms".format(fps, sec * 1000)  
        textScreen = "FPS: {:.2f}".format(mediaFps)      
        print(textConsole)

      frameCountTotal += 1
      if FIND_CAR:
        carroDetected, x, y = detect_object_in_image(netCarro, img, classesCarro, size=(416, 416))
        for carro in carroDetected:
          ler_placa(carro["img"], img, x, y)  #envia apenas imagem do carro
      else:
        ler_placa(img, img) #envia imagem inteira

      cv2.putText(img, textScreen, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 5)  #borda preta
      cv2.putText(img, textScreen, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)    
      cv2.imshow("Video", img)
      if cv2.waitKey(10) == ord('q'):
        break
    counter += 1
  cam.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()