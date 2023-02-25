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
netCarro.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netCarro.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

netPlaca = cv2.dnn.readNet(weightsPlaca, cfgPlaca)
netPlaca.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netPlaca.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

netOCR = cv2.dnn.readNet(weightsOcr, cfgOcr)
netOCR.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netOCR.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def ler_placa(carro, img, x = [], y = []):
  if FIND_CAR:
    placaDetected, _, _ = detect_object_in_image(netPlaca, carro, classesPlaca, size=(416, 416))
  else:
    placaDetected, x, y = detect_object_in_image(netPlaca, carro, classesPlaca, size=(416, 416))
  i = 0
  for placa in placaDetected:
    caracteresOCR = detect_object_in_image(netOCR, placa["img"], classesOcr, size=(352, 128), ocr=True)
    cv2.putText(img, caracteresOCR, (x[i] + 70, y[i] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    i+=1
  
    
def calculate_fps(prev_frame_time):
  new_frame_time = time.time() 
  sec = (new_frame_time-prev_frame_time)
  fps = 1/sec
  return new_frame_time, fps, sec
  
def main():
  prev_frame_time = 0
  cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
  totalFps = 0
  totalFpsMovel = 0
  mediaFpsMovel = 0
  mediaTimeMovel = 0
  frameCountTotal = 0
  frameCountMovel = 0
  while cam.isOpened():
    prev_frame_time, fps, sec = calculate_fps(prev_frame_time)   
    _, img = cam.read()
    textConsole = ""
    textScreenLeft = ""
    textScreenRight = ""
    if frameCountTotal > 0:  #pular primeiro frame (ignorar delay de abrir a camera)
      if frameCountMovel == 30:
        mediaFpsMovel = totalFpsMovel / frameCountMovel
        mediaTimeMovel = 1/mediaFpsMovel
        frameCountMovel = 0
        totalFpsMovel = 0
      totalFps += fps
      totalFpsMovel += fps
      mediaFps = totalFps / frameCountTotal
      textConsole += "FPS: {:.2f} AVG: {:.2f} time: {:.2f} ms".format(fps, mediaFpsMovel, mediaTimeMovel * 1000)  
      textScreenLeft = "AVG: {:.2f}".format(mediaFps)
      textScreenRight = "FPS: {:.2f}".format(mediaFpsMovel)
      print(textConsole)

    frameCountTotal += 1
    frameCountMovel += 1
    if FIND_CAR:
      carroDetected, x, y = detect_object_in_image(netCarro, img, classesCarro, size=(480, 480))
      for carro in carroDetected:
        ler_placa(carro["img"], img, x, y)  #envia apenas imagem do carro
    else:
      ler_placa(img, img) #envia imagem inteira
    cv2.putText(img, textScreenLeft, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 5)  #borda preta
    cv2.putText(img, textScreenLeft, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)    
    cv2.putText(img, textScreenRight, (515, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 5) #borda preta
    cv2.putText(img, textScreenRight, (515, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) == ord('q'):
      break
  cam.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()