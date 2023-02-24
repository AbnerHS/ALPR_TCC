import sys
import cv2
import glob
from detect import detect_object_in_image
from ocr import ocr_in_image
from segmentarCaracteres import segmenta
from templateMatching import reconhecer
import time

def ler_txt(imageName):
  i = 0
  placa = ""
  with open(imageName + ".txt", "r") as arquivo:
    for linha in arquivo:
      if i == 1:
        placa = linha.strip().split(" ")[1]
      i+=1
  return placa

def teste_imagens(metodoOcr):
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

  imagesPath = sorted(glob.glob("RodoSol/cars-*/*.jpg"))
  imagesPathSize = len(imagesPath)

  placaTotal = 0
  letraCorreta = 0
  placaCorreta = 0

  tempoTotalCarro = 0
  tempoTotalPlaca = 0
  tempoTotalOCR = 0
  
  for image in imagesPath:
    startCarro = time.time()
    img = cv2.imread(image)    
    carroDetected, _, _ = detect_object_in_image(netCarro, img, classesCarro, size=(480, 480))
    endCarro = time.time() - startCarro
    tempoTotalCarro += endCarro
    placaTotal += 1
    for carro in carroDetected:
      startPlaca = time.time()
      placaDetected, _, _ = detect_object_in_image(netPlaca, carro["img"], classesPlaca, size=(416, 416))
      endPlaca = time.time() - startPlaca
      tempoTotalPlaca += endPlaca
      for placa in placaDetected:
        startOCR = time.time()
        placaOcr = ""
        if metodoOcr == 1:
          imgCaracteres = segmenta(placa["img"])
          placaOcr = reconhecer(imgCaracteres, placa["label"])        
        elif metodoOcr == 0:
          placaOcr = ocr_in_image(netOCR, placa["img"], classesOcr)
        
        endOCR = time.time() - startOCR
        tempoTotalOCR += endOCR
        placaReal = ler_txt(image[:-4])
        for i in range(len(placaOcr)):
          if i == len(placaReal):
            break
          if placaOcr[i] == placaReal[i]:
            letraCorreta += 1
        print("{} - Real: {} OCR: {}".format(placaTotal, placaReal, placaOcr), end='')
        if placaReal == placaOcr:
          print(" OK")
          placaCorreta += 1
        else:
          print(" ERRO")
  
  tempoMedioCarro = tempoTotalCarro / imagesPathSize * 1000
  tempoMedioPlaca = tempoTotalPlaca / imagesPathSize * 1000
  tempoMedioOCR = tempoTotalOCR / imagesPathSize * 1000
  tempoMedioTotal = tempoMedioCarro + tempoMedioPlaca + tempoMedioOCR
  print("{} placas corretas - {} letras corretas - tempo médio: {:.2f} ms".
    format(
      placaCorreta,  
      letraCorreta, 
      tempoMedioTotal
    )
  )
    
  print("veículo: {:.2f} ms \nplaca: {:.2f} ms \nocr: {:.2f} ms".format(tempoMedioCarro, tempoMedioPlaca, tempoMedioOCR))
  
def main():
  args = sys.argv[1:] #0 OCR com YOLO - 1 OCR com template matching
  if not len(args):
    teste_imagens(0)
  else:
    teste_imagens(int(args[0]))

  
if __name__ == '__main__':
  main()