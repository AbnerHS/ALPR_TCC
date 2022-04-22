import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import encontraObjeto as encontra
import segmentarCaracteres as char
import reconhecerCaracteres as ocr

def lerCsv(img_id):
  test_file = "SSIG-SegPlate/test.csv"
  df = pd.read_csv(test_file, usecols=['img_id','text'], dtype=str)
  label = df.loc[df['img_id'] == img_id]['text'].iloc[0]
  return label

def main():
  weightsCarro = "yolo/yolov4_tiny_training_last.weights"
  cfgCarro = "yolo/yolov4_tiny_training.cfg"
  classesCarro = ['frente','traseira']
  weightsPlaca = "yolo/yolov3_training_6000.weights"
  cfgPlaca = "yolo/yolov3_training.cfg"
  classesPlaca = ['mercosul','antiga']
  images_path = glob.glob("SSIG-SegPlate/test/*.png")
  acertosYOLO = 0
  acertosOCR = 0
  for image in images_path:
    # fig = plt.figure(figsize=(19.2,10.8))
    # fig.canvas.manager.window.wm_geometry("+%d+%d" % (0,0))
    # plt.subplot(4, 1, 1)
    img = cv2.imread(image)
    img_origem, img_carro, _ = encontra.encontraObjeto(img, weightsCarro, cfgCarro, classesCarro)
    # plt.imshow(cv2.cvtColor(img_origem, cv2.COLOR_BGR2RGB))
    for carro in img_carro:
      # plt.subplot(4, 1, 2)
      # plt.imshow(cv2.cvtColor(carro, cv2.COLOR_BGR2RGB))
      _, img_placa, _ = encontra.encontraObjeto(carro, weightsPlaca, cfgPlaca, classesPlaca)
    for placa in img_placa:
      plt.subplot(4, 1, 3)
      plt.imshow(cv2.cvtColor(placa, cv2.COLOR_BGR2RGB))
      placa = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
      cv2.imwrite("placas/"+image.split("\\")[1], img=placa)
      placa = np.stack((placa,) * 3, axis=-1)
      caracteres, img_char = ocr.reconhecerCaracteres(placa)
      # plt.subplot(4, 1, 4)
      # plt.imshow(cv2.cvtColor(img_char, cv2.COLOR_BGR2RGB))
      # label = image.split("\\")[1][:-4]
      label = lerCsv(image.split("\\")[1][:-4])
      print(label, caracteres, end='')
      # plt.show()
      if(label == caracteres):
        print(" OK")
        acertosOCR += 1
      else:
        print(" ERRO")
      # segmentos = char.segmenta(placa)
      # for seg in segmentos:
      #   plt.imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
      #   plt.show()
  print("Acertos OCR: " + str(acertosOCR))
  # print("Acertos YOLO: " + str(acertosYOLO))
  
if __name__ == '__main__':
  main()