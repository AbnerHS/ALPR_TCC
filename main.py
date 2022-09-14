import cv2
import pandas as pd
import glob
from detect import detect_object_in_image
from ocr import ocr_in_image

def ler_csv(img_id):
  test_file = "SSIG-SegPlate/test.csv"
  df = pd.read_csv(test_file, usecols=['img_id','text'], dtype=str)
  label = df.loc[df['img_id'] == img_id]['text'].iloc[0]
  return label

def ler_txt(img_id):
  i = 0
  placa = ""
  with open("RodoSol/cars-br/" + img_id + ".txt", "r") as arquivo:
    for linha in arquivo:
      if i == 1:
        placa = linha.strip().split(" ")[1]
      i+=1
  return placa

def teste_imagens():
  weights_carro = "yolo/yolov4_tiny_carro.weights"
  cfg_carro = "yolo/yolov4_tiny_carro.cfg"
  labels_carro = ['veiculo']
  weights_placa = "yolo/yolov4_tiny_placa.weights"
  cfg_placa = "yolo/yolov4_tiny_placa.cfg"
  labels_placa = ['brasileira','mercosul']
  images_path = sorted(glob.glob("RodoSol/cars-br/*.jpg"))
  # images_path = glob.glob("SSIG-SegPlate/test/*.png")
  # images_path = glob.glob("ocr/*.jpg")
  # images_path = glob.glob("imagens/*.jpg")
  placaTotal = 0
  acertosLocalizarPlaca = 0 
  letraTotal = 0
  letraCorreta = 0
  letraPlacaCorreta = 0
  acertosOCR = 0
  for image in images_path[:]:
    img = cv2.imread(image)
    img_carro, _, _, _ = detect_object_in_image(img, weights_carro, cfg_carro, labels_carro)
    placaTotal += 1
    for carro in img_carro:
      # cv2.imwrite("RodoSol/treino-me/"+str(i)+ "_" +image.split("\\")[1], carro)
      # i = 0
      img_placa, _, _, _ = detect_object_in_image(carro, weights_placa, cfg_placa, labels_placa, size=(480, 480))
      acertosLocalizarPlaca += 1
      for placa in img_placa:
        # image_name = "placas/toOcr/" + image.split("\\")[1].split(".")[0] + "_" + str(i) + ".jpg"
        # cv2.imwrite(image_name, placa)
        # i+=1
        # img_caracteres = segmenta.segmenta(placa)
        # caracteres = templateMatching.reconhecer(img_caracteres, tipo_placa)
        # label = ler_csv(image.split("\\")[1][:-4])
        caracteres = ocr_in_image(placa)
        label = ler_txt(image.split("\\")[1][:-4])
        if len(caracteres) <= 7:
          letraPlacaCorreta = 0
          for i in range(len(caracteres)):
            letraTotal += 1
            if caracteres[i] == label[i]:
              letraPlacaCorreta += 1
              letraCorreta += 1
          print(str(placaTotal) + ' - Label:', label, 'Placa:', caracteres, letraPlacaCorreta, end='')
          if(label == caracteres):
            print(" OK")
            acertosOCR += 1
          else:
            print(" ERRO")
  print("{} placas corretas de {} - {} letras corretas de {}".format(acertosOCR, acertosLocalizarPlaca, letraCorreta, letraTotal))
  
def main():
  teste_imagens()
  
if __name__ == '__main__':
  main()