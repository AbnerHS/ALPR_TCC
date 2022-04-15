import cv2
import matplotlib.pyplot as plt
import glob
import encontraPlaca as encontra
import segmentarCaracteres as char

def main():
  images_path = glob.glob("imagens/*.jpg")
  acertos = 0
  for image in images_path:
    img = cv2.imread(image)
    img_carro, img_placa = encontra.encontraPlaca(img)
    acertos += len(img_placa)
    for placa in img_placa:
      img_binaria = char.segmenta(placa)
      plt.imshow(cv2.cvtColor(img_binaria, cv2.COLOR_BGR2RGB))
      plt.show()
  print(acertos)
  
if __name__ == '__main__':
  main()