import cv2
import matplotlib.pyplot as plt
import glob
import encontrarPlaca as placa

def main():
  images_path = glob.glob("imagens/*.jpg")
  acertos = 0
  for image in images_path:
    img = cv2.imread(image)
    imgPlaca, acerto = placa.encontrarPlaca(img)
    acertos += acerto
    plt.imshow(cv2.cvtColor(imgPlaca, cv2.COLOR_BGR2RGB))
    plt.show()
  print(acertos)
if __name__ == '__main__':
  main()