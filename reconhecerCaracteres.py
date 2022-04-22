import cv2
import numpy as np
import glob

def reconhecerCaracteres(img):
  swapL = {'5': 'S', '7': 'Z', '1': 'I', '8': 'B', '2': 'Z', '4': 'A', '6': 'G', '0': 'O'}
  swapN = {'Q': '0', 'D': '0', 'Z': '7', 'S': '5', 'J': '1', 'I': '1', 'A': '4', 'B': '8', 'O': '0'}
  weights = "ocr/lpscr-net.weights"
  cfg = "ocr/lpscr-net.cfg"
  names = open('ocr/lpscr-names.txt', 'r')
  classes = names.read().splitlines()
  net = cv2.dnn.readNet(weights, cfg)
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
  colors = np.random.uniform(0, 255, size=(len(classes), 3))
  
  height, width, _ = img.shape

  blob = cv2.dnn.blobFromImage(img, 1/255, (240, 80), (0, 0, 0), True, crop=False)
  
  net.setInput(blob)
  outs = net.forward(output_layers)

  confidences = []
  boxes = []
  class_ids = []
  placa = ""
  for out in outs:
      for detection in out:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          if confidence > 0.3:
              # Object detected
              center_x = int(detection[0] * width)
              center_y = int(detection[1] * height)
              w = int(detection[2] * width)
              h = int(detection[3] * height)

              # Rectangle coordinates
              x = int(center_x - w / 2)
              y = int(center_y - h / 2)

              boxes.append([x, y, w, h])
              confidences.append(float(confidence))
              class_ids.append(class_id)
  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  

  class_ids_novo = [x for _,x in sorted(zip(boxes, class_ids))]

  font = cv2.FONT_HERSHEY_PLAIN
  for i in range(len(boxes)):
    if i in indexes:
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      color = colors[class_ids[i]]
      cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
      cv2.putText(img, label, (x, y + 30), font, 2, color, 1)
      char = classes[class_ids_novo[i]]
      if i < 3:
        if char.isdigit():
          char = swapL.get(char, char)
      else:
        if not char.isdigit():
          char = swapN.get(char, char)
      placa += str(char)
  
  # img = cv2.resize(img, None, fx=2, fy=2)
  # cv2.imshow("Image", img)
  return [placa, img]

def main():
  images_path = glob.glob("placas/*.jpg")
  acertosOCR = 0
  for image in images_path:
    img = cv2.imread(image, 0)
    img = cv2.resize(img, (240, 80))
    img = np.stack((img,) * 3, axis=-1)
    caracteres = reconhecerCaracteres(img)
    label = image.split("\\")[1][:-4]
    print(label, caracteres, end='')
    if(label == caracteres):
      print(" OK")
      acertosOCR += 1
    else:
      print(" ERRO")
  print("Acertos OCR: " + str(acertosOCR))


if __name__ == '__main__':
  main()