import cv2
import numpy as np

def ocr_in_image(img):
  swapL = {'5': 'S', '7': 'Z', '1': 'I', '8': 'B', '2': 'Z', '4': 'A', '6': 'G', '0': 'O'}
  swapN = {'Q': '0', 'D': '0', 'Z': '7', 'S': '5', 'J': '1', 'I': '1', 'A': '4', 'B': '8', 'O': '0'}
  weights = "yolo/yolov4_tiny_ocr_last.weights"
  cfg = "yolo/yolov4_tiny_ocr.cfg"
  names = open('yolo/caracteres.names', 'r')

  classes = names.read().splitlines()
  net = cv2.dnn.readNet(weights, cfg)
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
  
  height, width, _ = img.shape

  blob = cv2.dnn.blobFromImage(img, 1/255, (352, 128), (0, 0, 0), True, crop=False)
  # blob = cv2.dnn.blobFromImage(img, 1/255, (576, 192), (0, 0, 0), True, crop=False)
  
  net.setInput(blob)
  outs = net.forward(output_layers)

  confidences = []
  boxes = []
  class_ids = []
  
  for out in outs:
      for detection in out:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          if confidence > 0.1:
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

  posX = []
  labels = []
  placa = ""

  img_blank = np.zeros((int(height), width, 3), np.uint8)
  img = cv2.vconcat([img, img_blank])
  font = cv2.FONT_HERSHEY_PLAIN
  for i in range(len(boxes)):
    if i in indexes:
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      posX.append(x)
      labels.append(label)
      color = (0, 255, 0)
      cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
      cv2.putText(img, label, (x, y + 50), font, 2, color, 2)
     
  labels = [x for _,x in sorted(zip(posX, labels))]
  # labels = ''.join(labels)
  index = 0
  for char in labels:
    if index < 3:
      if char.isdigit():
        char = swapL.get(char, char)
        # print("swap Numero para Letra")
    elif index == 4:
      pass
    else:
      if not char.isdigit():
        char = swapN.get(char, char)
        # print("swap Letra para Numbero")
    placa += str(char)
    index += 1
  # img = cv2.resize(img, None, fx=2, fy=2)
  cv2.imshow("Image", img)
  # cv2.waitKey(0)
  return placa