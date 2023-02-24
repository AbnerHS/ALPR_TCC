import cv2
import numpy as np

def detect_object_in_image(net, img, classes, size=(416, 416), show = False, ocr = False):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    height, width, _ = img.shape

    #redimensiona imagem
    blob = cv2.dnn.blobFromImage(img, 1/255, size, (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    classIds = []
    label = ""
    saidas = []

    #iterar saídas
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.1:
                #obter posições da detecção do YOLO
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(classId)

    #supressão não máxima para eliminar bounding boxes sobrepostas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    posX = []
    posY = []
    labels = []
    if ocr:
        #criar espaço vazio para mostrar caracteres
        img_blank = np.zeros((int(height), width, 3), np.uint8)
        img = cv2.vconcat([img, img_blank])
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = (0, 255, 0)

            #limitar posicão em mínimo de 0
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            
            #centralizar posição da label
            posX.append(x)
            posY.append(y+h)
            segmento = img.copy()[y:y+h,x:x+w]
            label = str(classes[classIds[i]])
            saidas.append({
                "img" : segmento,
                "label" : label
            })
            if ocr:
                labels.append(label)
            #colocar bouding box e label na imagem
            if show:
                #mostrar caracteres
                if ocr:
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
                else:
                    text = "{}: {:.3f}".format(label, confidences[i])
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 8)
                    cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    #mostrar imagem
    if show:
        cv2.imshow("Image", img)
        cv2.waitKey()

    if ocr:
        placa = ""
        swapLetterToNumber = {'5': 'S', '7': 'Z', '1': 'I', '8': 'B', '2': 'Z', '4': 'A', '6': 'G', '0': 'O'}
        swapNumberToLetter = {'Q': '0', 'D': '0', 'Z': '7', 'S': '5', 'J': '1', 'I': '1', 'A': '4', 'B': '8', 'O': '0'}
        labels = [x for _,x in sorted(zip(posX, labels))]
        labels = ''.join(labels)
        index = 0
        for char in labels:
            if index < 3:
                if char.isdigit():
                    char = swapLetterToNumber.get(char, char) #swap numero para letra
            elif index == 4:
                pass
            else:
                if not char.isdigit():
                    char = swapNumberToLetter.get(char, char) #swap letra para numero
            placa += str(char)
            index += 1    
        
        return placa

    return saidas, posX, posY