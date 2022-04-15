import cv2
import numpy as np

def encontraPlaca(img):
    weights = "yolo/yolov3_training_6000.weights"
    cfg = "yolo/yolov3_training.cfg"
    net = cv2.dnn.readNet(weights, cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    classes = ['mercosul','antiga']
    
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    class_ids = []
    img_cortada = []
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
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = (0,255,0)
            img_cortada.append(img.copy()[y:y+h,x:x+w])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 8)
            label = str(classes[class_ids[i]])
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    img = cv2.resize(img, None, fx=0.6, fy=0.6)
    return [img, img_cortada]