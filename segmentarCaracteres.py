import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def binarizar(img):
  return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)

def rgbToGray(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def projecaoHorizontal(img, width, height):
  lista = [] 
  for y in range(height):
    countV = 0
    for x in range(width):
      if img[y][x] == 0:
        countV += 1
    lista.append(countV)
  return lista

def projecaoVertical(img, width, height):
  lista = []
  for x in range(width):
    countH = 0
    for y in range(height):
      if img[y][x] == 0:
        countH += 1
    countH = 0 if countH < 2 else countH
    lista.append(countH)
  return lista

def minimosLocais(vetor):
	lista = []
	status = False
	for i in range (len(vetor)-1):
		if vetor[i+1] > 1 and not status:
			lista.append(i)
			status = True

		if vetor[i+1] < 1 and status:
			lista.append(i+1)
			status = False
	return lista

def minimos(vetor):
  lista = []
  for i in range(1, len(vetor) - 1):  #loop do inicio pro fim
    if vetor[i] <= vetor[i+1]:
      lista.append(i)
      break
  for i in range(len(vetor)-1, 1, -1):  #loop do fim pro começo
    if vetor[i] <= vetor[i-1]:
      lista.append(i)
      break
  return lista

def bounding_box(img, pontos_x, pontos_y):
  for i in range(0, len(pontos_x) - 1, 2):
    x = pontos_x[i]
    w = pontos_x[i+1]
    y = pontos_y[0]
    h = pontos_y[1]
    cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 1)
  return img

def dilata(img):
	preDilatar = 255 - img;
	kernel = np.array([[0, 0, 0],
					   [0, 1, 0],
					   [0, 1, 0]], np.uint8)
	dilatar = cv2.dilate(preDilatar, kernel, iterations=1)
	dilatar = 255 - dilatar
	return dilatar

def view1(img1, img2, img3):
	plt.subplot(3, 1, 1)
	plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	plt.subplot(3, 1, 2)
	plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
	plt.subplot(3, 1, 3)
	plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
	plt.show()

def view2(img1, img2, y1, listaV, x3, listaMax):
	plt.subplot(4, 1, 1)
	plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	plt.subplot(4, 1, 2)
	plt.bar(y1, listaV, color='black')
	plt.subplot(4, 1, 3)
	plt.scatter(listaMax, x3, color = 'black')
	plt.subplot(4, 1, 4)
	plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
	plt.show()

def view3(img1, x1, listaH, x2, listaMin, img2):
	plt.subplot(4, 1, 1)
	plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	plt.subplot(4, 1, 2)
	plt.bar(x1, listaH, color='black')
	plt.subplot(4, 1, 3)
	plt.scatter(listaMin, x2, color='black')
	plt.subplot(4, 1, 4)
	plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
	plt.show()

def view4(img1, img2):
	plt.subplot(2, 1, 1)
	plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	plt.subplot(2, 1, 2)
	plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
	plt.show()

def segmenta(img):
  img_original = cv2.resize(img.copy(), (90, 30))
  img = rgbToGray(img)
  img = cv2.resize(img, (90, 30))
  height, width = img.shape
  topo = int(height * 0.3)                                    #tamanho da sarjeta da placa
  height -= topo
  img = img[topo:, :]                                         #cortando a sarjeta da imagem
  img_binaria = binarizar(img)
  img_binaria[:, 0] = 255
  img_binaria[:, -1] = 255

  lista_projecao_h = projecaoHorizontal(img_binaria, width, height)
  mins_h = minimos(lista_projecao_h)                            #pegar picos minimos
  img_binaria = img_binaria[mins_h[0] : mins_h[1], :]           #cortar partes brancas acima e abaixo
  height, width = img_binaria.shape       #pegar novos tamanhos da imagem

  lista_projecao_v = projecaoVertical(img_binaria, width, height)
  mins_v = minimosLocais(lista_projecao_v)
  
  mins_h = list(map(lambda x: x + topo, mins_h))          
  img_original = bounding_box(img_original, mins_v, mins_h)   #adicionar bounding boxes na imagem
  
  return img_original
