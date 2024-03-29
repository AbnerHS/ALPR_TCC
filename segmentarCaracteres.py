import cv2

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
    countH = 0 if countH < 3 else countH
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

def eliminarFalsoCaracteres(img, pontos_x):
  for i in range(0, len(pontos_x)-1, 2):
    imgCopy = img.copy()
    imgCopy = imgCopy[:,pontos_x[i]:pontos_x[i+1]]
    height, width = imgCopy.shape
    lista_projecao_h = projecaoHorizontal(imgCopy, width, height)
    if sum(lista_projecao_h) < 40:
      pontos_x[i] = -1
      pontos_x[i+1] = -1
  pontos_x = [x for x in pontos_x if x != -1] 
  return pontos_x

def separarCaracteres(pontos_x):
  for i in range(0, len(pontos_x)-1, 2):
    if (pontos_x[i+1]-pontos_x[i]) > 30: #corta no meio o que é muito grande
      e = int((pontos_x[i] + pontos_x[i + 1]) / 2)
      pontos_x.append(e)
      pontos_x.append(e+1)
  pontos_x.sort()
  return pontos_x

def eliminarSegmentosExcedentes(pontos_x, width):
  if len(pontos_x) > 14:
    inicio = [x for x in pontos_x if x <= int(width / 2)]   #segmentos da primeira metade da placa
    fim = [x for x in pontos_x if x > int(width / 2)]       #segmentos da segunda metade da placa
    #se ambas as listas de segmentos estiverem com numero par de pontos, tira um da primeira metade e coloca na segunda metade
    if len(inicio) % 2 and len(fim) % 2:                    
      fim.insert(0, inicio.pop())

    #se a primeira metade tiver mais que 6 pontos, tirar os excedentes no inicio
    if len(inicio) > 6:
      dif = len(inicio) - 6
      del pontos_x[:dif]

    #se a segunda metade tiver mais que 8 pontos, tirar os excedentes no fim
    if len(fim) > 8:
      dif = len(fim) - 8
      del pontos_x[-dif:]
  return pontos_x


def segmenta(img, show = False):
  img_original = cv2.resize(img.copy(), (180, 60))
  img = rgbToGray(img)
  img = cv2.resize(img, (180, 60))
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
  mins_v = minimosLocais(lista_projecao_v)  #pontos em x de cada segmento
  
  #Tratamento da segmentação
  mins_v = eliminarFalsoCaracteres(img_binaria, mins_v)
  mins_v = separarCaracteres(mins_v)
  mins_v = eliminarSegmentosExcedentes(mins_v, width)

  mins_h = list(map(lambda x: x + topo, mins_h))  #somar valor do topo que foi retirado

  if show:
    img_boxes = bounding_box(img_original.copy(), mins_v, mins_h)   #adicionar bounding boxes na imagem
    # Mostrar imagem original com bounding boxes
    cv2.imshow("Image", img_boxes)
    cv2.waitKey()
    
  segmentos = []

  for i in range(0, len(mins_v)-1, 2):
    segmento = img_original[mins_h[0]:mins_h[1], mins_v[i]:mins_v[i+1]]
    segmentos.append(segmento)

  return segmentos
