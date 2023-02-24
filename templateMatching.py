import cv2 as cv
import numpy as np
import glob
import math
import time
import utils
import matplotlib.pyplot as plt

def converter(cantoSuperior, cantoInferior, classe):
    posX = math.ceil((cantoInferior[0] + cantoSuperior[0])/2)
    posY = math.ceil((cantoInferior[1] + cantoSuperior[1])/2)
    for letra in eval('utils.'+classe):
        if posX >= eval('utils.'+classe)[letra]['x0'] and posX <= eval('utils.'+classe)[letra]['xf']:
            if posY >= eval('utils.'+classe)[letra]['y0'] and posY <= eval('utils.'+classe)[letra]['yf']:
                return letra
                break
    return ''


def maximos(vetor):
    lista = []
    size = len(vetor)
    test = True
    for i in range(0, int(size/3)):
        if vetor[i] > vetor[i+1]:
            lista.append(i)
            test = False
            break
    if(test):
        lista.append(0)
    test = True
    for i in range(size-1,int(size/3), -1):
        if vetor[i] > vetor[i - 1]:
            lista.append(i)
            test = False
            break
    if(test):
        lista.append(size-1)
    return lista


def preprocessamento(image):
    w, h = image.shape[::-1]
    listaH = []
    listaV = []
    # projeção horizontal
    for x in range(h):
        countH = 0
        for y in range(w):
            if image[x][y] == 255:
                countH += 1
        listaH.append(countH)
    # projeção vertical
    for y in range(w):
        countV = 0
        for x in range(h):
            if image[x][y] == 255:
                countV += 1
        listaV.append(countV)
    listaH = maximos(listaH)
    listaV = maximos(listaV)
    image = image[listaH[0]:listaH[1]+1,listaV[0]:listaV[1]+1]
    return image

def reconhecer(images, classe, show=False):
    i = 0
    placa = ''
    for template in images:
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY) #converter imagem para tons de cinza
        _, template = cv.threshold(template, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)   #threshold de otsu
        template = preprocessamento(template)
        template = cv.resize(template, (25,40))
        kernel = np.ones((3, 3), np.uint8)  #matriz 3x3 de 1
        template = cv.erode(template, kernel, iterations=1) #erosão           
        w, h = template.shape[::-1]
        if i < 3:
            mapa = cv.imread('templates/template_letras_'+classe+'.jpg', 0)
        elif i == 4 and classe == 'mercosul':   #se for o quinto caractere e mercosul -> comparar apenas com letras
            mapa = cv.imread('templates/template_letras_mercosul.jpg', 0)
        else:
            mapa = cv.imread('templates/template_numeros_'+classe+'.jpg', 0)
        
        res = cv.matchTemplate(mapa, template, cv.TM_CCOEFF)
        _, _, _, max_loc = cv.minMaxLoc(res)    #localização do caractere com maior proximidade
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        placa += converter(max_loc, bottom_right, classe)   #obter caractere ASCII da localização encontrada
        i += 1

        if show:
            cv.rectangle(mapa, max_loc, bottom_right, 0, 2) #colocar bounding box no caractere
            cv.imshow("Template", template)
            cv.imshow("Image", mapa)
            cv.waitKey()

    return placa