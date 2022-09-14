import cv2 as cv
import numpy as np
import glob
import math
import time
import utils

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

def reconhecer(images, classe, origem=0):
    i = 0
    placa = ''
    for template in images:
        if(origem == 1):
            template = cv.imread(template, 0)
        else:
          template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
        _, template = cv.threshold(template, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        # plt.subplot(1, 2, 2)
        # plt.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
        # plt.show()
        # template = cv.adaptiveThreshold(template, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 5)
        # template = cv.adaptiveThreshold(template, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 8)
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
        template = preprocessamento(template)
        template = cv.resize(template, (25,40))
        kernel = np.ones((3, 3), np.uint8)
        template = cv.erode(template, kernel, iterations=1)
        # plt.subplot(1, 2, 2)
        # plt.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
        # plt.show()
        # cv.imshow("normal", template)
        w, h = template.shape[::-1]
        if i < 3:
            mapa = cv.imread('templates/template_letras_'+classe+'.jpg', 0)
        elif i == 4 and classe == 'mercosul':
            mapa = cv.imread('templates/template_letras_mercosul.jpg', 0)
        else:
            mapa = cv.imread('templates/template_numeros_'+classe+'.jpg', 0)

        # if i < 3:
        #     mapa = cv.imread('template_letras_mercosul.jpg', 0)
        # elif i == 4:
        #     mapa = cv.imread('template.jpg', 0)
        # else:
        #     mapa = cv.imread('template_numeros_mercosul.jpg', 0)

        img2 = mapa.copy()
        res = cv.matchTemplate(img2, template, cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        cv.rectangle(img2, max_loc, bottom_right, 0, 2)
        placa += converter(max_loc, bottom_right, classe)
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
        # plt.subplot(1, 2, 2)
        # plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
        # plt.show()
        # cv.imshow("imagem", img2)
        # cv.waitKey(0)
        i += 1
    return placa


def main():
    tipos = ['mercosul','antiga']
    placaCorreta = 0
    placaTotal = 0
    letraCorreta = 0
    letraTotal = 0
    media = 0
    for tipo in tipos:
        pastas = glob.glob('caracteres/'+tipo+'/*')
        for pasta in pastas:
            label = pasta.split('\\')[1]
            images = glob.glob(pasta+"/*.jpg")
            i = 0
            start = time.time()
            placa = reconhecer(images,tipo,1)
            end = time.time()
            media += (end - start)
            placaTotal += 1
            if label == placa:
                placaCorreta += 1
                print('Label:', label, 'Placa:', placa, 'OK')
            else:
                print('Label:', label, 'Placa:', placa, 'ERRO')
            for i in range(len(placa)):
                letraTotal += 1
                if placa[i] == label[i]:
                    letraCorreta += 1
    print("Placas Corretas: ", placaCorreta, " Total: ", placaTotal)
    print("Letras Corretas: ", letraCorreta, " Total: ", letraTotal)
    print(media)
if __name__ == '__main__':
    main()