# Projeto de Reconhecimento Automático de Placas Veiculares (ALPR)
O algoritmo foi desenvolvido utilizando 3 modelos de rede neural convolucional (CNN), para detecção do veículo, da placa e dos caracteres, treinados com uma base de dados pública RodoSol-ALPR.

## Tecnologias
- ``Python 3.10``
- ``OpenCV``
- ``YOLO``


## Execução
Este projeto possui dois arquivos principais de execução:
- `main.py`: executa o algoritmo para todas as imagens da base de dados, armazenadas localmente;
- `real_time.py`: executa o algoritmo em tempo real com uma Webcam;

O arquivo `real_time_rasp.py` é utilizado para execução em tempo real no Raspberry.

### Observação
As linhas 

```python
netCarro.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netCarro.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

netPlaca.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netPlaca.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

netOCR.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netOCR.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

em ambos os arquivos de execução, são utilizadas para utilizar o processamento da GPU, e só funcionará em ambiente CUDA com OpenCV configurado. Caso contrário, deverão ser comentadas ou apagadas.
