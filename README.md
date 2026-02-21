# Detecção de Pneumonia (Classificação de Imagens Médicas).

Este projeto apresenta uma solução construída em **PyTorch** para a classificação de radiografias de tórax em duas categorias: **Normal** e **Pneumonia**. 

# Objetivo:
- Classificar o diagnóstico de imagens de raio-x em uma classificação binária(Normal ou Pneumonia) e transformar em uma distribuição probabilística (função SoftMax).

- Comparar o desempenho de diferentes arquiteturas avançadas de visão computacional (CNNs vs. Transformers).

  
- Garantir a interpretabilidade das decisões do modelo utilizando mapas de ativação de classe (Grad-CAM).

# Base de Dados
- Fonte: <a href="https://www.kaggle.com/competitions/ligia-compviz/overview">Kaggle – Lígia - CV
  
- Distribuição dos dados:
  - Conjunto de Treinamento e Validação: 5232 imagens (1349 Normal, 3883 Pneumonia).
  - Conjunto de Teste: 624 imagens.

- Pré-processamento e Augmentation:
  - Redimensionamento para 224x224 pixels.
  - Normalização utilizando médias e desvios-padrão do ImageNet.
  - Transformações sintéticas (RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter) para mitigar *overfitting*.

## Arquivos .pth e dataset:
 Uma vez que o github não aceita arquivos com mais de 30MB, eu separei o dataset e o modelo já treinado(vision transformers e resnet) e coloquei-os nesse <a href="https://drive.google.com/drive/folders/14kfJhBG6EoWypZf-7x8BWTTFI0fSapkM?usp=sharing">Drive 

## Estrutura do Projeto

```text

  ├── 📂 Código dos modelos/                # Pasta com todos os notebooks utilizados.
  │   ├── 📂 grad_cam/
  │   │   └── 📓 gradcam.ipynb              # Notebook para IA explicativa (Grad-CAM)
  │   │
  │   ├── 📂 resnet50/
  │   │   ├── 📓 resnet50.ipynb             # Treinamento e avaliação ResNet
  │   │   ├── 📓 gerar_csv_resnet.ipynb     # Script para gerar submissão Kaggle
  │   │   └── 📄 modelo_colab_resnet50.pth  # <- Este artefato está presente para baixar no drive
  │   │
  │   └── 📂 vision_transformers/
  │       ├── 📓 vision_transformers.ipynb # Treinamento do modelo ViT
  │       ├── 📓 gerar_csv_vision_transformers.ipynb
  │       └── 📄 modelo_colab_vision_transformers.pth # <- Este artefato está presente para baixar no drive
  │
  ├── 📂 dataset/                           # Base de dados do kaggle e drive(1.2gb)
  │   ├── 📂 train/                         # Imagens rotuladas (NORMAL/PNEUMONIA)
  │   │   ├── 📂 NORMAL                 
  │   │   ├── 📂 PNEUMONIA              
  │   │
  │   ├── 📂 test_images/                   # Imagens de teste sem rótulo(conjunto de teste)
  │   ├── 📄 train.csv
  │   └── 📄 test.csv
  │
  ├── 📂 gradcam_results/                   # Resultados dos mapas de calor
  ├── 📂 graficos resnet50/                 # Métricas visuais do modelo ResNet
  ├── 📂 Gráficos vision transformers/      # Métricas visuais do modelo ViT
  │
  ├── 📄 .gitignore                         # Configurado para ignorar venv e dataset
  ├── 📄 README.md                          # Documentação do projeto
  └── 📄 requirements.txt                   # Dependências do ambiente
```
# Metodologia
A análise inicial do problema indicou a necessidade de modelos robustos capazes de extrair características complexas de imagens médicas. O problema foi formulado como uma tarefa de classificação de imagens utilizando Transfer Learning.

Dada a especificidade das radiografias, adotou-se a estratégia de fine-tuning parcial: as camadas iniciais e intermediárias foram congeladas para preservar as features de baixo nível, enquanto os blocos de alto nível e o classificador final foram treinados. O classificador original foi substituído por uma rede sequencial contendo redução de dimensionalidade, Batch Normalization e Dropout para estabilização e regularização.

# Modelos Avaliados
- Redes Neurais Convolucionais: ResNet-50 (com aprendizado residual)

- Vision Transformers: ViT-16 (com mecanismo de self-attention)
  
# Resultados
Os resultados evidenciaram que a arquitetura ResNet-50 apresentou um desempenho levemente superior e mais estável. O modelo alcançou uma ROC-AUC de 0.9980 e um Recall de 0.98 para a classe de Pneumonia, indicando excelente capacidade de minimizar falsos negativos (cenário crítico onde um paciente doente seria liberado sem tratamento).

# Reprodutibilidade(Recomendo o uso do vscode como IDE):
   ```
   Todo o código foi produzido e rodou localmente no python 3.10.7
   ```

1. Clone o repositório
   ```bash
   git clone https://github.com/arthlz/Pneumonia-detection-CV.git
   ```
   
2. Acesse a pasta do projeto
   ```pasta
   cd Pneumonia-detection-CV
   ```

3. Baixe a pasta "dataset" presente no drive ou baixe o conjunto de dados presente no kaggle, mude o nome da pasta principal para dataset e organize as pastas(Use como apoio a *estrutura do projeto*)
   ```dataset
   Extraia os arquivos baixados e coloque-os na raiz do projeto dentro de uma pasta chamada dataset
   Certifique-se de que a estrutura esteja como: dataset/train/... e dataset/test_images/....
   ```

4. No terminal, crie um ambiente virtual e ative-o
   ```Venv
    python -m venv venv <- Cria o ambiente virtual
   
    - Dispositivos Windows:
    .\venv\Scripts\activate -< Ativa o ambiente virtual
   
    - Dispositivos Linux/Mac:
    source venv/bin/activate <- Ativa o ambiente virtual
   ```

5. Instale os requerimentos necessários:
   ```bash
   pip install -r requirements.txt
   ```

6. Abra e execute qualquer arquivo desejado.


## 💻Programador:

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/arthlz">
        <img src="https://avatars.githubusercontent.com/u/173482833?v=4" width="120px;" alt="Arthur Luz"/><br>
        <sub><b>Arthur Luz</b></sub>
      </a>
    </td>
  </tr>
</table>

## Tecnologias Utilizadas:
<div align="left">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
<img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" />
</div>
