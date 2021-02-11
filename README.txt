O programa main.py gera imagens comparando a imagem .png original e as respectivas soluções do exércicio 1 e 2. Ao realizar os cálculos o programa gera imagens em .png com o plot lado a lado de cada imagem para cada delta:
	- As imagens são gerados na pasta "figs";
	- Quando o determinante da matriz A é 0 (em alguns casos para delta=0), a imagem não é gerada e é explicitado relatando erro de overflow quando  a matriz é levada ao solver.
    - A imagem terá um nome padronizado: Tomografia (número da imagem) com delta= (delta utilizado).png";
    - Para comparação, imagens utilizando o solver do numpy também foram geradas e estão contidas na pasta "figsNumpySolver".
    - O programa "printa", os valores de determinante e erro quadrático


Erros quadráticos
Solver Gauss
Tomografia 1 com delta= 0.1 tem erro= 21.088218739015105
Tomografia 1 com delta= 0.01 tem erro= 21.270189339489356
Tomografia 1 com delta= 0.001 tem erro= 21.320071635561042
Tomografia 2 com delta= 0.1 tem erro= 23.420897460551654
Tomografia 2 com delta= 0.01 tem erro= 23.51088783126543
Tomografia 2 com delta= 0.001 tem erro= 23.635383640220386
Tomografia 3 com delta= 0.1 tem erro= 27.56899269849989
Tomografia 3 com delta= 0.01 tem erro= 27.645123773832616
Tomografia 3 com delta= 0.001 tem erro= 27.683642605447886

Solver Numpy
Tomografia 1 com delta= 0.1 tem erro= 21.072433045335504
Tomografia 1 com delta= 0.01 tem erro= 21.29007417174771
Tomografia 1 com delta= 0.001 tem erro= 21.317011948710523
Tomografia 2 com delta= 0.1 tem erro= 23.408016369869344
Tomografia 2 com delta= 0.01 tem erro= 23.544128543451183
Tomografia 2 com delta= 0.001 tem erro= 23.558247313085594
Tomografia 3 com delta= 0.1 tem erro= 27.57505874544766
Tomografia 3 com delta= 0.01 tem erro= 27.670813978092365
Tomografia 3 com delta= 0.001 tem erro= 27.680609038913648