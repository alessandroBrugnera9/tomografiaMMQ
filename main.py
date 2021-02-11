import numpy as np
from numpy import load
import matplotlib.pyplot as plt
from PIL import Image
import os

if not os.path.exists('figs'):  # garantir que pasta existe para salvar imagens
    os.makedirs('figs')


def solver(A, B):  # metodo de elimincao de Gauss sem pivotamento
    elementos = (len(A))
    L = np.zeros((elementos, elementos))

    if A[0][0]==0:  # evitar erros
        return

    for k in range(0, elementos):
        # Criando L
        for i in range(k+1, elementos):
            L[i][k] = A[i][k]/A[k][k]
            A[i][k] = 0

        # Eliminações
        for j in range(k+1, elementos):
            for l in range(k+1, elementos):
                A[l][j] = A[l][j]-(L[l][k]*A[k][j])

            B[j] = B[j]-(L[j][k]*B[k])

    # Alterando diagonal principal de L para 1
    for i in range(0, elementos):
        L[i][i] = 1

    # vetor x solução
    x = np.zeros(elementos)
    # Resolvendo matriz triangular
    for i in range(elementos-1, -1, -1):
        if A[i][i]==0:
            return
        x[i] = round(B[i]/A[i][i], 3)

        for j in range(0, i+1):
            B[j] = B[j]-(A[j][i]*x[i])

    return x


def montaA(n):
    # Criando matrizes nxn e juntando-as para formar uma 2nxn^2
    blocoHorizontal = np.array([]).reshape((n, 0))
    for j in range(n):
        blocoHorizontal = np.concatenate([blocoHorizontal, np.identity((n))], axis=1)
    blocoVertical = np.array([]).reshape((n, 0))
    for j in range(n):  # percorre as 3 matrizes do bloco inferior, raios x que percorrem na vertical
        matrizAux = np.zeros((n, n))
        for k in range(n):  # percorre cada linha da matrizAux
            if (j==k):
                matrizAux[k] = np.ones(n)
        blocoVertical = np.concatenate([blocoVertical, matrizAux], axis=1)
    A = np.concatenate([blocoHorizontal, blocoVertical])

    return A


def montaADois(n):
    # Criando matrizes nxn e juntando-as para formar uma 2nxn^2
    blocoHorizontal = np.array([]).reshape((n, 0))
    for j in range(n):
        blocoHorizontal = np.concatenate([blocoHorizontal, np.identity((n))], axis=1)
    blocoVertical = np.array([]).reshape((n, 0))
    for j in range(n):  # percorre as 3 matrizes do bloco inferior, raios x que percorrem na vertical
        matrizAux = np.zeros((n, n))
        for k in range(n):  # percorre cada linha da matrizAux
            if (j==k):
                matrizAux[k] = np.ones(n)
        blocoVertical = np.concatenate([blocoVertical, matrizAux], axis=1)
    A = np.concatenate([blocoHorizontal, blocoVertical])

    blocoDNE = np.array([]).reshape((2*n-1, 0))  # (6n-2 - 2n)/2=2n-1 linhas de submatriz
    for k in range(n):  # percorre as 3 submatrizes das diagonais cima direita (NE)
        matrizAux = np.zeros((2*n-1, n))  # (6n-2 - 2n)/2=2n-1 linhas de submatriz
        for i in range(2*n-1):  # percorre cada linha da matrizAux
            for j in range(n):  # percorre cada linha da matrizAux
                if (i+j==(
                        n-1)+k):  # buscando a diagional secundaria (soma dos indices i,j=n-1) de cada submatriz, defasando para baixo conforme cada matriz
                    matrizAux[i][j] = 1
        blocoDNE = np.concatenate([blocoDNE, matrizAux], axis=1)

    A = np.concatenate([A, blocoDNE])

    blocoDSE = np.array([]).reshape((2*n-1, 0))  # (6n-2 - 2n)/2=2n-1 linhas de submatriz
    for k in range(n):  # percorre as 3 submatrizes das diagonais baixo direita (SE)
        matrizAux = np.zeros((2*n-1, n))  # (6n-2 - 2n)/2=2n-1 linhas de submatriz
        for i in range(2*n-1):  # percorre cada linha da matrizAux
            for j in range(n):  # percorre cada linha da matrizAux
                if (
                        i==j+k):  # buscando a diagional secundaria (soma dos indices i,j=n-1) de cada submatriz, defasando para baixo conforme cada matriz
                    matrizAux[i][j] = 1
        blocoDSE = np.concatenate([blocoDSE, matrizAux], axis=1)

    A = np.concatenate([A, blocoDSE])

    return A


def main():
    for imagem in range(1, 4):
        for delta in [0, 0.1, 0.01, 0.001]:
            # Leitura dos dados
            vetorP1 = load("im"+str(imagem)+"/p1.npy")
            n = int(len(vetorP1)/2)
            A = montaA(n)
            # leitura da imagem e transformação em matriz
            img = Image.open("im"+str(imagem)+"/im"+str(imagem)+".png")
            fEstrela = np.array(img.convert("L"))

            # Operações com a matriz A
            mmqA = np.dot(np.transpose(A), A)+delta*np.identity(n**2)
            print('Tomografia '+str(imagem)+" com delta= "+str(delta)+" tem determinante= "+str(np.linalg.det(mmqA)))
            mmqB = np.dot(vetorP1, A)

            try:  # Verifica se é possível resolver o sistema, porque quando delta=0 o determinante de A pode ser nulo
                #f = np.linalg.solve(mmqA, mmqB) # solver do numpy para comparação
                f = solver(mmqA, mmqB)

                matrizF = np.zeros((n, n))
                indice = 0
                for j in range(n):
                    for i in range(n):
                        matrizF[i][j] = f[indice]
                        indice = indice+1

                matrizF = (255/np.max(f))*matrizF  # Colocando matriz na mesma escala de f*

                fig, toms = plt.subplots(3)

                fig.suptitle('Tomografia '+str(imagem)+" com delta= "+str(delta))
                toms[0].imshow(fEstrela, cmap='gray', vmin=0, vmax=255)
                toms[1].imshow(matrizF, cmap='gray', vmin=0, vmax=255)
                toms[0].set_title("Imagem Original")
                toms[1].set_title("Tomografia com cortes em 2 direções")
            except:
                img = Image.open("im"+str(imagem)+"/im"+str(imagem)+".png")

                fig, toms = plt.subplots(3)

                fig.suptitle('Tomografia '+str(imagem)+" com delta= "+str(delta))
                toms[0].imshow(fEstrela, cmap='gray', vmin=0, vmax=255)
                toms[0].set_title("Imagem Original")
                toms[1].set_title("Determinante de A=0, imagem indisponível")

            # ------------------------------Adicionando cortes na diagonal----------------
            vetorP2 = load("im"+str(imagem)+"/p2.npy")
            n = int((len(vetorP2)+2)/6)
            A = montaADois(n)

            mmqA = np.dot(np.transpose(A), A)+delta*np.identity(n**2)
            print('Tomografia '+str(imagem)+" com delta= "+str(delta)+" tem determinante= "+str(np.linalg.det(mmqA)))
            mmqB = np.dot(vetorP2, A)
            f = solver(mmqA, mmqB)
            f = (255/np.max(f))*f  # Colocando matriz na mesma escala de f*
            # f = np.linalg.solve(mmqA, mmqB)

            matrizF = np.zeros((n, n))
            indice = 0
            erroQuadratico = 0
            somaQuadratica = 0
            for j in range(n):
                for i in range(n):
                    matrizF[i][j] = f[indice]
                    erroQuadratico = (f[indice]-fEstrela[i][j])**2+erroQuadratico
                    somaQuadratica = fEstrela[i][j]**2+somaQuadratica
                    indice = indice+1

            erro = 100*np.sqrt(erroQuadratico)/np.sqrt(somaQuadratica)
            print('Tomografia '+str(imagem)+" com delta= "+str(delta)+" tem erro= "+str(erro))

            toms[2].imshow(matrizF, cmap='gray', vmin=0, vmax=255)
            toms[2].set_title("Tomografia com cortes em 4 direções")
            toms[2].annotate("Erro Quadrático= {:.2f} %".format(erro), (280, 5), xycoords='figure pixels')
            fig.tight_layout()
            plt.show()
            fig.savefig('figs/Tomografia '+str(imagem)+" com delta= " + str(delta) + ".png")


if __name__=='__main__':
    main()
