 /*
	Program: mst_seq.c
	Description: Implements the Algorithm for generating tree of minimum cost.
	Developer: Jucele Vasconcellos
	Date: 01/06/2016

	Compilation: nvcc -arch sm_30 -o mst_cuda.exe mst_cuda.cu
	Execution:	./mst_cuda.exe input.txt output.txt
	
	Input data: this program reads a ghaph information like this
	8
	16
	4 5 0.35
	4 7 0.37
	5 7 0.28
	0 7 0.16
	1 5 0.32
	0 4 0.38
	2 3 0.17
	1 7 0.19
	0 2 0.26
	1 2 0.36
	1 3 0.29
	2 7 0.34
	6 2 0.40
	3 6 0.52
	6 0 0.58
	6 4 0.93

	where the first line represents the number of vertices,
			the second line represents the number of edges and
			the subsequent lines are the edges in the format v1 v2 weight
*/

#include <stdio.h> // printf
#include<stdbool.h> // true, false
#include <stdlib.h> //malloc
#include <time.h> //clock
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

// Grafo Original
typedef struct { 
	unsigned short v, u; 
} aresta_go;

typedef struct { 
	int n, m;
	aresta_go *vertices_arestas;
	float *custos_arestas;
} grafo_original;


//Grafo Bipartido
typedef struct { 
	unsigned short ind_v;
	int ind_u; 
	int ind_ac; // indice da outra aresta correspondente
} aresta_gb;

typedef struct { 
	unsigned short id;
	int menorAresta; 
} vertice_v;

typedef struct { 
	int ind_ago; // indice para aresta do grafo original que deu origem a este vértice 
} vertice_u;

typedef struct { 
	int n_v, n_u, m;
	vertice_v *vertices_v;
	vertice_u *vertices_u;
	aresta_gb *arestas;
} grafo_bipartido;


// Strut
typedef struct { 
	int ind_ago; // indice para aresta do grafo original que deu origem a este vértice 
	unsigned int grau;
	int inda1, inda2;
} vertice_u_strut;

typedef struct { 
	int ind_v, ind_u; 
	int ind_agb; // indice da aresta no grafo bipartido
	int ind_acgb; // indice da aresta correspondente no grafo bipartido
} aresta_strut;

typedef struct { 
	int n_v, n_u, m;
	vertice_u_strut *vertices_u;
	aresta_strut *arestas;
} strut;

typedef struct {
	int v1, v2;
} aresta_E;

////////////////////////////////////
//Texturas
////////////////////////////////////
texture<float, 1, cudaReadModeElementType> tex_custos_arestasGO;

// Funções e Procedimentos

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line );
        exit(EXIT_FAILURE);
    }
}

#define CHECK_ERROR(err) (HandleError(err, __FILE__, __LINE__))

struct Cmp1 {
	__host__ __device__ bool operator()(const aresta_gb &aresta1, const aresta_gb &aresta2) 
	{
		return (aresta1.ind_v < aresta2.ind_v);
	}
};

__global__ void CriaGrafoBipartido(aresta_go*, int, int, aresta_gb*, vertice_v*, vertice_u*);
__global__ void ContaOcorrencias_u(aresta_gb*, unsigned int*, int);
__global__ void OrdenaArestasGB_u(aresta_gb*, int, unsigned int*, aresta_gb*);
__global__ void CorrigeInd_ac(aresta_gb*, int, aresta_gb*);
__global__ void AtualizaC(aresta_E*, int*, int*, int); 
__global__ void MarcarArestas1(aresta_gb*, int*, int*, int, int, int);
__global__ void MarcarArestas2(vertice_u*, aresta_gb*, int*, int*, int, int, int, int);
__global__ void EncontraMenorAresta(vertice_u*, aresta_gb*, int*, int);
__global__ void AtualizaMenorAresta(vertice_v*, int*, int);
__global__ void CriaVertices_u(vertice_u*, vertice_u*, aresta_gb*, int); 
__global__ void CriaVertices_v(vertice_v*, int*, int);
__global__ void CriaVertices_u_Strut(vertice_u_strut*, vertice_u*, int n_u);
__global__ void CriaArestas_Strut(aresta_strut*, aresta_gb*, vertice_u_strut*, vertice_v*, int);

grafo_original LeGrafo(char *);

// Função Principal
int main (int argc, char** argv){
	grafo_original GO;
	grafo_bipartido GB, H;
	strut S;
	double tempoTotal, tempo1, tempo2;
	double tempo1p, tempo2p;
	int *SolutionEdgeSet;
	int SolutionSize, i, it, aux;
	double SolutionVal;
	int num_zerodiff;
	int *CD;
	FILE *Arq;
		
	int dimBloco, dimGrid;
	aresta_go *d_vertices_arestasGO;
	aresta_gb *d_arestasGB;
	vertice_v *d_vertices_v;
	vertice_u *d_vertices_u, *d_novo_vertices_u;
	aresta_strut *d_arestas_Strut;
	vertice_u_strut *d_vertices_u_Strut;
	
	// d_C e d_B usados na ordenação
	unsigned int *d_C; //Declara d_C para armazenar as ContaOcorrencias
	aresta_gb *d_B; //Declara d_B para armazenar as arestas ordenadas
	int *d_menorAresta;
	
	float *d_custos_arestasGO;
	
	
	// Passo 1: Verificação de parâmetros
	// Passo 2: Leitura dos dados do grafo 
	// Passo 3: Criação do grafo bipartido correspondente às arestas recebidas
	// Passo 4: Encontra a solução
		// Passo 4.1: Escolher arestas que comporão a strut
		// Passo 4.2: Calcular o num_zero_diff e computar novas componenetes conexas
		// Passo 4.3: Compactar o grafo
	
	
	// ==============================================================================
	// Passo 1: Verificação de parâmetros
	// ==============================================================================
	
	//Verificando os parametros
	if(argc < 3 ){
	   printf( "\nParametros incorretos\n Uso: ./cms_seq.exe <ArqEntrada> <ArqSaida> <dimBloco> <S/N> onde:\n" );
	   printf( "\t <ArqEntrada> (obrigatorio) - Nome do arquivo com as informações do grafo (número de vértices, número de arestas e custos das arestas.\n" );
		printf( "\t <ArqSaida> (obrigatorio) - Nome do arquivo de saida.\n" );
		printf( "\t <S ou N> - Mostrar ou não as arestas da MST.\n" );
		return 0;
	} 	
	dimBloco = 32;
	
	// ==============================================================================
	// Passo 2: Leitura dos dados do Grafo G
	// ==============================================================================
	tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
	GO = LeGrafo(argv[1]);

	printf("Grafo de entrada lido\n");
	SolutionEdgeSet = (int *) malloc((GO.n-1)*sizeof(int)); 
	SolutionSize = 0;
	SolutionVal = 0;
	tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
	printf("Tempo Passo 2: %lf\n", tempo2p - tempo1p);
	
	// ==============================================================================
	// Passo 3: Transforma em grafo bipartido
	// ==============================================================================
	//Iniciando contagem do tempo
	tempo1 = (double) clock( ) / CLOCKS_PER_SEC;
	tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
	
	GB.n_v = GO.n;
	GB.n_u = GO.m;
	GB.m = GO.m * 2;
	
	//Aloca memória no device para as arestas do grafo original
	CHECK_ERROR(cudaMalloc((void **) &d_vertices_arestasGO, GO.m * sizeof(aresta_go)));
	//Copia as arestas do grafo original do host para o device
	CHECK_ERROR(cudaMemcpy(d_vertices_arestasGO, GO.vertices_arestas, GO.m * sizeof(aresta_go), cudaMemcpyHostToDevice));
	//Aloca memória no device para as custos das arestas do grafo original
	CHECK_ERROR(cudaMalloc((void **) &d_custos_arestasGO, GO.m * sizeof(float)));
	//Copia as custos das arestas do grafo original do host para o device
	CHECK_ERROR(cudaMemcpy(d_custos_arestasGO, GO.custos_arestas, GO.m * sizeof(float), cudaMemcpyHostToDevice));
	//Liga a textura as arestas do grafo original
	CHECK_ERROR(cudaBindTexture(0, tex_custos_arestasGO, d_custos_arestasGO, GO.m * sizeof(float)));

	//Aloca memória no device para as arestas do grafo bipartido
	CHECK_ERROR(cudaMalloc((void **) &d_arestasGB, GB.m * sizeof(aresta_gb)));
	CHECK_ERROR(cudaMalloc((void **) &d_vertices_v, GO.n * sizeof(vertice_v)));
	CHECK_ERROR(cudaMalloc((void **) &d_vertices_u, GO.m * sizeof(vertice_u)));

	//Aloca memória para copiar arestas para host
	GB.arestas = (aresta_gb *) malloc(GB.m*sizeof(aresta_gb)); 
	GB.vertices_v = (vertice_v *) malloc(GB.n_v*sizeof(vertice_v)); 
	GB.vertices_u = (vertice_u *) malloc(GB.n_u*sizeof(vertice_u)); 
	
	//Define a dimensão para o grid
	dimGrid = ((GO.m-1)/dimBloco)+1;
	//Cria grafo bipartido na GPU
	CriaGrafoBipartido<<<dimGrid, dimBloco>>>(d_vertices_arestasGO, GO.m, GO.n, d_arestasGB, d_vertices_v, d_vertices_u);
	
	//Aloca memória para d_B que será usada em ordenação posterior
 	CHECK_ERROR(cudaMalloc((void **) &d_B, GB.m * sizeof(aresta_gb)));
	
	//Encontra MenorAresta para cada vértice v
	//Aloca memória para d_menorAresta
	CHECK_ERROR(cudaMalloc((void **) &d_menorAresta, GB.n_v * sizeof(int)));
	//Inicializa d_menorAresta com -1
 	CHECK_ERROR(cudaMemset(d_menorAresta, -1, GB.n_v * sizeof(int)));
	//Chama kernel para encontrar menorAresta de cada v
 	dimGrid = ((GB.m-1)/dimBloco)+1;
	EncontraMenorAresta<<<dimGrid, dimBloco>>>(d_vertices_u, d_arestasGB,  d_menorAresta, GB.m);
	
	CHECK_ERROR(cudaDeviceSynchronize());
	
	dimGrid = ((GB.n_v-1)/dimBloco)+1;
	AtualizaMenorAresta<<<dimGrid, dimBloco>>>(d_vertices_v, d_menorAresta, GB.n_v);
	
	//Copia arestas do grafo bipartido para o host
	CHECK_ERROR(cudaMemcpy(GB.arestas, d_arestasGB, GB.m * sizeof(aresta_gb), cudaMemcpyDeviceToHost));
// 	CHECK_ERROR(cudaMemcpy(d_arestasGB, d_B, GB.m * sizeof(aresta_gb), cudaMemcpyDeviceToDevice));
	CHECK_ERROR(cudaMemcpy(GB.vertices_u, d_vertices_u, GB.n_u * sizeof(vertice_u), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(GB.vertices_v, d_vertices_v, GB.n_v * sizeof(vertice_v), cudaMemcpyDeviceToHost));
	//Libera memória de variáveis
	CHECK_ERROR(cudaFree(d_vertices_arestasGO));	


	tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
	printf("Tempo Passo 3: %lf\n", tempo2p - tempo1p);

	
	// ==============================================================================
	// Passo 4: Encontra solução
	// ==============================================================================
	
	it = 0;
	num_zerodiff = 0;
	while (SolutionSize < (GO.n-1))
	{
   		printf("***** Iteração %d ****\n", it);
   		printf("\t Número de fragmentos = %d   Número de arestas = %d    SolutionSize = %d    SolutionVal = %lf\n", GB.n_v, GB.m, SolutionSize, SolutionVal);

		// ==============================================================================
		// Passo 4.1: Escolher arestas que comporão a strut
		// ==============================================================================
		tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
		
		// Coloca dados das arestas escolhidas na estrutura S
		S.n_v = GB.n_v;
		S.n_u = GB.n_u;
		S.m = GB.n_v;
	
		S.vertices_u = (vertice_u_strut *) malloc(S.n_u*sizeof(vertice_u_strut)); 
		S.arestas = (aresta_strut *) malloc(S.m*sizeof(aresta_strut)); 
		
		CHECK_ERROR(cudaMalloc((void **) &d_vertices_u_Strut, S.n_u * sizeof(vertice_u_strut)));
		CHECK_ERROR(cudaMalloc((void **) &d_arestas_Strut, S.m * sizeof(aresta_strut)));
		//Chama kernel para criar vertices_u da Strut
		dimGrid = ((S.n_u-1)/dimBloco)+1;
		CriaVertices_u_Strut<<<dimGrid, dimBloco>>>(d_vertices_u_Strut, d_vertices_u, S.n_u);
		//Chama kernel para criar arestas da Strut
		dimGrid = ((S.m-1)/dimBloco)+1;
		CriaArestas_Strut<<<dimGrid, dimBloco>>>(d_arestas_Strut, d_arestasGB, d_vertices_u_Strut, d_vertices_v, S.m);
		
		CHECK_ERROR(cudaMemcpy(S.vertices_u, d_vertices_u_Strut, S.n_u * sizeof(vertice_u_strut), cudaMemcpyDeviceToHost));
		CHECK_ERROR(cudaFree(d_vertices_u_Strut));
		CHECK_ERROR(cudaMemcpy(S.arestas, d_arestas_Strut, S.m * sizeof(aresta_strut), cudaMemcpyDeviceToHost));
		CHECK_ERROR(cudaFree(d_arestas_Strut));
 	
		tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
		printf("Tempo Passo 4.1: %lf\n", tempo2p - tempo1p);
		
		// ==============================================================================
		// Passo 4.2: Calcular o num_zero_diff
		// ==============================================================================
		tempo1p = (double) clock( ) / CLOCKS_PER_SEC;

		num_zerodiff = 0;
		for(i = 0; i < S.n_u; i++)
		{
			if(S.vertices_u[i].grau > 0)
			{
				SolutionEdgeSet[SolutionSize] = S.vertices_u[i].ind_ago;
				SolutionVal += GO.custos_arestas[S.vertices_u[i].ind_ago];
				SolutionSize++;
				if (S.vertices_u[i].grau == 2)
					num_zerodiff++;
			}
		} // end for(i = 0; i < S.n_u; i++)
 
		tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
		printf("Tempo Passo 4.2: %lf     num_zerodiff = %d     SolutionSize = %d\n", tempo2p - tempo1p, num_zerodiff, SolutionSize);
		
		// ==============================================================================
		// Passo 4.3: Compactar o grafo
		// ==============================================================================
		if(SolutionSize < (GO.n-1))
		{
			tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
			
			// ==============================================================================
			// Passo 4.3.1: Computar componenetes conexas
			// ==============================================================================
			tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
			CD = (int *) malloc(GB.n_v*sizeof(int)); 
			aresta_E *h_arestasE, *d_arestasE;
			h_arestasE = (aresta_E *) malloc(GB.n_v*sizeof(aresta_E)); 
			CHECK_ERROR(cudaMalloc((void **) &d_arestasE, GB.n_v * sizeof(aresta_E)));
			
			//Inicializa CD e h_arestasE
			for(i = 0; i < GB.n_v; i++)
			{
				h_arestasE[i].v1 = S.arestas[i].ind_v;
				h_arestasE[i].v2 = GB.arestas[S.arestas[i].ind_acgb].ind_v;
				CD[i] = h_arestasE[i].v1;
			}
			//Copia arestas_E do host para o device
			CHECK_ERROR(cudaMemcpy(d_arestasE, h_arestasE, GB.n_v * sizeof(aresta_E), cudaMemcpyHostToDevice));
			//Cria d_CD
			int *d_CD;
			CHECK_ERROR(cudaMalloc((void **) &d_CD, GB.n_v * sizeof(int)));
			CHECK_ERROR(cudaMemcpy(d_CD, CD, GB.n_v * sizeof(int), cudaMemcpyHostToDevice));
			int h_fim, *d_fim;
			CHECK_ERROR(cudaMalloc((void**)&d_fim, sizeof(int)));
			do
			{
				h_fim = 0;
				CHECK_ERROR(cudaMemcpy(d_fim, &h_fim, sizeof(int), cudaMemcpyHostToDevice));
				dimGrid = ((GB.n_v-1)/dimBloco)+1;
				AtualizaC<<<dimGrid, dimBloco>>>(d_arestasE, d_CD, d_fim, GB.n_v);
				CHECK_ERROR(cudaMemcpy(&h_fim, d_fim, sizeof(int), cudaMemcpyDeviceToHost));		
			}while (h_fim == 1);
			
			CHECK_ERROR(cudaMemcpy(CD, d_CD, GB.n_v * sizeof(int), cudaMemcpyDeviceToHost));
			aux = 1;
			for(i = 1; i < GB.n_v; i++)
			{
				if(CD[i] == i)
				{
					CD[i] = aux;
					aux++;
				}
				else
					CD[i] = CD[CD[i]];
			}
			CHECK_ERROR(cudaMemcpy(d_CD, CD, GB.n_v * sizeof(int), cudaMemcpyHostToDevice));

			tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
			printf("Tempo Passo 4.3.1: %lf\n", tempo2p - tempo1p);

			//Liberando variáveis
			free(S.vertices_u); 
			free(S.arestas);
			free(h_arestasE);
			CHECK_ERROR(cudaFree(d_arestasE));
			
			// ==============================================================================
			// Passo 4.3.2: Marcar arestas
			// ==============================================================================
			tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
			
			H.m = GB.m;
			int *d_mH;
 			
			//Aloca memória para d_mH
			CHECK_ERROR(cudaMalloc((void **) &d_mH, sizeof(int)));
			//Copia H.m para d_mH
			CHECK_ERROR(cudaMemcpy(d_mH, &H.m, sizeof(int), cudaMemcpyHostToDevice));
			//Aloca custos para todos os n_v
			//CHECK_ERROR(cudaMalloc((void **) &d_menorAresta, (GB.n_v * GB.n_v) * sizeof(int)));
			CHECK_ERROR(cudaMalloc((void **) &d_menorAresta, (num_zerodiff * num_zerodiff) * sizeof(int)));
			//Inicializa d_menorAresta com -1
			CHECK_ERROR(cudaMemset(d_menorAresta, -1, (num_zerodiff * num_zerodiff) * sizeof(int)));
				
			//Marca as arestas para remoção
			dimGrid = ((GB.m-1)/dimBloco)+1;
			MarcarArestas1<<<dimGrid, dimBloco>>>(d_arestasGB, d_CD, d_mH, GB.m, GB.n_v, GB.n_u);
			
			MarcarArestas2<<<dimGrid, dimBloco>>>(d_vertices_u, d_arestasGB, d_mH, d_menorAresta, GB.m, GB.n_v, GB.n_u, num_zerodiff);
			
			CHECK_ERROR(cudaDeviceSynchronize());
			
			//Copia d_mH para H.m
			CHECK_ERROR(cudaMemcpy(&H.m, d_mH, sizeof(int), cudaMemcpyDeviceToHost));
			
			CHECK_ERROR(cudaFree(d_mH));
			CHECK_ERROR(cudaFree(d_CD));
			CHECK_ERROR(cudaFree(d_menorAresta));
			
			tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
			printf("Tempo Passo 4.3.2: %lf\n", tempo2p - tempo1p);		
			
			// ==============================================================================
			// Passo 4.3.3: Gerar novo grafo bipartido
			// ==============================================================================
			tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
			
			//Ordenar todas (G.m) as arestas de GB que podem ter G.n_u+1 valores
			//Aloca memória para d_C
			CHECK_ERROR(cudaMalloc((void **) &d_C, (GB.n_u+1) * sizeof(unsigned int)));
			//Inicializa d_C com 0
			CHECK_ERROR(cudaMemset(d_C, 0, (GB.n_u+1) * sizeof(unsigned int)));
			//Conta o número de ocorrências de cada v
			dimGrid = ((GB.m-1)/dimBloco)+1;
			ContaOcorrencias_u<<<dimGrid, dimBloco>>>(d_arestasGB, d_C, GB.m);
			//Atualiza d_C com complemento
			thrust::device_ptr<unsigned int>d_C_ptr = thrust::device_pointer_cast(d_C);
			thrust::inclusive_scan(d_C_ptr, d_C_ptr + (GB.n_u+1), d_C_ptr);	
			//Ordena as arestas
			dimGrid = ((GB.m-1)/dimBloco)+1;
			OrdenaArestasGB_u<<<dimGrid, dimBloco>>>(d_arestasGB, GB.m, d_C, d_B);
			dimGrid = ((H.m-1)/dimBloco)+1;
			CorrigeInd_ac<<<dimGrid, dimBloco>>>(d_arestasGB, H.m, d_B);
			//Copia arestas do grafo bipartido para o host
			CHECK_ERROR(cudaMemcpy(d_arestasGB, d_B, GB.m * sizeof(aresta_gb), cudaMemcpyDeviceToDevice));
			CHECK_ERROR(cudaMemcpy(GB.arestas, d_arestasGB, H.m * sizeof(aresta_gb), cudaMemcpyDeviceToHost));
			//Libera variáveis
			CHECK_ERROR(cudaFree(d_C));

			
			//Criar o vetor vertices_u
			H.n_u = H.m/2;
			free(GB.vertices_u);
			//Aloca memória para GB.vertices_u
			GB.vertices_u = (vertice_u *) malloc(H.n_u*sizeof(vertice_u)); 
			//Aloca memória para d_novo_vertices_u
			CHECK_ERROR(cudaMalloc((void **) &d_novo_vertices_u, H.n_u * sizeof(vertice_u)));
			//Chama kernel para criar o vetor vertices_u
			dimGrid = ((H.m-1)/dimBloco)+1;
			CriaVertices_u<<<dimGrid, dimBloco>>>(d_vertices_u, d_novo_vertices_u, d_arestasGB, H.m);
			//Copia vertices_u do grafo bipartido para o host
			CHECK_ERROR(cudaMemcpy(GB.vertices_u, d_novo_vertices_u, H.n_u * sizeof(vertice_u), cudaMemcpyDeviceToHost));
			CHECK_ERROR(cudaMemcpy(d_vertices_u, d_novo_vertices_u, H.n_u * sizeof(vertice_u), cudaMemcpyDeviceToDevice));

			
			//Cria vetor GB.vertices_v
			H.n_v = num_zerodiff;
			free(GB.vertices_v);
			GB.vertices_v = (vertice_v *) malloc(H.n_v*sizeof(vertice_v)); 
			//Aloca espaço para variável d_menorAresta
			CHECK_ERROR(cudaMalloc((void **) &d_menorAresta, GB.n_v * sizeof(int)));
			//Inicializa d_menorAresta com -1
			CHECK_ERROR(cudaMemset(d_menorAresta, -1, GB.n_v * sizeof(int)));
			//Calcula dimensão do grid
			dimGrid = ((H.m-1)/dimBloco)+1;
			//Chama kernel EncontraMenorAresta
			EncontraMenorAresta<<<dimGrid, dimBloco>>>(d_novo_vertices_u, d_arestasGB, d_menorAresta, H.m);
			//Calcula dimensão do grid
			dimGrid = ((H.n_v-1)/dimBloco)+1;
			//Chama kernel CriaVertices_v
			CriaVertices_v<<<dimGrid, dimBloco>>>(d_vertices_v, d_menorAresta, H.n_v);
			//Copia vertices_u do grafo bipartido do device para o host
			CHECK_ERROR(cudaMemcpy(GB.vertices_v, d_vertices_v, H.n_v * sizeof(vertice_v), cudaMemcpyDeviceToHost));
			//Libera variáveis
			CHECK_ERROR(cudaFree(d_menorAresta));

			
			GB.m = H.m;
			GB.n_v = H.n_v;
			GB.n_u = H.n_u;
			
			tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
			printf("Tempo Passo 4.3.3: %lf\n", tempo2p - tempo1p);		
		}
		it++;
	} // fim while
	
	tempo2 = (double) clock( ) / CLOCKS_PER_SEC;
	tempoTotal = tempo2 - tempo1;

	printf("\nSolutionSize = %d      Custo total da MST: %lf\n", SolutionSize, SolutionVal);
	printf("Tempo Total: %lf\n", tempoTotal); 

	Arq = fopen(argv[2], "a");
 	fprintf(Arq, "\n*** Arquivo de entrada: %s\n", argv[1]); 
	fprintf(Arq, "*** Custo total da MST: %lf\n", SolutionVal);
	fprintf(Arq, "Tempo Total: %lf\n", tempoTotal); 
	fprintf(Arq, "Número de iterações: %d\n", it);
	fprintf(Arq, "SolutionSize: %d\n", SolutionSize);

  	if((argc == 4) && (argv[3][0] == 'S' || argv[3][0] == 's'))
	{
  		fprintf(Arq, "*** MST formada pelas %d arestas\n", SolutionSize);
  		for(i = 0; i < SolutionSize; i++)
  			fprintf(Arq, "Aresta %d - %d = %lf\n", GO.vertices_arestas[SolutionEdgeSet[i]].v, GO.vertices_arestas[SolutionEdgeSet[i]].u, GO.custos_arestas[SolutionEdgeSet[i]]);
  	}
  	fclose(Arq);
	
	free(GO.vertices_arestas);
	free(GO.custos_arestas);
	free(GB.arestas);
	free(SolutionEdgeSet);
	
	CHECK_ERROR(cudaFree(d_vertices_v));
	CHECK_ERROR(cudaFree(d_vertices_u));		
	CHECK_ERROR(cudaFree(d_novo_vertices_u));
	CHECK_ERROR(cudaFree(d_custos_arestasGO));	
	CHECK_ERROR(cudaFree(d_arestasGB));	
	CHECK_ERROR(cudaFree(d_B));	
	
	return 0;

}


// ==============================================================================
// Função LeGrafo:  Lê as informações do Grafo de um arquivo e armazena em uma 
//                  estrutura
// ==============================================================================
grafo_original LeGrafo(char *Arquivo){
	int i, aux;
	grafo_original G;
   FILE *Arq;
    
   Arq = fopen(Arquivo, "r");

   i = 0;
	fscanf(Arq,"%d",&i);
	G.n = i;
	
	fscanf(Arq,"%d",&i);
	G.m = i;
	
	G.vertices_arestas = (aresta_go *) malloc(G.m*sizeof(aresta_go));
	G.custos_arestas = (float *) malloc(G.m*sizeof(float));
	
	for(i = 0; i < G.m; i++){
		fscanf(Arq,"%hu",&G.vertices_arestas[i].u);
		fscanf(Arq,"%hu",&G.vertices_arestas[i].v);
		if(G.vertices_arestas[i].v > G.vertices_arestas[i].u)
		{
			aux = G.vertices_arestas[i].v;
			G.vertices_arestas[i].v = G.vertices_arestas[i].u;
			G.vertices_arestas[i].u = aux;
		}
		fscanf(Arq,"%f",&G.custos_arestas[i]);
	}
	
	fclose(Arq);
   return G;
}


// ==============================================================================
// Função CriaGrafoBipartido: Criação do grafo bipartido correspondente às 
//                            arestas recebidas
// ==============================================================================
__global__ void CriaGrafoBipartido(aresta_go* arestasGO, int mGO, int nGO, aresta_gb* arestasGB, vertice_v *vertices_v, vertice_u *vertices_u) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(id < mGO)
	{
		if(id < nGO)
		{
			vertices_v[id].id = id;
		}
		vertices_u[id].ind_ago = id;

		arestasGB[2*id].ind_v = arestasGO[id].v;
		arestasGB[2*id].ind_u = id;
		arestasGB[2*id].ind_ac = 2*id+1;
		
		arestasGB[2*id+1].ind_v = arestasGO[id].u;
		arestasGB[2*id+1].ind_u = id;
		arestasGB[2*id+1].ind_ac = 2*id;
	}
}


// ==============================================================================
// Função ContaOcorrencias_v:  Conta o número de arestas para cada vertice v
// ==============================================================================
__global__ void ContaOcorrencias_v(aresta_gb* arestasGB, unsigned int* C, int n)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if(id < n)
	{
		atomicInc(&C[arestasGB[id].ind_v], UINT_MAX);
	}	
}

// ==============================================================================
// Função ContaOcorrencias_u:  Conta o número de arestas para cada vertice u
// ==============================================================================
__global__ void ContaOcorrencias_u(aresta_gb* arestasGB, unsigned int* C, int n)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if(id < n)
	{
		atomicInc(&C[arestasGB[id].ind_u], UINT_MAX);
	}	
}


// ==============================================================================
// Função OrdenaArestasGB_u:  Ordena as arestas do grafo bipartido pelo vértice u, 
//                        do menor para o maior, utilizando CountSort
// ==============================================================================
__global__ void OrdenaArestasGB_u(aresta_gb* arestasGB, int n, unsigned int *C, aresta_gb* B) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n )
	{
		unsigned int i = atomicSub(&C[arestasGB[id].ind_u], 1);
		i--;
		B[i] = arestasGB[id];
		arestasGB[id].ind_ac = i;
	}
}


// ==============================================================================
// Função CorrigeInd_ac:  Corrige os indices das arestas correspondentes
// ==============================================================================
__global__ void CorrigeInd_ac(aresta_gb* arestasGB, int n, aresta_gb* B) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n )
	{
		B[id].ind_ac = arestasGB[B[id].ind_ac].ind_ac;
	}
}


// ==============================================================================
// Função MarcarArestas:  Marca as arestas do grafo a serem removidas
// ==============================================================================
__global__ void MarcarArestas1(aresta_gb *arestasGB, int* C, int *mH, int mGB, int n_v, int n_u)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int x, y;
	
	if( id < mGB )
	{
		if(id < arestasGB[id].ind_ac)
		{
			x = C[arestasGB[id].ind_v];
			y = C[arestasGB[arestasGB[id].ind_ac].ind_v];
			
			if(x == y)
			{
//  				printf("1 - Marcar arestas (v1 = %d   v2 = %d   ago=%d   custo=%f)   x=y=%d   id = %d   ind_ac = %d   thread = %d \n", arestasGB[id].ind_v, arestasGB[arestasGB[id].ind_ac].ind_v, vertices_u[arestasGB[id].ind_u].ind_ago, arestasGO[vertices_u[arestasGB[id].ind_u].ind_ago].custo, x, id, arestasGB[id].ind_ac, id);
				arestasGB[id].ind_v = n_v;
				arestasGB[arestasGB[id].ind_ac].ind_v = n_v;
				arestasGB[id].ind_u = n_u;
				arestasGB[arestasGB[id].ind_ac].ind_u = n_u;
				atomicSub(&mH[0], 2);
// 				printf("finalizando id = %d    x = %d   y = %d   mH=%d\n", id, x, y, mH[0]);
			}
			else
			{
				arestasGB[id].ind_v = x;
				arestasGB[arestasGB[id].ind_ac].ind_v = y;
			}
		} //fim if(id < arestasGB[id].ind_ac)
	} //fim if( id < mGB )
}

// ==============================================================================
// Função MarcarArestas:  Marca as arestas do grafo a serem removidas
// ==============================================================================
__global__ void MarcarArestas2(vertice_u* vertices_u, aresta_gb* arestasGB, int* mH, int* custos, int mGB, int n_v, int n_u, int num_zerodiff)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int x, y, aux;
	int old_id;
	float old_custo, meu_custo;
	
	if( id < mGB )
	{
		if(id < arestasGB[id].ind_ac)
		{
			if(arestasGB[id].ind_v < n_v)
			{
				x = arestasGB[id].ind_v;
				y = arestasGB[arestasGB[id].ind_ac].ind_v;
				
				if(x > y)
				{
					aux = x;
					x = y;
					y = aux;
				}

				aux = -1;
				old_id = atomicCAS(&custos[x*num_zerodiff+y], aux, id);
 				//old_custo = arestasGO[vertices_u[arestasGB[old_id].ind_u].ind_ago].custo;
				old_custo = tex1Dfetch(tex_custos_arestasGO, vertices_u[arestasGB[old_id].ind_u].ind_ago);
				
 				if(old_id != aux)
 				{
					//meu_custo = arestasGO[vertices_u[arestasGB[id].ind_u].ind_ago].custo;
					meu_custo = tex1Dfetch(tex_custos_arestasGO, vertices_u[arestasGB[id].ind_u].ind_ago);
					
					while((old_custo > meu_custo) && (old_id != aux))
					{
						aux = atomicCAS(&custos[x*num_zerodiff+y], old_id, id);
						if(old_id != aux)
						{
							old_id = aux;
							aux = -1;
							//old_custo = arestasGO[vertices_u[arestasGB[old_id].ind_u].ind_ago].custo;
							old_custo = tex1Dfetch(tex_custos_arestasGO, vertices_u[arestasGB[old_id].ind_u].ind_ago);
						}
					}
			
					if(old_custo <= meu_custo)
					{
						arestasGB[id].ind_v = n_v;
						arestasGB[arestasGB[id].ind_ac].ind_v = n_v;
						arestasGB[id].ind_u = n_u;
						arestasGB[arestasGB[id].ind_ac].ind_u = n_u;
						atomicSub(&mH[0], 2);
					}
				} // fim if(id_old_custo1 != -1)
			} //fim if(arestasGB[id].ind_v < n_v)
		} //fim if(id < arestasGB[id].ind_ac)
	} //fim if( id < mGB )
}



// ==============================================================================
// Função AtualizaC: Atualiza vetor C para definição das componentes conexas
// ==============================================================================
__global__ void AtualizaC(aresta_E *arestasE, int* C, int *m, int n) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int c1, c2, v1, v2;
	
	if(id < n)
	{
		v1 = arestasE[id].v1;
		v2 = arestasE[id].v2;
		c1 = C[v1];
		c2 = C[v2];
		if(c1 < c2)
		{
			atomicMin(&C[v2], c1);
			m[0] = 1;
		}
		else if(c2 < c1)
		{
			atomicMin(&C[v1], c2);
			m[0] = 1;
		}
	}
}


// ==============================================================================
// Função EncontraMenorAresta:  Para cada vértice v encontra o id da aresta de menor peso
// ==============================================================================
__global__ void EncontraMenorAresta(vertice_u* vertices_u, aresta_gb* arestasGB, int* menorAresta, int mGB)
{
	int meu_id = threadIdx.x + blockDim.x * blockIdx.x;
	int x, aux;
	int old_id;
	float old_custo, meu_custo;
	
	if( meu_id < mGB )
	{
		x = arestasGB[meu_id].ind_v;

		aux = -1;
		old_id = atomicCAS(&menorAresta[x], aux, meu_id);
				
 		if(old_id != aux)
 		{
			//meu_custo = arestasGO[vertices_u[arestasGB[meu_id].ind_u].ind_ago].custo;
			meu_custo = tex1Dfetch(tex_custos_arestasGO, vertices_u[arestasGB[meu_id].ind_u].ind_ago);
			//old_custo = arestasGO[vertices_u[arestasGB[old_id].ind_u].ind_ago].custo;
			old_custo = tex1Dfetch(tex_custos_arestasGO, vertices_u[arestasGB[old_id].ind_u].ind_ago);
				
			while(((old_custo > meu_custo) || ((old_custo == meu_custo) && (arestasGB[old_id].ind_u > arestasGB[meu_id].ind_u))) && (old_id != aux))
			{
				aux = atomicCAS(&menorAresta[x], old_id, meu_id);
				if(old_id != aux)
				{
					old_id = aux;
					aux = -1;
					//old_custo = arestasGO[vertices_u[arestasGB[old_id].ind_u].ind_ago].custo;
					old_custo = tex1Dfetch(tex_custos_arestasGO, vertices_u[arestasGB[old_id].ind_u].ind_ago);
				}
			}
		} // fim if(id_old_custo1 != -1)
	} //fim if( meu_id < mGB )
}


// ==============================================================================
// Função AtualizaMenorAresta:  Atualiza o vetor d_vertices_v com a menorAresta
// ==============================================================================
__global__ void AtualizaMenorAresta(vertice_v* vertices_v, int* menorAresta, int n_v)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n_v )
	{
		vertices_v[id].menorAresta = menorAresta[id];
	}	
}



// ==============================================================================
// Função CriaVertices_u:  Cria novo vetor vertices_u após compactação de arestas
// ==============================================================================
__global__ void CriaVertices_u(vertice_u* vertices_u, vertice_u* novo_vertices_u, aresta_gb* arestasGB, int H_m) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < H_m )
	{
		if((id % 2) == 0) 
		{
			novo_vertices_u[id/2].ind_ago = vertices_u[arestasGB[id].ind_u].ind_ago;
			arestasGB[id].ind_u = id/2;
			arestasGB[id+1].ind_u = id/2;
		}
	}
}

// ==============================================================================
// Função CriaVertices_v:  Cria novo vetor vertices_v após compactação de arestas
// ==============================================================================
__global__ void CriaVertices_v(vertice_v* vertices_v, int* menorAresta, int n_v)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n_v )
	{
		vertices_v[id].id = id;
		vertices_v[id].menorAresta = menorAresta[id];
	} //fim if( id < n_v )
}


// ==============================================================================
// Função CriaVertices_u_Strut:  Cria vetor vertices_u da Strut
// ==============================================================================
__global__ void CriaVertices_u_Strut(vertice_u_strut* vertices_u_Strut, vertice_u* vertices_u, int n_u)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n_u )
	{
		vertices_u_Strut[id].ind_ago = vertices_u[id].ind_ago;
		vertices_u_Strut[id].grau = 0;
		vertices_u_Strut[id].inda1 = -1;
		vertices_u_Strut[id].inda2 = -1;
	} //fim if( id < n_u )
}


// ==============================================================================
// Função CriaArestas_Strut:  Cria vetor de arestas da Strut
// ==============================================================================
__global__ void CriaArestas_Strut(aresta_strut* arestas_Strut, aresta_gb* arestasGB, vertice_u_strut* vertices_u_Strut, vertice_v* vertices_v, int m)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int j, old_grau;
	
	if( id < m )
	{
		j = vertices_v[id].menorAresta;
		arestas_Strut[id].ind_v = arestasGB[j].ind_v;
		arestas_Strut[id].ind_u = arestasGB[j].ind_u;
		arestas_Strut[id].ind_acgb = arestasGB[j].ind_ac;
		arestas_Strut[id].ind_agb = j;
		
		old_grau = atomicInc(&vertices_u_Strut[arestas_Strut[id].ind_u].grau, UINT_MAX);
		
		if(old_grau == 0)
			vertices_u_Strut[arestas_Strut[id].ind_u].inda1 = id;
		else
			vertices_u_Strut[arestas_Strut[id].ind_u].inda2 = id;
	} //fim if( id < n_u )
}
