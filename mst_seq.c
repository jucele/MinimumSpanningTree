 /*
	Program: mst_seq.c
	Description: Implements the Algorithm for generating tree of minimum cost.
	Developer: Jucele Vasconcellos
	Date: 01/06/2016

	Compilation:	gcc -o mst_seq.exe mst_seq.c
	Execution:	./mst_seq.exe input.txt output.txt
	
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

// Grafo Original
typedef struct { 
	unsigned short v, u; 
	float custo; 
} aresta_go;

typedef struct { 
	int n, m;
	aresta_go *arestas;
} grafo_original;


//Grafo Bipartido
typedef struct { 
	unsigned short ind_v;
	int ind_u; 
	int ind_ac; // indice da outra aresta correspondente
} aresta_gb;

typedef struct { 
	unsigned short id;
	int grau, menorAresta; 
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
	int grau;
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
    int ch, tam;
} uc;


// Funções e Procedimentos
grafo_original LeGrafo(char *);
void MostraGrafoOriginal(grafo_original);
aresta_go *OrdenaArestasGO_v_u(aresta_go*, int, int, bool);
grafo_bipartido CriaGrafoBipartido(grafo_original GO);
void MostraGrafoBipartido(grafo_bipartido, grafo_original, bool);
aresta_gb *OrdenaArestasGB_v_u(aresta_gb *, int, int, bool);
void OrdenaArestasGB_custo(aresta_gb *, int, int, grafo_original, grafo_bipartido);
strut GeraStrut(grafo_bipartido);
void MostraStrut(strut, bool);
grafo_bipartido CompactarGrafo(grafo_bipartido, grafo_original, uc *, int);
void CD_Inic(int, uc *);
int CD_chefe(int, uc *);
void CD_Uniao(int, int, uc *);

// Função Principal
int main (int argc, char** argv){
	grafo_original GO;
	grafo_bipartido GB, H;
	strut S;
	double tempoTotal, tempo1, tempo2;
	double tempo1p, tempo2p;
	int *SolutionEdgeSet;
	int SolutionSize, i, j, k;
	int it;
	double SolutionVal;
	int num_zerodiff;
	uc *CD;
	int x, y;
	FILE *Arq;
	
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
	   printf( "\nParametros incorretos\n Uso: ./cms_seq.exe <ArqEntrada> <ArqSaida onde:\n" );
	   printf( "\t <ArqEntrada> (obrigatorio) - Nome do arquivo com as informações do grafo (número de vértices, número de arestas e custos das arestas.\n" );
		printf( "\t <ArqSaida> (obrigatorio) - Nome do arquivo de saida.\n" );
		printf( "\t <S ou N> - Mostrar ou não as arestas da MST.\n" );

		return 0;
	} 	
	
	// ==============================================================================
	// Passo 2: Leitura dos dados do Grafo G
	// ==============================================================================
	tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
	GO = LeGrafo(argv[1]);
	//MostraGrafoOriginal(GO);
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
	GB = CriaGrafoBipartido(GO);
//   	printf("Grafo bipartido gerado\n");
	
// 	printf("Grafo bipartido inicial ordenado\n");
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

		
		//MostraGrafoBipartido(GB, GO, true);
		
		// ==============================================================================
		// Passo 4.1: Escolher arestas que comporão a strut
		// ==============================================================================
		tempo1p = (double) clock( ) / CLOCKS_PER_SEC;

		for(i = 0; i < GB.n_v; i++)
			GB.vertices_v[i].menorAresta = -1;
		
		for(i = 0; i < GB.m; i++)
			if((GB.vertices_v[GB.arestas[i].ind_v].menorAresta == -1) || (GO.arestas[GB.vertices_u[GB.arestas[GB.vertices_v[GB.arestas[i].ind_v].menorAresta].ind_u].ind_ago].custo > GO.arestas[GB.vertices_u[GB.arestas[i].ind_u].ind_ago].custo))
				GB.vertices_v[GB.arestas[i].ind_v].menorAresta = i;

			//MostraGrafoBipartido(GB, GO, true);
				
		// Coloca dados das arestas escolhidas na estrutura S
		S = GeraStrut(GB);
//   		printf("Strut gerada\n");
		tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
		printf("Tempo Passo 4.1: %lf\n", tempo2p - tempo1p);
		
		// ==============================================================================
		// Passo 4.2: Calcular o num_zero_diff e computa novas componenetes conexas
		// ==============================================================================
		tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
		CD = (uc *) malloc(GB.n_v*sizeof(uc)); 
		CD_Inic(GB.n_v, CD);
//  		printf("== CD Inicializado ===\n");
		//for(i =0; i < GB.n_v; i++)
			//printf("CD[%d].ch = %d \t CD[%d].tam = %d\n", i, CD[i].ch, i, CD[i].tam);

		num_zerodiff = 0;
		for(i = 0; i < S.n_u; i++)
		{
			if(S.vertices_u[i].grau > 0)
			{
				SolutionEdgeSet[SolutionSize] = S.vertices_u[i].ind_ago;
// 				printf("%d Adicionada aresta: %d - %d com custo %lf\n", S.vertices_u[i].grau, GO.arestas[SolutionEdgeSet[SolutionSize]].v, GO.arestas[SolutionEdgeSet[SolutionSize]].u, GO.arestas[SolutionEdgeSet[SolutionSize]].custo);
				SolutionVal += GO.arestas[S.vertices_u[i].ind_ago].custo;
				SolutionSize++;
				if (S.vertices_u[i].grau == 2)
					num_zerodiff++;
					
				x = CD_chefe(GB.arestas[S.arestas[S.vertices_u[i].inda1].ind_agb].ind_v, CD);
				y = CD_chefe(GB.arestas[S.arestas[S.vertices_u[i].inda1].ind_acgb].ind_v, CD);
				//printf("Analisando %d com chefe %d \t e %d com chefe %d\n", GB.arestas[S.arestas[S.vertices_u[i].inda1].ind_agb].ind_v,  x , GB.arestas[S.arestas[S.vertices_u[i].inda1].ind_acgb].ind_v,  y);
				if (x != y)
				{
					CD_Uniao(x, y, CD);
					//printf("Unindo %d a %d\n", x , y);
				}
			}
		} // end for(i = 0; i < S.n_u; i++)
 
		free(S.vertices_u); 
		free(S.arestas);

		tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
		printf("Tempo Passo 4.2: %lf\n", tempo2p - tempo1p);
		
//  		printf("== CD Atualizado ===\n");
		//for(i =0; i < GB.n_v; i++)
			//printf("CD[%d].ch = %d \t CD[%d].tam = %d\n", i, CD[i].ch, i, CD[i].tam);
		// ==============================================================================
		// Passo 4.3: Compactar o grafo
		// ==============================================================================
		if(SolutionSize < (GO.n-1))
		{
			tempo1p = (double) clock( ) / CLOCKS_PER_SEC;
			H = CompactarGrafo(GB, GO, CD, num_zerodiff);
//  			printf("Grafo compactado\n");
			GB = H;
			tempo2p = (double) clock( ) / CLOCKS_PER_SEC;
			printf("Tempo Passo 4.3: %lf\n", tempo2p - tempo1p);		
		}
		
		free(CD);
		it++;
	} // fim while
	tempo2 = (double) clock( ) / CLOCKS_PER_SEC;
	tempoTotal = tempo2 - tempo1;

	printf("***** Iteração %d ****\n", it);
	printf("\t Número de fragmentos = %d   Número de arestas = %d    SolutionSize = %d    SolutionVal = %lf\n", GB.n_v, GB.m, SolutionSize, SolutionVal);
	printf("\nCusto total da MST: %lf\n", SolutionVal);
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
  			fprintf(Arq, "Aresta %d - %d = %lf\n", GO.arestas[SolutionEdgeSet[i]].v, GO.arestas[SolutionEdgeSet[i]].u, GO.arestas[SolutionEdgeSet[i]].custo);
  	}
  	fclose(Arq);

	
	free(SolutionEdgeSet);
	
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
	
	G.arestas = (aresta_go *) malloc(G.m*sizeof(aresta_go)); 
	for(i = 0; i < G.m; i++){
		fscanf(Arq,"%hu",&G.arestas[i].u);
		fscanf(Arq,"%hu",&G.arestas[i].v);
		if(G.arestas[i].v > G.arestas[i].u)
		{
			aux = G.arestas[i].v;
			G.arestas[i].v = G.arestas[i].u;
			G.arestas[i].u = aux;
		}
		fscanf(Arq,"%f",&G.arestas[i].custo);
	}
	
	fclose(Arq);
   return G;
}


// ==============================================================================
// Função MostraGrafoOriginal:  Mostra as informações de vértices e arestas do
//                              grafo original 
// ==============================================================================

void MostraGrafoOriginal(grafo_original G)
{
	int i;
	
	printf("****** DADOS DO GRAFO ******\n");
	printf("Número de vértices do grafo = %d\n", G.n);
	printf("Número de arestas do grafo = %d\n", G.m);
	for(i = 0; i < G.m; i++)
		printf("Aresta %d \t v = %d \t u = %d \t custo = %lf\n", i, G.arestas[i].v, G.arestas[i].u, G.arestas[i].custo);
	printf("****************************\n");
}



// ==============================================================================
// Função OrdenaArestasGO_v:  Ordena as arestas do grafo original pelo primeiro vértice, 
//                        do menor para o maior, utilizando QuickSort
// ==============================================================================
aresta_go *OrdenaArestasGO_v_u(aresta_go *A, int n, int k, bool v)
{
	int *C;
	int i;
	aresta_go *B;
	
	C = (int *) malloc(k*sizeof(int)); 
	B = (aresta_go *) malloc(n*sizeof(aresta_go)); 
	
	for(i = 0; i < k; i++)
		C[i] = 0;
	if(v)
		for(i = 0; i < n; i++)
			C[A[i].v]++;
	else
		for(i = 0; i < n; i++)
			C[A[i].u]++;
	
	for(i = 1; i < k; i++)
		C[i] = C[i] + C[i-1];
	
	if(v)
		for(i = n-1; i >= 0; i--)
		{
			B[C[A[i].v]-1] = A[i];
			C[A[i].v]--;
		}
	else
		for(i = n-1; i >= 0; i--)
		{
			B[C[A[i].u]-1] = A[i];
			C[A[i].u]--;
		}
		
	free(C);
	free(A);
	
	return B;
}


// ==============================================================================
// Função CriaGrafoBipartido:  Cria um grafo bipartido GB a partir de um grafo 
//                             qualquer adicionando um vértice para cada aresta
//                             do grafo original
// ==============================================================================

grafo_bipartido CriaGrafoBipartido(grafo_original GO)
{
	int i,j;
	grafo_bipartido GB;
	
	GB.n_v = GO.n;
	GB.n_u = GO.m;
	GB.m = GO.m * 2;
	
	GB.vertices_v = (vertice_v *) malloc(GB.n_v*sizeof(vertice_v)); 
	GB.vertices_u = (vertice_u *) malloc(GB.n_u*sizeof(vertice_u)); 
 	GB.arestas = (aresta_gb *) malloc(GB.m*sizeof(aresta_gb)); 
 	
 	for(i = 0; i < GB.n_v; i++)
	{
 		GB.vertices_v[i].id = i;
		GB.vertices_v[i].grau = 0;
	}
 	 	
 	for(i = j = 0; i < GO.m; i++)
 	{
		GB.vertices_v[GO.arestas[i].v].grau++;
		GB.vertices_v[GO.arestas[i].u].grau++;
		
 		GB.vertices_u[i].ind_ago = i;

		GB.arestas[j].ind_v = GO.arestas[i].v;
		GB.arestas[j].ind_u = i;
		GB.arestas[j].ind_ac = j+1;
		j++;
		GB.arestas[j].ind_v = GO.arestas[i].u;;
		GB.arestas[j].ind_u = i;
		GB.arestas[j].ind_ac = j-1;
		j++;
	}
	return GB;
}


// ==============================================================================
// Função MostraGrafoBipartido:  Mostra as informações de vértices e arestas do
//                              grafo bipartido
// ==============================================================================

void MostraGrafoBipartido(grafo_bipartido G, grafo_original GO, bool completo)
{
	int i;
	
	printf("****** DADOS DO GRAFO BIPARTIDO ******\n");
	printf("Número de vértices tipo v do grafo = %d\n", G.n_v);
	printf("Número de vértices tipo u do grafo = %d\n", G.n_u);
	printf("Número de arestas do grafo = %d\n", G.m);
	
	if(completo)
	{
		printf("*** Vertices v ***\n");
		for(i = 0; i < G.n_v; i++)
			printf("Vertice %d \t id = %d \t grau = %d \t menorAresta = %d\n", i, G.vertices_v[i].id, G.vertices_v[i].grau, G.vertices_v[i].menorAresta);
	
		printf("*** Vertices u ***\n");
		for(i = 0; i < G.n_u; i++)
			printf("Vertice %d \t v1 = %d \t v2 = %d \t custo = %lf \t ago = %d\n", i, GO.arestas[G.vertices_u[i].ind_ago].v, GO.arestas[G.vertices_u[i].ind_ago].u, GO.arestas[G.vertices_u[i].ind_ago].custo, G.vertices_u[i].ind_ago);
	}
	
	printf("*** Arestas G.m = %d***\n", G.m);
	for(i = 0; i < G.m; i++)
		//printf("Aresta %d \t ind_v = %d \t ind_u = %d \t ind_ac = %d \t custo = %lf \t ago = %d \n", i, G.arestas[i].ind_v, G.arestas[i].ind_u, G.arestas[i].ind_ac, GO.arestas[G.vertices_u[G.arestas[i].ind_u].ind_ago].custo, G.vertices_u[G.arestas[i].ind_u].ind_ago);
		printf("Aresta %d \t ind_v = %d \t ind_u = %d \t ind_ac = %d \t ago = %d \n", i, G.arestas[i].ind_v, G.arestas[i].ind_u, G.arestas[i].ind_ac, G.vertices_u[G.arestas[i].ind_u].ind_ago);
	printf("****************************\n");
}


// ==============================================================================
// Função OrdenaArestasGB_v_u:  Ordena as arestas do grafo bipartido pelo primeiro ou segundo vértice, 
//                        do menor para o maior, utilizando CountSort
// ==============================================================================
aresta_gb *OrdenaArestasGB_v_u(aresta_gb *A, int n, int k, bool v)
{
	int *C;
	int i;
	aresta_gb *B;
	
	C = (int *) malloc(k*sizeof(int)); 
	B = (aresta_gb *) malloc(n*sizeof(aresta_gb)); 
	
	for(i = 0; i < k; i++)
		C[i] = 0;
		
	if(v)
		for(i = 0; i < n; i++)
			C[A[i].ind_v]++;
	else
		for(i = 0; i < n; i++)
			C[A[i].ind_u]++;
	
	for(i = 1; i < k; i++)
		C[i] = C[i] + C[i-1];

	if(v)
		for(i = n-1; i >= 0; i--)
		{
			B[C[A[i].ind_v]-1] = A[i];
			A[i].ind_ac = C[A[i].ind_v]-1;
			C[A[i].ind_v]--;
		}		
	else
		for(i = n-1; i >= 0; i--)
		{
			B[C[A[i].ind_u]-1] = A[i];
			A[i].ind_ac = C[A[i].ind_u]-1;
			C[A[i].ind_u]--;
		}

	for(i = 0; i < n; i++)
		B[i].ind_ac = A[B[i].ind_ac].ind_ac;

	free(A);
	free(C);
	
	return B;
}

// ==============================================================================
// Função GeraStrut:  Gera uma Strut a partir de um grafo bipartido
// ==============================================================================

strut GeraStrut(grafo_bipartido G)
{
 	int i,j;
 	strut S;
 	
	S.n_v = G.n_v;
 	S.n_u = G.n_u;
	S.m = G.n_v;
 	
	
	S.vertices_u = (vertice_u_strut *) malloc(S.n_u*sizeof(vertice_u_strut)); 
	S.arestas = (aresta_strut *) malloc(S.m*sizeof(aresta_strut)); 
	
 	for(i = 0; i < S.n_u; i++)
 	{
 		S.vertices_u[i].ind_ago = G.vertices_u[i].ind_ago;
		S.vertices_u[i].grau = 0;
		S.vertices_u[i].inda1 = -1;
		S.vertices_u[i].inda2 = -1;
 	}
	
 	for(i = 0; i < S.m; i++){
		j = G.vertices_v[i].menorAresta;
		S.arestas[i].ind_v = G.arestas[j].ind_v;
		S.arestas[i].ind_u = G.arestas[j].ind_u;
		S.arestas[i].ind_acgb =  G.arestas[j].ind_ac;
		S.arestas[i].ind_agb = j;
		
		if(S.vertices_u[S.arestas[i].ind_u].grau == 0)
			S.vertices_u[S.arestas[i].ind_u].inda1 = i;
		else
			S.vertices_u[S.arestas[i].ind_u].inda2 = i;
 		S.vertices_u[S.arestas[i].ind_u].grau += 1;
 	}
 	return S;
}



// ==============================================================================
// Função MostraStrut:  Mostra as informações de vértices e arestas da Strut
// ==============================================================================

void MostraStrut(strut G, bool completa)
{
	int i;
	
	printf("****** DADOS DA STRUT ******\n");
	printf("Número de vértices v da strut = %d\n", G.n_v);
	printf("Número de vértices u da strut = %d\n", G.n_u);
	printf("Número de arestas da strut = %d\n", G.m);
	
	if(completa)
	{
		printf("*** Vertices u ***\n");
		for(i = 0; i < G.n_u; i++)
			if(G.vertices_u[i].inda1 != -1)
			printf("Vertice %d \t ind_ago = %d \t grau = %d \t inda1 = %d \t inda2 = %d\n", i, G.vertices_u[i].ind_ago, G.vertices_u[i].grau, G.vertices_u[i].inda1, G.vertices_u[i].inda2);
	}
	
	printf("*** Arestas ***\n");
	for(i = 0; i < G.m; i++)
		printf("Aresta %d \t ind_v = %d \t ind_u = %d \t ind_agb = %d \t ind_acgb = %d\n", i, G.arestas[i].ind_v, G.arestas[i].ind_u, G.arestas[i].ind_agb, G.arestas[i].ind_acgb);
	printf("****************************\n");
}

// ==============================================================================
// Função CompactarGrafo:  Gera um novo grafo bipartido através da compactação
//                         dos vértices zero-diff 
// ==============================================================================


grafo_bipartido CompactarGrafo(grafo_bipartido G, grafo_original GO, uc *CD, int num_zerodiff)
{
	grafo_bipartido GC;
	int i, j, x, y, aux, v_ant;
	int *custos;
	
	GC.m = G.m;
	
	//printf("=====================================================================\n");
// 	printf("============  1 - GRAFO BIPARTIDO A SER COMPACTADO  =================\n");
	//printf("=====================================================================\n");
	//MostraGrafoBipartido(G, GO, true);
	//printf("=====================================================================\n");
	
	custos = (int *)malloc((G.n_v)*sizeof(int));
	
	v_ant = -1;
	
 	for(i = 0; i < G.m; i++) // Utilizado para marcar arestas que serão removidas
	{
		//printf("Aresta i = %d \t G.arestas[i].ind_v = %d \t v_ant = %d \t => ", i, G.arestas[i].ind_v, v_ant);
		if((i < G.arestas[i].ind_ac) && (G.arestas[i].ind_v != G.n_v))
		{
			//printf("VAMOS TRABALHAR\n");
			if(G.arestas[i].ind_v >  v_ant)
			{
				// Inicializa vetor custos
				//printf("Inicializa vetor de custos para ind_v = %d\n", G.arestas[i].ind_v);
				for (j = 0; j < G.n_v; j++)
					custos[j] = -1;
				v_ant = G.arestas[i].ind_v;
			}

			x = CD_chefe(G.arestas[i].ind_v, CD);
			y = CD_chefe(G.arestas[G.arestas[i].ind_ac].ind_v, CD);
			if(x == y)
			{
				//As arestas i e sua correspondente devem ser marcadas para serem retiradas
				//Correspondem a arestas da Strut
				//printf("Eliminando1 aresta %d e %d\n", i, G.arestas[i].ind_ac);
				G.arestas[i].ind_v = G.n_v;
				G.arestas[i].ind_u = G.n_u;
				G.arestas[G.arestas[i].ind_ac].ind_v = G.n_v;
				G.arestas[G.arestas[i].ind_ac].ind_u = G.n_u;
				GC.m -= 2;
			}
			else
			{
				G.arestas[i].ind_v = x;
				G.arestas[G.arestas[i].ind_ac].ind_v = y;
			
				if(custos[y] == -1) // Primeira aresta que interliga G.arestas[i].ind_v a G.arestas[G.arestas[i].ind_ac].ind_v
				{
					custos[y] = i;
				}
				else
				{
					if(GO.arestas[G.vertices_u[G.arestas[custos[y]].ind_u].ind_ago].custo > GO.arestas[G.vertices_u[G.arestas[i].ind_u].ind_ago].custo)
					{
						//A aresta i interliga x e y com menor custo do que a anteriormente selecionada
						aux = custos[y];
						custos[y] = i;
						
						//As arestas anteriormente selecionadas devem ser marcadas para serem retiradas
						//printf("Eliminando2 aresta %d e %d\n", i, G.arestas[i].ind_ac);
					}
					else
					{
						//As arestas i e sua correspondente devem ser marcadas para serem retiradas
						//Pois já existe outra interligando x e y com menor custo
						//printf("Eliminando3 aresta %d e %d\n", i, G.arestas[i].ind_ac);
						aux = i;
					}
					G.arestas[aux].ind_v = G.n_v;
					G.arestas[aux].ind_u = G.n_u;
					G.arestas[G.arestas[aux].ind_ac].ind_v = G.n_v;
					G.arestas[G.arestas[aux].ind_ac].ind_u = G.n_u;
					GC.m -= 2;
				}
			}
		}
		//else
			//printf("IGNORAR\n");
	}
	free(custos);
// 	printf("custos liberado\n");
	
	//printf("=====================================================================\n");
// 	printf("============= 2 - GRAFO BIPARTIDO SENDO COMPACTADO  =================\n");
	//printf("=====================================================================\n");
	//MostraGrafoBipartido(G, GO, true);
	//printf("=====================================================================\n");

	GC.n_v = num_zerodiff;
	GC.n_u = GC.m/2;
	
	//printf("Arestas redundantes removidas \t G.m = %d \t GC.m = %d\n", G.m, GC.m);
  	GC.vertices_u = (vertice_u *) malloc(GC.n_u*sizeof(vertice_u)); 
// 	printf("GC.vertices_u alocado\n");
	//printf("Alocada estrutura para grafo compactado GC.n_v = %d \t GC.n_u = %d \t GC.m = %d\n", GC.n_v, GC.n_u, GC.m);
	
	//Ordenar todas (G.m) as arestas de G que podem ter G.n_u+1 valores
	G.arestas = OrdenaArestasGB_v_u(G.arestas, G.m, G.n_u+1, false);
	
	
  	aux = -1;
  	for(i = 0; G.arestas[i].ind_u < G.n_u; i++) // Utilizado para criar os vertices_u do grafo compactado
  	{
  		if((aux == -1) || (G.vertices_u[G.arestas[i].ind_u].ind_ago != GC.vertices_u[aux].ind_ago))
  		{
  			aux++;
  			GC.vertices_u[aux].ind_ago = G.vertices_u[G.arestas[i].ind_u].ind_ago;
			//printf("Criado o vértices_u %d com ago = %d.\n", aux, G.vertices_u[G.arestas[i].ind_u].ind_ago);
  		}
  		G.arestas[i].ind_u = aux;
  	}
  	//printf("Criados os vértices u.\n");
  	
	G.arestas = OrdenaArestasGB_v_u(G.arestas, G.m, G.n_v+1, true);
	
	free(G.vertices_u);
// 	printf("G.vertices_u liberado\n");
	
	GC.vertices_v = (vertice_v *) malloc(GC.n_v*sizeof(vertice_v)); 
// 	printf("GC.vertices_v alocado\n");
  	aux = -1;
  	for(i = 0; i < GC.m; i++) // Utilizado para criar os vertices_v e as arestas do grafo compactado
  	{
  		if((aux == -1) || (GC.vertices_v[aux].id != G.vertices_v[G.arestas[i].ind_v].id))
  		{
  			aux++;
  			GC.vertices_v[aux].id = G.vertices_v[G.arestas[i].ind_v].id;
 			GC.vertices_v[aux].grau = 1;
			//printf("Criado o vértices_v %d para G.arestas[i].ind_v = %d com id = %d.\n", aux, G.arestas[i].ind_v,  G.vertices_v[G.arestas[i].ind_v].id);
  		}
  		else
  			GC.vertices_v[aux].grau++;
  		
  		G.arestas[i].ind_v = aux;
  		//printf("Adicionada a aresta %d \t ind_v = %d \t ind_u = %d \t ind_ac = %d\n", i, GC.arestas[i].ind_v, GC.arestas[i].ind_u, GC.arestas[i].ind_ac);
  	}
	GC.arestas = G.arestas;
	
	free(G.vertices_v);
// 	printf("G.vertices_v liberado\n");
  	
	//printf("Acabaram as arestas G.arestas[i].ind_v = %d\n",G.arestas[i].ind_v);
	//printf("Criados os vértices v e as arestas.\n");

	
	//printf("=====================================================================\n");
// 	printf("================  GRAFO BIPARTIDO COMPACTADO  =======================\n");
	//printf("=====================================================================\n");
	//MostraGrafoBipartido(GC, GO, true);
	//printf("=====================================================================\n");
	//printf("Finalizada a compactação do grafo.\n");
  	return GC;
}



// ==============================================================================
// Função Inic_CD:  Inicializa a estrutura de CD
// ==============================================================================
void CD_Inic(int n, uc *CD){ 
	int i;
	
   for (i = 0; i < n; i++) { 
      CD[i].ch = i; 
      CD[i].tam = 1; 
   }
}

int CD_chefe(int v, uc *CD) { 
   int x = v; 
   while(x != CD[x].ch) 
      x = CD[x].ch; 
	return x; 
}

void CD_Uniao(int x, int y, uc *CD) { 
   if (CD[x].tam < CD[y].tam) { 
      CD[x].ch = y; 
      CD[y].tam += CD[x].tam; 
   }
   else { 
		CD[y].ch = x; 
		CD[x].tam += CD[y].tam; 
   }
}
