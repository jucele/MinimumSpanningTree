 /*
	Program: cms_cuda.c
	Description: Implements the Algoritmo for Minimum Spanning Tree
	Programmer: Jucele Vasconcellos
	Date: 1/8/2018

	Compilation:	nvcc -arch sm_30 -o cms_cuda.exe cms_cuda.cu
	Execution:	./cms_cuda.exe in/grafo/grafo1000a cuda.out
	
	Input: This program reads a graph data in the format
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

	where in the first line we have the number of vertices,
	in the second line we have the number of edges
	and the remaining lines are the edges in the format v1 v2 weight
		
	Output: This program produces an output file containing the edges that form the minimum spanning tree
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

// Original Graph
typedef struct { 
	unsigned int v, u; 
} edge_og;

typedef struct { 
	int n, m;
	edge_og *edges;
	float *edge_weigths;
} original_graph;


//Bipartite Graph
typedef struct { 
	unsigned int ind_v;
	int ind_u; 
	int ind_ce; // index of the correspondent edge
} edge_bg;

typedef struct { 
	unsigned int id;
	int smaller_edge; 
} vertex_v;

typedef struct { 
	int ind_eog; // index for the edge of the original graph that originated this vertex
} vertex_u;

typedef struct { 
	int n_v, n_u, m;
	vertex_v *vertices_v;
	vertex_u *vertices_u;
	edge_bg *edges;
} bipartite_graph;


// Strut
typedef struct { 
	int ind_eog; // index for the edge of the original graph that originated this vertex
	unsigned int degree;
	int inde1, inde2;
} vertex_u_strut;

typedef struct { 
	int ind_v, ind_u; 
	int ind_ebg; // index of the edge in the bipartite graph
	int ind_cebg; // index of the corresponding edge in the bipartite graph
} edge_strut;

typedef struct { 
	int n_v, n_u, m;
	vertex_u_strut *vertices_u;
	edge_strut *edges;
} strut;

typedef struct {
	int v1, v2;
} edge_E;

////////////////////////////////////
//Texturas
////////////////////////////////////
texture<float, 1, cudaReadModeElementType> tex_edge_weigthsOG;

// Functions and Procedures

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
	__host__ __device__ bool operator()(const edge_bg &aresta1, const edge_bg &aresta2) 
	{
		return (aresta1.ind_v < aresta2.ind_v);
	}
};

__global__ void CreateBipartiteGraph(edge_og*, int, int, edge_bg*, vertex_v*, vertex_u*);
__global__ void CountOccurrences_u(edge_bg*, unsigned int*, int);
__global__ void SortEdgesBG_u(edge_bg*, int, unsigned int*, edge_bg*);
__global__ void CorrectInd_ce(edge_bg*, int, edge_bg*);
__global__ void UpdateC(edge_E*, int*, int*, int); 
__global__ void MarkEdges1(edge_bg*, int*, int*, int, int, int);
__global__ void MarkEdges2(vertex_u*, edge_bg*, int*, int*, int, int, int, int);
__global__ void FindSmallerEdge(vertex_u*, edge_bg*, int*, int);
__global__ void UpdateSmallerEdge(vertex_v*, int*, int);
__global__ void CreateVertices_u(vertex_u*, vertex_u*, edge_bg*, int); 
__global__ void CreateVertices_v(vertex_v*, int*, int);
__global__ void CreateVertices_u_Strut(vertex_u_strut*, vertex_u*, int n_u);
__global__ void CreateEdges_Strut(edge_strut*, edge_bg*, vertex_u_strut*, vertex_v*, int);

original_graph ReadGraph(char *);

// Main Function
int main (int argc, char** argv){
	original_graph OG;
	bipartite_graph BG, H;
	strut S;
	double totalTime, time1, time2;
	double time1p, time2p;
	int *SolutionEdgeSet;
	int SolutionSize, i, it, aux;
	double SolutionVal;
	int num_zerodiff;
	int *CD;
	FILE *Arq;
		
	int dimBloco, dimGrid;
	edge_og *d_edgesOG;
	edge_bg *d_edgesBG;
	vertex_v *d_vertices_v;
	vertex_u *d_vertices_u, *d_new_vertices_u;
	edge_strut *d_edges_Strut;
	vertex_u_strut *d_vertices_u_Strut;
	
	// d_C e d_B used to sort during compaction
	unsigned int *d_C; //Declares d_C to store CountOccurrences
	edge_bg *d_B; //Declares d_B to store ordered edges
	int *d_smaller_edge;
	
	float *d_edge_weigthsOG;
	
	
	// Step 1: Parameter check
	// Step 2: Reading the data of the graph
	// Step 3: Creation of the bipartite graph corresponding to the edges received
	// Step 4: Find solution
		// Step 4.1: Choose edges that will compose the strut
		// Step 4.2: Calculate num_zero_diff
		// Step 4.3: Compacting the graph
	
	
	// ==============================================================================
	// Step 1: Parameter check
	// ==============================================================================
	
	//Checking the parameters
	if(argc < 3 ){
	   printf( "\nIncorrect Parameters\n Usage: ./cms_seq.exe <InputFileName> <OutputFileName> <Y/N> where:\n" );
	   printf( "\t <InputFileName> (required) - Name of the file containing the graph information (number of vertices, number of edges and costs of edges).\n" );
		printf( "\t <OutputFileName> (required) - Output file name.\n" );
		printf( "\t <Y or N> - Show or not the edges of the MST.\n" );
		return 0;
	} 	
	dimBloco = 64;
	
	// ==============================================================================
	// Step 2: Reading the data of the graph
	// ==============================================================================
	time1p = (double) clock( ) / CLOCKS_PER_SEC;
	OG = ReadGraph(argv[1]);

// 	printf("Input graph read\n");
	SolutionEdgeSet = (int *) malloc((OG.n-1)*sizeof(int)); 
	SolutionSize = 0;
	SolutionVal = 0;
	time2p = (double) clock( ) / CLOCKS_PER_SEC;
  	printf("Time Step 2: %lf\n", time2p - time1p);
	
	// ==============================================================================
	// Step 3: Creation of the bipartite graph corresponding to the edges received
	// ==============================================================================
	//Starting timer count
	time1 = (double) clock( ) / CLOCKS_PER_SEC;
	time1p = (double) clock( ) / CLOCKS_PER_SEC;
	
	BG.n_v = OG.n;
	BG.n_u = OG.m;
	BG.m = OG.m * 2;
	
	//Allocate memory in the device to the edges of the original graph
	CHECK_ERROR(cudaMalloc((void **) &d_edgesOG, OG.m * sizeof(edge_og)));
	//Copy the edges of the original graph from host to the device
	CHECK_ERROR(cudaMemcpy(d_edgesOG, OG.edges, OG.m * sizeof(edge_og), cudaMemcpyHostToDevice));
	//Allocate memory in the device to the costs of the edges of the original graph
	CHECK_ERROR(cudaMalloc((void **) &d_edge_weigthsOG, OG.m * sizeof(float)));
	//Copy costs from the edges of the original graph from host to the device
	CHECK_ERROR(cudaMemcpy(d_edge_weigthsOG, OG.edge_weigths, OG.m * sizeof(float), cudaMemcpyHostToDevice));
	//Link the texture to the edges of the original graph
	CHECK_ERROR(cudaBindTexture(0, tex_edge_weigthsOG, d_edge_weigthsOG, OG.m * sizeof(float)));

	//Allocate memory in the device to the edges of the bipartite graph
	CHECK_ERROR(cudaMalloc((void **) &d_edgesBG, BG.m * sizeof(edge_bg)));
	CHECK_ERROR(cudaMalloc((void **) &d_vertices_v, OG.n * sizeof(vertex_v)));
	CHECK_ERROR(cudaMalloc((void **) &d_vertices_u, OG.m * sizeof(vertex_u)));

	//Allocate memory to copy edges to host
	BG.edges = (edge_bg *) malloc(BG.m*sizeof(edge_bg)); 
	BG.vertices_v = (vertex_v *) malloc(BG.n_v*sizeof(vertex_v)); 
	BG.vertices_u = (vertex_u *) malloc(BG.n_u*sizeof(vertex_u)); 
	
	//Set the dimension for the grid
	dimGrid = ((OG.m-1)/dimBloco)+1;
	//Create bipartite graph on GPU
	CreateBipartiteGraph<<<dimGrid, dimBloco>>>(d_edgesOG, OG.m, OG.n, d_edgesBG, d_vertices_v, d_vertices_u);
	
	//Allocates memory for d_B that will be used in later sorting
 	CHECK_ERROR(cudaMalloc((void **) &d_B, BG.m * sizeof(edge_bg)));
	
	//Find smaller edge for each vertex v
	//Allocate memory for d_smaller_edge
	CHECK_ERROR(cudaMalloc((void **) &d_smaller_edge, BG.n_v * sizeof(int)));
	//Initializes d_smaller_edge with -1
 	CHECK_ERROR(cudaMemset(d_smaller_edge, -1, BG.n_v * sizeof(int)));
	//Call kernel to find smaller_edge of each v
 	dimGrid = ((BG.m-1)/dimBloco)+1;
	FindSmallerEdge<<<dimGrid, dimBloco>>>(d_vertices_u, d_edgesBG,  d_smaller_edge, BG.m);
	
	CHECK_ERROR(cudaDeviceSynchronize());
	
	dimGrid = ((BG.n_v-1)/dimBloco)+1;
	UpdateSmallerEdge<<<dimGrid, dimBloco>>>(d_vertices_v, d_smaller_edge, BG.n_v);
	
	//Copy edges of bipartite graph to host
	CHECK_ERROR(cudaMemcpy(BG.edges, d_edgesBG, BG.m * sizeof(edge_bg), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(BG.vertices_u, d_vertices_u, BG.n_u * sizeof(vertex_u), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(BG.vertices_v, d_vertices_v, BG.n_v * sizeof(vertex_v), cudaMemcpyDeviceToHost));
	//Release memory of variables
	CHECK_ERROR(cudaFree(d_edgesOG));	


	time2p = (double) clock( ) / CLOCKS_PER_SEC;
  	printf("Time Step 3: %lf\n", time2p - time1p);

	
	// ==============================================================================
	// Step 4: Find solution
	// ==============================================================================
	
	it = 0;
	num_zerodiff = 0;
	while (SolutionSize < (OG.n-1))
	{
		printf("***** Iteration %d ****\n", it);
 		printf("\t Number of components = %d   Number of edges = %d    SolutionSize = %d    SolutionVal = %lf\n", BG.n_v, BG.m, SolutionSize, SolutionVal);

		// ==============================================================================
		// Step 4.1: Choose edges that will compose the strut
		// ==============================================================================
		time1p = (double) clock( ) / CLOCKS_PER_SEC;
		
		// Places data from chosen edges in structure S
		S.n_v = BG.n_v;
		S.n_u = BG.n_u;
		S.m = BG.n_v;
	
		S.vertices_u = (vertex_u_strut *) malloc(S.n_u*sizeof(vertex_u_strut)); 
		S.edges = (edge_strut *) malloc(S.m*sizeof(edge_strut)); 
		
		CHECK_ERROR(cudaMalloc((void **) &d_vertices_u_Strut, S.n_u * sizeof(vertex_u_strut)));
		CHECK_ERROR(cudaMalloc((void **) &d_edges_Strut, S.m * sizeof(edge_strut)));
		//Call kernel to create Struts vertex_u
		dimGrid = ((S.n_u-1)/dimBloco)+1;
		CreateVertices_u_Strut<<<dimGrid, dimBloco>>>(d_vertices_u_Strut, d_vertices_u, S.n_u);
		//Call kernel to create Strut edges
		dimGrid = ((S.m-1)/dimBloco)+1;
		CreateEdges_Strut<<<dimGrid, dimBloco>>>(d_edges_Strut, d_edgesBG, d_vertices_u_Strut, d_vertices_v, S.m);
		
		CHECK_ERROR(cudaMemcpy(S.vertices_u, d_vertices_u_Strut, S.n_u * sizeof(vertex_u_strut), cudaMemcpyDeviceToHost));
		CHECK_ERROR(cudaFree(d_vertices_u_Strut));
		CHECK_ERROR(cudaMemcpy(S.edges, d_edges_Strut, S.m * sizeof(edge_strut), cudaMemcpyDeviceToHost));
		CHECK_ERROR(cudaFree(d_edges_Strut));
 	
		time2p = (double) clock( ) / CLOCKS_PER_SEC;
  		printf("Time Step 4.1: %lf\n", time2p - time1p);
		
		// ==============================================================================
		// Step 4.2: Calculate num_zero_diff 
		// ==============================================================================
		time1p = (double) clock( ) / CLOCKS_PER_SEC;

		num_zerodiff = 0;
		for(i = 0; i < S.n_u; i++)
		{
			if(S.vertices_u[i].degree > 0)
			{
				SolutionEdgeSet[SolutionSize] = S.vertices_u[i].ind_eog;
				SolutionVal += OG.edge_weigths[S.vertices_u[i].ind_eog];
				SolutionSize++;
				if (S.vertices_u[i].degree == 2)
					num_zerodiff++;
			}
		} // end for(i = 0; i < S.n_u; i++)
 
		time2p = (double) clock( ) / CLOCKS_PER_SEC;
  		printf("Time Step 4.2: %lf     num_zerodiff = %d     SolutionSize = %d\n", time2p - time1p, num_zerodiff, SolutionSize);
		
		// ==============================================================================
		// Step 4.3: Compacting the graph
		// ==============================================================================
		if(SolutionSize < (OG.n-1))
		{
			time1p = (double) clock( ) / CLOCKS_PER_SEC;
			
			// ==============================================================================
			// Step 4.3.1: Compute the connected components
			// ==============================================================================
			time1p = (double) clock( ) / CLOCKS_PER_SEC;
			CD = (int *) malloc(BG.n_v*sizeof(int)); 
			edge_E *h_edgesE, *d_edgesE;
			h_edgesE = (edge_E *) malloc(BG.n_v*sizeof(edge_E)); 
			CHECK_ERROR(cudaMalloc((void **) &d_edgesE, BG.n_v * sizeof(edge_E)));
			
			//Initializes CD and h_edgesE
			for(i = 0; i < BG.n_v; i++)
			{
				h_edgesE[i].v1 = S.edges[i].ind_v;
				h_edgesE[i].v2 = BG.edges[S.edges[i].ind_cebg].ind_v;
				CD[i] = h_edgesE[i].v1;
			}
			//Copy edges_E from host to device
			CHECK_ERROR(cudaMemcpy(d_edgesE, h_edgesE, BG.n_v * sizeof(edge_E), cudaMemcpyHostToDevice));
			//Create d_CD
			int *d_CD;
			CHECK_ERROR(cudaMalloc((void **) &d_CD, BG.n_v * sizeof(int)));
			CHECK_ERROR(cudaMemcpy(d_CD, CD, BG.n_v * sizeof(int), cudaMemcpyHostToDevice));
			int h_end, *d_end;
			CHECK_ERROR(cudaMalloc((void**)&d_end, sizeof(int)));
			do
			{
				h_end = 0;
				CHECK_ERROR(cudaMemcpy(d_end, &h_end, sizeof(int), cudaMemcpyHostToDevice));
				dimGrid = ((BG.n_v-1)/dimBloco)+1;
				UpdateC<<<dimGrid, dimBloco>>>(d_edgesE, d_CD, d_end, BG.n_v);
				CHECK_ERROR(cudaMemcpy(&h_end, d_end, sizeof(int), cudaMemcpyDeviceToHost));		
			}while (h_end == 1);
			
			CHECK_ERROR(cudaMemcpy(CD, d_CD, BG.n_v * sizeof(int), cudaMemcpyDeviceToHost));
			aux = 1;
			for(i = 1; i < BG.n_v; i++)
			{
				if(CD[i] == i)
				{
					CD[i] = aux;
					aux++;
				}
				else
					CD[i] = CD[CD[i]];
			}
			CHECK_ERROR(cudaMemcpy(d_CD, CD, BG.n_v * sizeof(int), cudaMemcpyHostToDevice));

			time2p = (double) clock( ) / CLOCKS_PER_SEC;
  			printf("Time Step 4.3.1: %lf\n", time2p - time1p);

			//Releasing variables
			free(S.vertices_u); 
			free(S.edges);
			free(h_edgesE);
			CHECK_ERROR(cudaFree(d_edgesE));
			
			// ==============================================================================
			// Step 4.3.2: Mark edges
			// ==============================================================================
			time1p = (double) clock( ) / CLOCKS_PER_SEC;
			
			H.m = BG.m;
			int *d_mH;
 			
			//Allocate memory for d_mH
			CHECK_ERROR(cudaMalloc((void **) &d_mH, sizeof(int)));
			//Copy H.m to d_mH
			CHECK_ERROR(cudaMemcpy(d_mH, &H.m, sizeof(int), cudaMemcpyHostToDevice));
			//Allocate costs for all n_v
			//CHECK_ERROR(cudaMalloc((void **) &d_smaller_edge, (BG.n_v * BG.n_v) * sizeof(int)));
			CHECK_ERROR(cudaMalloc((void **) &d_smaller_edge, (num_zerodiff * num_zerodiff) * sizeof(int)));
			//Initialize d_smaller_edge with -1
			CHECK_ERROR(cudaMemset(d_smaller_edge, -1, (num_zerodiff * num_zerodiff) * sizeof(int)));
				
			//Mark the edges for removal
			dimGrid = ((BG.m-1)/dimBloco)+1;
			MarkEdges1<<<dimGrid, dimBloco>>>(d_edgesBG, d_CD, d_mH, BG.m, BG.n_v, BG.n_u);
			
			MarkEdges2<<<dimGrid, dimBloco>>>(d_vertices_u, d_edgesBG, d_mH, d_smaller_edge, BG.m, BG.n_v, BG.n_u, num_zerodiff);
			
			//CHECK_ERROR(cudaDeviceSynchronize());
			
			//Copy d_mH to H.m
			CHECK_ERROR(cudaMemcpy(&H.m, d_mH, sizeof(int), cudaMemcpyDeviceToHost));
			
			CHECK_ERROR(cudaFree(d_mH));
			CHECK_ERROR(cudaFree(d_CD));
			CHECK_ERROR(cudaFree(d_smaller_edge));
			
			time2p = (double) clock( ) / CLOCKS_PER_SEC;
  			printf("Time Step 4.3.2: %lf\n", time2p - time1p);		
			
			// ==============================================================================
			// Step 4.3.3: Generate new bipartite graph
			// ==============================================================================
			time1p = (double) clock( ) / CLOCKS_PER_SEC;
			
			//Sort all the (G.m) edges of BG that can have (G.n_u + 1) values
			//Aloca memória para d_C
			CHECK_ERROR(cudaMalloc((void **) &d_C, (BG.n_u+1) * sizeof(unsigned int)));
			//Initialize d_C with 0
			CHECK_ERROR(cudaMemset(d_C, 0, (BG.n_u+1) * sizeof(unsigned int)));
			//Count the number of occurrences of each v
			dimGrid = ((BG.m-1)/dimBloco)+1;
			CountOccurrences_u<<<dimGrid, dimBloco>>>(d_edgesBG, d_C, BG.m);
			//Update d_C with complement
			thrust::device_ptr<unsigned int>d_C_ptr = thrust::device_pointer_cast(d_C);
			thrust::inclusive_scan(d_C_ptr, d_C_ptr + (BG.n_u+1), d_C_ptr);	
			//Sort the edges
			dimGrid = ((BG.m-1)/dimBloco)+1;
			SortEdgesBG_u<<<dimGrid, dimBloco>>>(d_edgesBG, BG.m, d_C, d_B);
			dimGrid = ((H.m-1)/dimBloco)+1;
			CorrectInd_ce<<<dimGrid, dimBloco>>>(d_edgesBG, H.m, d_B);
			//Copy the edges of bipartite graph to the host
			CHECK_ERROR(cudaMemcpy(d_edgesBG, d_B, BG.m * sizeof(edge_bg), cudaMemcpyDeviceToDevice));
			CHECK_ERROR(cudaMemcpy(BG.edges, d_edgesBG, H.m * sizeof(edge_bg), cudaMemcpyDeviceToHost));
			//Releases variables
			CHECK_ERROR(cudaFree(d_C));

			
			//Create the vector vertices_u
			H.n_u = H.m/2;
			free(BG.vertices_u);
			//Allocate memory for BG.vertices_u
			BG.vertices_u = (vertex_u *) malloc(H.n_u*sizeof(vertex_u)); 
			//Allocate memory for d_new_vertices_u
			CHECK_ERROR(cudaMalloc((void **) &d_new_vertices_u, H.n_u * sizeof(vertex_u)));
			//Call kernel to create vector vertices_u
			dimGrid = ((H.m-1)/dimBloco)+1;
			CreateVertices_u<<<dimGrid, dimBloco>>>(d_vertices_u, d_new_vertices_u, d_edgesBG, H.m);
			//Copy vertices_u of bipartite graph to host
			CHECK_ERROR(cudaMemcpy(BG.vertices_u, d_new_vertices_u, H.n_u * sizeof(vertex_u), cudaMemcpyDeviceToHost));
			CHECK_ERROR(cudaMemcpy(d_vertices_u, d_new_vertices_u, H.n_u * sizeof(vertex_u), cudaMemcpyDeviceToDevice));

			
			//Create vector BG.vertices_v
			H.n_v = num_zerodiff;
			free(BG.vertices_v);
			BG.vertices_v = (vertex_v *) malloc(H.n_v*sizeof(vertex_v)); 
			//Allocate memory for variable d_smaller_edge
			CHECK_ERROR(cudaMalloc((void **) &d_smaller_edge, BG.n_v * sizeof(int)));
			//Initialize d_smaller_edge with -1
			CHECK_ERROR(cudaMemset(d_smaller_edge, -1, BG.n_v * sizeof(int)));
			//Set the dimension for the grid
			dimGrid = ((H.m-1)/dimBloco)+1;
			//Call kernel FindSmallerEdge
			FindSmallerEdge<<<dimGrid, dimBloco>>>(d_new_vertices_u, d_edgesBG, d_smaller_edge, H.m);
			//Set the dimension for the grid
			dimGrid = ((H.n_v-1)/dimBloco)+1;
			//Call kernel CreateVertices_v
			CreateVertices_v<<<dimGrid, dimBloco>>>(d_vertices_v, d_smaller_edge, H.n_v);
			//Copy vertices_u of bipartite graph from device to host
			CHECK_ERROR(cudaMemcpy(BG.vertices_v, d_vertices_v, H.n_v * sizeof(vertex_v), cudaMemcpyDeviceToHost));
			//Releases variables
			CHECK_ERROR(cudaFree(d_smaller_edge));

			
			BG.m = H.m;
			BG.n_v = H.n_v;
			BG.n_u = H.n_u;
			
			time2p = (double) clock( ) / CLOCKS_PER_SEC;
  			printf("Time Step 4.3.3: %lf\n", time2p - time1p);		
		}
		it++;
	} // end while
	
	time2 = (double) clock( ) / CLOCKS_PER_SEC;
	totalTime = time2 - time1;

	printf("SolutionSize = %d      MST total cost: %lf\n", SolutionSize, SolutionVal);
	printf("Total Time: %lf\n", totalTime); 

	Arq = fopen(argv[2], "a");
 	fprintf(Arq, "\n*** Input file: %s\n", argv[1]); 
	fprintf(Arq, "*** MST total cost: %lf\n", SolutionVal);
	fprintf(Arq, "Total Time: %lf\n", totalTime); 
	fprintf(Arq, "Number of iterations: %d\n", it);
	fprintf(Arq, "SolutionSize: %d\n", SolutionSize);

  	if((argc == 4) && (argv[3][0] == 'Y' || argv[3][0] == 'y'))
	{
  		fprintf(Arq, "*** MST formed by %d edges\n", SolutionSize);
  		for(i = 0; i < SolutionSize; i++)
  			fprintf(Arq, "Edge %d - %d = %lf\n", OG.edges[SolutionEdgeSet[i]].v, OG.edges[SolutionEdgeSet[i]].u, OG.edge_weigths[SolutionEdgeSet[i]]);
  	}
  	fclose(Arq);
	
	free(OG.edges);
	free(OG.edge_weigths);
	free(BG.edges);
	free(SolutionEdgeSet);
	
	CHECK_ERROR(cudaFree(d_vertices_v));
	CHECK_ERROR(cudaFree(d_vertices_u));		
	CHECK_ERROR(cudaFree(d_new_vertices_u));
	CHECK_ERROR(cudaFree(d_edge_weigthsOG));	
	CHECK_ERROR(cudaFree(d_edgesBG));	
	CHECK_ERROR(cudaFree(d_B));	
	
	return 0;

}


// ==============================================================================
// Function ReadGraph:  Reads the graph information from a file and stores it 
//                      in a structure
// ==============================================================================
original_graph ReadGraph(char *Arquivo){
	int i, aux;
	original_graph G;
   FILE *Arq;
    
   Arq = fopen(Arquivo, "r");

   i = 0;
	fscanf(Arq,"%d",&i);
	G.n = i;
	
	fscanf(Arq,"%d",&i);
	G.m = i;
	
	G.edges = (edge_og *) malloc(G.m*sizeof(edge_og));
	G.edge_weigths = (float *) malloc(G.m*sizeof(float));
	
	for(i = 0; i < G.m; i++){
		fscanf(Arq,"%d",&G.edges[i].u);
		fscanf(Arq,"%d",&G.edges[i].v);
		if(G.edges[i].v > G.edges[i].u)
		{
			aux = G.edges[i].v;
			G.edges[i].v = G.edges[i].u;
			G.edges[i].u = aux;
		}
		fscanf(Arq,"%f",&G.edge_weigths[i]);
	}
	
	fclose(Arq);
   return G;
}


// ==============================================================================
// Function CreateBipartiteGraph: Creates the bipartite graph corresponding 
//                                to the input graph
// ==============================================================================
__global__ void CreateBipartiteGraph(edge_og* edgesOG, int mOG, int nOG, edge_bg* edgesBG, vertex_v *vertices_v, vertex_u *vertices_u) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(id < mOG)
	{
		if(id < nOG)
		{
			vertices_v[id].id = id;
		}
		vertices_u[id].ind_eog = id;

		edgesBG[2*id].ind_v = edgesOG[id].v;
		edgesBG[2*id].ind_u = id;
		edgesBG[2*id].ind_ce = 2*id+1;
		
		edgesBG[2*id+1].ind_v = edgesOG[id].u;
		edgesBG[2*id+1].ind_u = id;
		edgesBG[2*id+1].ind_ce = 2*id;
	}
}

// ==============================================================================
// Function CountOccurrences_u:  Counts the number of edges for each vertex u
// ==============================================================================
__global__ void CountOccurrences_u(edge_bg* edgesBG, unsigned int* C, int n)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if(id < n)
	{
		atomicInc(&C[edgesBG[id].ind_u], UINT_MAX);
	}	
}


// ==============================================================================
// Function SortEdgesBG_u:  Sorts the edges of the bipartite graph by vertex u, 
//                          from lowest to highest, using CountSort
// ==============================================================================
__global__ void SortEdgesBG_u(edge_bg* edgesBG, int n, unsigned int *C, edge_bg* B) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n )
	{
		unsigned int i = atomicSub(&C[edgesBG[id].ind_u], 1);
		i--;
		B[i] = edgesBG[id];
		edgesBG[id].ind_ce = i;
	}
}


// ==============================================================================
// Function CorrectInd_ce:  Corrects the indices of the corresponding edges
// ==============================================================================
__global__ void CorrectInd_ce(edge_bg* edgesBG, int n, edge_bg* B) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n )
	{
		B[id].ind_ce = edgesBG[B[id].ind_ce].ind_ce;
	}
}


// ==============================================================================
// Function MarkEdges1:  Update ind_v of the edges
//                       If edge[i].ind_v == edge[edge[i].ind_ce].ind_v then 
//                          mark the edges i and edge[i].ind_ce to be removed
// ==============================================================================
__global__ void MarkEdges1(edge_bg *edgesBG, int* C, int *mH, int mBG, int n_v, int n_u)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int x, y;
	
	if( id < mBG )
	{
		if(id < edgesBG[id].ind_ce)
		{
			x = C[edgesBG[id].ind_v];
			y = C[edgesBG[edgesBG[id].ind_ce].ind_v];
			
			if(x == y)
			{
				edgesBG[id].ind_v = n_v;
				edgesBG[edgesBG[id].ind_ce].ind_v = n_v;
				edgesBG[id].ind_u = n_u;
				edgesBG[edgesBG[id].ind_ce].ind_u = n_u;
				atomicSub(&mH[0], 2);
			}
			else
			{
				edgesBG[id].ind_v = x;
				edgesBG[edgesBG[id].ind_ce].ind_v = y;
			}
		} //end if(id < edgesBG[id].ind_ce)
	} //end if( id < mBG )
}

// ==============================================================================
// Function MarkEdges2:  Mark the edges of the graph to be removed
// ==============================================================================
__global__ void MarkEdges2(vertex_u* vertices_u, edge_bg* edgesBG, int* mH, int* custos, int mBG, int n_v, int n_u, int num_zerodiff)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int x, y, aux;
	int old_id;
	float old_custo, meu_custo;
	
	if( id < mBG )
	{
		if(id < edgesBG[id].ind_ce)
		{
			if(edgesBG[id].ind_v < n_v)
			{
				x = edgesBG[id].ind_v;
				y = edgesBG[edgesBG[id].ind_ce].ind_v;
				
				if(x > y)
				{
					aux = x;
					x = y;
					y = aux;
				}

				aux = -1;
				old_id = atomicCAS(&custos[x*num_zerodiff+y], aux, id);
 				//old_custo = edgesOG[vertices_u[edgesBG[old_id].ind_u].ind_eog].custo;
				old_custo = tex1Dfetch(tex_edge_weigthsOG, vertices_u[edgesBG[old_id].ind_u].ind_eog);
				
 				if(old_id != aux)
 				{
					//meu_custo = edgesOG[vertices_u[edgesBG[id].ind_u].ind_eog].custo;
					meu_custo = tex1Dfetch(tex_edge_weigthsOG, vertices_u[edgesBG[id].ind_u].ind_eog);
					
					while((old_custo > meu_custo) && (old_id != aux))
					{
						aux = atomicCAS(&custos[x*num_zerodiff+y], old_id, id);
						if(old_id != aux)
						{
							old_id = aux;
							aux = -1;
							//old_custo = edgesOG[vertices_u[edgesBG[old_id].ind_u].ind_eog].custo;
							old_custo = tex1Dfetch(tex_edge_weigthsOG, vertices_u[edgesBG[old_id].ind_u].ind_eog);
						}
					}
			
					if(old_custo <= meu_custo)
					{
						edgesBG[id].ind_v = n_v;
						edgesBG[edgesBG[id].ind_ce].ind_v = n_v;
						edgesBG[id].ind_u = n_u;
						edgesBG[edgesBG[id].ind_ce].ind_u = n_u;
						atomicSub(&mH[0], 2);
					}
				} // end if(id_old_custo1 != -1)
			} //end if(edgesBG[id].ind_v < n_v)
		} //end if(id < edgesBG[id].ind_ce)
	} //end if( id < mBG )
}



// ==============================================================================
// Function UpdateC: Updates vector C for definition of connected components
// ==============================================================================
__global__ void UpdateC(edge_E *edgesE, int* C, int *m, int n) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int c1, c2, v1, v2;
	
	if(id < n)
	{
		v1 = edgesE[id].v1;
		v2 = edgesE[id].v2;
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
// Function FindSmallerEdge:  For each vertex v, find the id of the lower weight edge
// ==============================================================================
__global__ void FindSmallerEdge(vertex_u* vertices_u, edge_bg* edgesBG, int* smaller_edge, int mBG)
{
	int meu_id = threadIdx.x + blockDim.x * blockIdx.x;
	int x, aux;
	int old_id;
	float old_custo, meu_custo;
	
	if( meu_id < mBG )
	{
		x = edgesBG[meu_id].ind_v;

		aux = -1;
		old_id = atomicCAS(&smaller_edge[x], aux, meu_id);
				
 		if(old_id != aux)
 		{
			//meu_custo = edgesOG[vertices_u[edgesBG[meu_id].ind_u].ind_eog].custo;
			meu_custo = tex1Dfetch(tex_edge_weigthsOG, vertices_u[edgesBG[meu_id].ind_u].ind_eog);
			//old_custo = edgesOG[vertices_u[edgesBG[old_id].ind_u].ind_eog].custo;
			old_custo = tex1Dfetch(tex_edge_weigthsOG, vertices_u[edgesBG[old_id].ind_u].ind_eog);
				
			while(((old_custo > meu_custo) || ((old_custo == meu_custo) && (edgesBG[old_id].ind_u > edgesBG[meu_id].ind_u))) && (old_id != aux))
			{
				aux = atomicCAS(&smaller_edge[x], old_id, meu_id);
				if(old_id != aux)
				{
					old_id = aux;
					aux = -1;
					//old_custo = edgesOG[vertices_u[edgesBG[old_id].ind_u].ind_eog].custo;
					old_custo = tex1Dfetch(tex_edge_weigthsOG, vertices_u[edgesBG[old_id].ind_u].ind_eog);
				}
			}
		} // end if(id_old_custo1 != -1)
	} //end if( meu_id < mBG )
}


// ==============================================================================
// Function UpdateSmallerEdge:  Update the vector d_vertices_v with the smaller_edge
// ==============================================================================
__global__ void UpdateSmallerEdge(vertex_v* vertices_v, int* smaller_edge, int n_v)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n_v )
	{
		vertices_v[id].smaller_edge = smaller_edge[id];
	}	
}



// ==============================================================================
// Function CreateVertices_u:  Creates new vector vertices_u after compaction of edges
// ==============================================================================
__global__ void CreateVertices_u(vertex_u* vertices_u, vertex_u* new_vertices_u, edge_bg* edgesBG, int H_m) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < H_m )
	{
		if((id % 2) == 0) 
		{
			new_vertices_u[id/2].ind_eog = vertices_u[edgesBG[id].ind_u].ind_eog;
			edgesBG[id].ind_u = id/2;
			edgesBG[id+1].ind_u = id/2;
		}
	}
}

// ==============================================================================
// Function CreateVertices_v:  Creates new vector vertices_v after compaction of edges
// ==============================================================================
__global__ void CreateVertices_v(vertex_v* vertices_v, int* smaller_edge, int n_v)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n_v )
	{
		vertices_v[id].id = id;
		vertices_v[id].smaller_edge = smaller_edge[id];
	} //end if( id < n_v )
}


// ==============================================================================
// Function CreateVertices_u_Strut:  Creates strut vertices_u vector
// ==============================================================================
__global__ void CreateVertices_u_Strut(vertex_u_strut* vertices_u_Strut, vertex_u* vertices_u, int n_u)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if( id < n_u )
	{
		vertices_u_Strut[id].ind_eog = vertices_u[id].ind_eog;
		vertices_u_Strut[id].degree = 0;
		vertices_u_Strut[id].inde1 = -1;
		vertices_u_Strut[id].inde2 = -1;
	} //end if( id < n_u )
}


// ==============================================================================
// Function CreateEdges_Strut:  Creates strut edges vector
// ==============================================================================
__global__ void CreateEdges_Strut(edge_strut* edges_Strut, edge_bg* edgesBG, vertex_u_strut* vertices_u_Strut, vertex_v* vertices_v, int m)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int j, old_degree;
	
	if( id < m )
	{
		j = vertices_v[id].smaller_edge;
		edges_Strut[id].ind_v = edgesBG[j].ind_v;
		edges_Strut[id].ind_u = edgesBG[j].ind_u;
		edges_Strut[id].ind_cebg = edgesBG[j].ind_ce;
		edges_Strut[id].ind_ebg = j;
		
		old_degree = atomicInc(&vertices_u_Strut[edges_Strut[id].ind_u].degree, UINT_MAX);
		
		if(old_degree == 0)
			vertices_u_Strut[edges_Strut[id].ind_u].inde1 = id;
		else
			vertices_u_Strut[edges_Strut[id].ind_u].inde2 = id;
	} //end if( id < n_u )
}
