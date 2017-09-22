#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helper_functions.h"

//indexes into the array of dimensions sent to all ranks
#define M_INDEX 0
#define N_INDEX 1
#define P_INDEX 2

int main(int argc, char** argv) {
    double **A, **B, **C, **A_inner, **C_inner,
            **B_transpose, **B_inner_transpose;
    int m, n, p, print, i, world_rank, world_size, use_openmp;
    struct timespec start, stop;
    int transpose_time_mills;
    int rank0_end_row_index; //rank 0 should work on row 0 to this index
    
    if (argc != 6) {
    	printf("Usage: matmult M N P  print num_openMP_threads\n");
    	printf("where matrices to multiply to  are A = MxN and B = NxP,\n");
    	printf("random uniform double values between 0 and 1.0.\n");
    	printf("Computes C = A*B\n");
    	printf("1 = print matrix, 0 = no,\n");
        printf("1 =  openMP, or 0 for no \n");
    	return 1;
	}
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	p = atoi(argv[3]);
    print = atoi(argv[4]);
    use_openmp = atoi(argv[5]);
    
    int dimensions[3];

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // row_indices[rank][0] = start row index for rank
    // row_indices[rank][1] = end row index for rank
	int row_indices[world_size][2]; 

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (world_rank == 0) {
        srand(time(NULL));
        B = allocateMatrix(n, p);
        createRandValues(B, n, p);
        
        // start timing before transposing B
        clock_gettime(CLOCK_REALTIME, &start);        
        B_transpose = transposeB(B, p, n);
        free(B[0]);
        free(B);
        clock_gettime(CLOCK_REALTIME, &stop);
        transpose_time_mills = ms_diff(start, stop);
        // done timing transpose, extra copy of B free from memory
        
        A = allocateMatrix(m, n);
        C = allocateMatrix(m, p);        
        createRandValues(A, m, n);
        //clear C, dynamically allocated memory may have garbage
        memset(C[0], 0, sizeof(double) * m * p);
        
        if (print == 1) {            
            printf("\nMatrices A, B_transpose, C,  before multiplication:\n");
            print_matrices_transpose(m, n, p, A , B_transpose, C);
        }
        
        dimensions[N_INDEX] = n;
        dimensions[P_INDEX] = p;
        
        // start timing for the rest of the MPI execution
        clock_gettime(CLOCK_REALTIME, &start);
                
        //create the row indices for the different processors to work on
        int rows_per_rank = m / world_size;
        int remainder_rows =  m - world_size * rows_per_rank;

        int last_end_index = -1;
        for (i = 0; i < world_size; i++) {
            row_indices[i][0] = last_end_index + 1;
            last_end_index = row_indices[i][0] + rows_per_rank - 1;
            /*now take 1 row of the remainder rows and toss it on if
            any rows remain */
            if (remainder_rows > 0) {
                last_end_index++;
                remainder_rows--;
            }
            row_indices[i][1] = last_end_index;
        }
        
        //rank 0 works on the first partition, not sent 
        rank0_end_row_index = row_indices[0][1];
        
        //these loops skip rank 0, no need to send to self
        for (int i = 1; i < world_size; i++) {
            dimensions[M_INDEX] = row_indices[i][1] - row_indices[i][0] + 1;
            MPI_Send(dimensions, 3, MPI_INT, i, 0,  MPI_COMM_WORLD);
            MPI_Send(A[row_indices[i][0]], n * dimensions[M_INDEX], MPI_DOUBLE, i, 0,  MPI_COMM_WORLD);
        }
        
        
        // send B to everyone, this the broadcast initiation line
        MPI_Bcast(B_transpose[0], p * n, MPI_DOUBLE, 0,  MPI_COMM_WORLD);        
        
        //note substitution of "rank0_end_row_index + 1" for m!
        //this selects the first sections of rows only
        if (use_openmp) {
            multiplyOMP_transpose(rank0_end_row_index + 1, n, p, A, B_transpose, C);
        } else {
            multiplySerial_transpose(rank0_end_row_index + 1, n, p, A, B_transpose, C);
        }

		for (int i = 1; i < world_size; i++) {
            int num_rows = row_indices[i][1] - row_indices[i][0] + 1;
			MPI_Recv(C[row_indices[i][0]], p * num_rows, MPI_DOUBLE, i, 0,  MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv C back
		}
           // Stop tracking execution time.
        clock_gettime(CLOCK_REALTIME, &stop);
        
        if (print) {
            printf("Print test gather\n");
            print_matrix(C, m, p);
        }
        
        free(A[0]);
        free(A);
        free(B_transpose[0]);
        free(B_transpose);
        free(C[0]);
        free(C);
        
        printf("Total wall clock execution time: transposing B: %d ms, MPI: %d ms\n", transpose_time_mills, ms_diff(start, stop));

            
    } else {
        MPI_Recv(dimensions, 3, MPI_INT, 0, 0,  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         
        m = dimensions[M_INDEX];
        n = dimensions[N_INDEX];
        p = dimensions[P_INDEX];                      
        //got the dimensions now allocate the matrices A and B (inner section) allocateMatrix(r, c)
                
        A_inner= allocateMatrix(m, n); 
        B_inner_transpose= allocateMatrix(p, n); 
        C_inner= allocateMatrix(m, p);
        
        MPI_Recv(A_inner[0], m * n, MPI_DOUBLE, 0, 0,  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        //This is a receive section for the B broadcast
        MPI_Bcast(B_inner_transpose[0], p * n, MPI_DOUBLE, 0,  MPI_COMM_WORLD);
        //***********************************************************************
        
        //ranks other than 0 already have truncated A and C matrices, no need to truncate
        multiplySerial_transpose(m, n, p, A_inner, B_inner_transpose, C_inner);
        
        if (use_openmp) {
            multiplyOMP_transpose(m, n, p, A_inner, B_inner_transpose, C_inner);
        } else {
            multiplySerial_transpose(m, n, p, A_inner, B_inner_transpose, C_inner);
        }


		MPI_Send(C_inner[0], m * p, MPI_DOUBLE, 0, 0,  MPI_COMM_WORLD); // send C to rank 0
        free(A_inner[0]);
        free(A_inner);
        free(B_inner_transpose[0]);
        free(B_inner_transpose);
        free(C_inner[0]);
        free(C_inner);
        
    }	
    // Finalize the MPI environment.  
	MPI_Finalize();
}


