#include <omp.h>

// ********************** All Multiplication Methods below this line
int multiplySerial_transpose(int m, int n, int p, double **A, double **B, double **C) {
	int i, j, k;
    
	/*this section works down the m rows, then computing down p columns*/
	for (i = 0; i < m; i++) {
    	for (j = 0; j < p; j++) {
         	for (k = 0; k < n; k++) {
                //just transposed the B arguments here
    			C[i][j] += A[i][k]*B[j][k];
            }
    	}
	}
	return 0;
}


int multiplyOMP_transpose(int m, int n, int p, double **A, double **B, double **C/*, int threads*/) {

	#pragma omp parallel for schedule(static) /*num_threads(threads)*/
	for (int i = 0; i < m; i++) {
    	for (int j = 0; j < p; j++) {
         	for (int k = 0; k < n; k++) {
                //just transposed the B arguments here
    			C[i][j] += A[i][k]*B[j][k];
            }
    	}
	}
	return 0;
}

// ********************** All Utility Methods below this line
/*void readinMartix(double **A, char *path, int m, int n) {
	float d;	
	FILE *f = fopen(path, "r");

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			A[i][j] = d;
		}
	}

	fclose(f);
}*/

void print_matrices_transpose(int m, int n, int p, double **A, double **B, double **C) {
	int i, j;
	/*print values for A*/
	for (i = 0; i < m; i++) {
    	for (j = 0; j < n; j++) {
        	printf("\t%5.4f",A[i][j]);
    	}
    	printf("\n");
	}
	printf("\n");
	/*print values for B*/
    //switched values for transpose here
	for (i = 0; i < p; i++) {
    	for (j = 0; j < n; j++) {
         	printf("\t%5.4f",B[i][j]);
    	}
    	printf("\n");
	}
    	printf("\n");
	/*print values for C*/
	for (i = 0; i < m; i++) {
    	for (j = 0; j < p; j++) {
         	printf("\t%5.4f",C[i][j]);
    	}
    	printf("\n");
	}
    printf("\n");
}

 void print_matrix(double **A, int m, int n) {
    int i, j;
	/*print values for A*/
	for (i = 0; i < m; i++) {
    	for (j = 0; j < n; j++) {
        	printf("\t%5.4f",A[i][j]);
    	}
    	printf("\n");
	}
	printf("\n");
}

double **allocateMatrix(int r, int c) {
        double ** A;
        int i;
        A = (double **)malloc(sizeof(double *) * r);
        A[0] = (double *)malloc(sizeof(double) * r * c);
        for (i = 0; i < r; i++)
            A[i] = (*A + c * i);
        return A;
}

void createRandValues(double **A, int m, int n) {
	double start_val = 0.0; //hard coded this time
        double range = 1.0;     //ditto
	
	for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = start_val + (rand()/(double) RAND_MAX) * range;
                /* above is random version, below is just for test purposes*/
                /*A[i][j] = (double) (i * n + j) + 1;*/
            }   
        }
}

double **transposeB(double **B, int p, int n) {
	double **B_transpose = allocateMatrix(p, n);
        
        //printf("Copying over values from B to B transpose\n");
        /*copy values from B to B transpose*/
        //switched values for transpose
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < n; j++) {
                B_transpose[i][j] = B[j][i];
            }   
        }
return B_transpose;
}

/*
 * This timing code was adopted from
 * Matt Alden's starter code for TCSS 372,
 * and Dave Kaplan's Homework 5 Wi 14
 * submission on Catalyst
 */
int ms_diff(struct timespec start, struct timespec stop) {

    return 1000 * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) / 1000000;

}
