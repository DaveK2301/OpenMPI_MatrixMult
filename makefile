mpi_matrix: mpi_matrix.c
	mpicc mpi_matrix.c -o mpi_matrix -funroll-loops -fomit-frame-pointer -O2 -fopenmp
