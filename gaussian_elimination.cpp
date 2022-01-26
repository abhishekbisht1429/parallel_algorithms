//
// Created by reckoner1429 on 23/01/22.
//

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <fstream>

void gen_matrix(double *matrix, int n) {
    srand(time(NULL));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n + 1; j++)
            // Random number between 100 and -100
            matrix[i * n + j] = (double(rand()) / double(RAND_MAX)) *
                    (100 - -100) + -100;
}

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cout<<"Invalid number of args\n";
        return 0;
    }

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int m_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    /* synchronize all processes */
    MPI_Barrier ( MPI_COMM_WORLD ) ;
    double start_time = MPI_Wtime () ;

    /* **************** Main Code Begin ******************** */

    /* io streams - only to be used by rank 0 */
    std::ifstream ifs;
    std::ofstream ofs;

    /* number of rows - input from user */
    int n = 0;

    /* 2d array for matrix */
    double *matrix;

    /* get input from file */
    if(m_rank == 0) {
        ifs = std::ifstream(argv[1]);
        ofs = std::ofstream(argv[2]);

        /* input number of rows */
        ifs>>n;

        /* input the augmented matrix */
        matrix = new double[n*(n+1)];
        gen_matrix(matrix, n);
        for(int i=0; i<n; ++i) {
            for(int j=0; j<n+1; ++j)
//                ifs>>matrix[i*(n+1)+j];
                ofs<<matrix[i*(n+1) + j]<<" ";
            ofs<<"\n";
        }
        ofs<<"\n";
    }

    /* send value to n to other processes */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* number of cols */
    int n_cols = n+1;

    /* ######## Scatter data ########### */
    /* number of divs */
    int n_divs = n / world_size;
    int rem = n % world_size;

    /* matrix to store the data for each process */
    double *data = new double[(n_divs + (rem!=0)) * n_cols];

    /* to store the length of data for each rank */
    int data_row_count = n_divs;

    /* scatter augmented matrix among processes */
    for(int i=0; i < n_divs; ++i) {
        MPI_Scatter(&matrix[i * n_cols * world_size],
                    n_cols,
                    MPI_DOUBLE,
                    &data[i * n_cols],
                    n_cols,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
    }

    /* scatter the remaining rows (if any)*/
    if(rem > 0) {
        if(m_rank == 0) {
            memcpy(&data[n_divs * n_cols],
                   &matrix[n_divs * world_size * n_cols],
                   n_cols * sizeof(double));

            for(int i=1; i<rem; ++i) {
                MPI_Send(&matrix[(n_divs * world_size * n_cols) + (i*n_cols)],
                         n_cols,
                         MPI_DOUBLE,
                         i,
                         0,
                         MPI_COMM_WORLD);
            }
            ++data_row_count;
        } else if(m_rank < rem) {
            MPI_Recv(&data[n_divs * (n+1)],
                     (n+1),
                     MPI_DOUBLE,
                     0,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            ++data_row_count;
        }
    }

    /* all processes wait till data is scattered */
    MPI_Barrier(MPI_COMM_WORLD);

    /*
     * ##################################################
     * Start gaussian elimination after data is scattered
     */

    double *row = new double[n_cols];


    for(int i=0; i<n; ++i) {
        /* index of row in submatrix */
        int local_i = i / world_size;

        /* rank to which the row is assigned */
        int row_rank = i % world_size;

        /* only the rank to which the row is assigned can normalize it */
        if(row_rank == m_rank) {
            /* normalization */
            double pivot = data[local_i * n_cols + i];
            for(int j=i+1; j<n_cols; ++j)
                data[local_i * n_cols + j] /= pivot;
            data[local_i * n_cols + i] = 1;

            /* send the row to other ranks after normalization */
            memcpy(row, &data[local_i * n_cols], n_cols * sizeof(double));
            MPI_Bcast(row,
                       n_cols,
                       MPI_DOUBLE,
                       row_rank,
                       MPI_COMM_WORLD);

            /* start elimination in parallel */
            for(int local_i2 = local_i + 1; local_i2 < data_row_count;
                                                        ++local_i2) {
                double scale = data[local_i2 * n_cols + i];

                data[local_i2 * n_cols + i] = 0;
                for(int j = i+1; j < n_cols; ++j)
                    data[local_i2 * n_cols + j] -= scale * row[j];

            }
        } else {
            /* wait till a row is received */
            MPI_Bcast(row,
                      n_cols,
                      MPI_DOUBLE,
                      row_rank,
                      MPI_COMM_WORLD);

            /* start elimination */
            for(int local_i2 = local_i; local_i2 < data_row_count; ++local_i2) {
                if((m_rank > row_rank) || (local_i2 > local_i)) {
                    double scale = data[local_i2 * n_cols + i];

                    for(int j = i+1; j < n_cols; ++j)
                        data[local_i2 * n_cols + j] -= scale * row[j];
                    data[local_i2 * n_cols + i] = 0;
                }
            }
        }
    }

    /* wait for all processes to finish */
    MPI_Barrier(MPI_COMM_WORLD);

    for(int i=0; i<n_divs; ++i) {
        MPI_Gather(&data[i * n_cols],
                   n_cols,
                   MPI_DOUBLE,
                   &matrix[i * world_size * n_cols],
                   n_cols,
                   MPI_DOUBLE,
                   0,
                   MPI_COMM_WORLD);
    }

    /* Gather remaining data */
    if(rem > 0) {
        if(m_rank == 0) {
            memcpy(&matrix[n_divs * world_size * n_cols],
                   &data[n_divs * n_cols],
                   n_cols * sizeof(double));
            /* receive processed data from other ranks */
            for(int i=1; i<rem; ++i) {
                MPI_Recv(&matrix[(n_divs * world_size * n_cols) + (i * n_cols)],
                         n_cols,
                         MPI_DOUBLE,
                         i,
                         0,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        } else if(m_rank < rem) {
            /* send processed data to rank 0 */
            MPI_Send(&data[n_divs * n_cols],
                     n_cols,
                     MPI_DOUBLE,
                     0,
                     0,
                     MPI_COMM_WORLD);
        }
    }

    /*
     * ################################
     * End - Gaussian Elimination
     */


    /*
     * #################################
     * Start - Solution using Back substitution
     */
    if(m_rank == 0) {
        /* Print the upper triangular matrix */
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                ofs << matrix[i * n_cols + j] << " ";
            }
            ofs << "\n";
        }

        ofs<<"\n";

        double *solution = new double[n];
        solution[n-1] = matrix[n * (n+1) - 1];
        for(int i=n-2; i>=0; --i) {
            double b = matrix[i * (n+1) + n];
            double sum = 0;
            for(int j=n-1; j>i; --j)
                sum += matrix[i * (n + 1) + j] * solution[j];
            solution[i] = (b - sum) / matrix[i*(n+1) + i];
        }

        /* print the solution vector */
        for(int i=0; i<n; ++i)
            ofs<<solution[i]<<" ";
        ofs<<"\n";
    }
    /* #######################################
     * End - Back Substitution Complete
     * */


    /* **************** Main Code End  ********************* */
    MPI_Barrier ( MPI_COMM_WORLD ) ;
    double end_time = MPI_Wtime () - start_time ;
    double maxTime ;

    // get max program run time for all processes
    MPI_Reduce ( & end_time , & maxTime , 1 , MPI_DOUBLE ,
                 MPI_MAX , 0 , MPI_COMM_WORLD ) ;
    if ( m_rank == 0 ) {
        std::cout << "Total time (s): "<<maxTime<<"\n" ;
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}