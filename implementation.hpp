#include <unordered_map>
#include <omp.h>
#include "helpers.hpp"

unsigned long SequenceInfo::gpsa_sequential(float** S) {
    unsigned long visited = 0;

	// Boundary
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
		visited++;
	}

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
		visited++;
	}
	
	// Main part
	for (unsigned int i = 1; i < rows; i++) {
		for (unsigned int j = 1; j < cols; j++) {
			float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
			float del = S[i - 1][j] + gap_penalty;
			float insert = S[i][j - 1] + gap_penalty;
			S[i][j] = std::max({match, del, insert});
		
			visited++;
		}
	}

    return visited;
}

// Taskloop version grain size can be specified with grain_size, or block sizes. You can use both, or just one of them.
unsigned long SequenceInfo::gpsa_taskloop(float** S, long grain_size=1, int block_size_x=1, int block_size_y=1) {
    unsigned long visited = 0;

    // I determine the appropriate block size from the given parameters
    unsigned int block_size = grain_size;
    if (block_size <= 1 && block_size_x > 1) {
        block_size = block_size_x;
    }
    if (block_size <= 1) {
        block_size = 64;
    }

    // Boundary
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        visited++;
    }

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        visited++;
    }

    // I process the matrix using block-based wavefront parallelization here
    unsigned int num_blocks_rows = (rows - 1 + block_size - 1) / block_size;
    unsigned int num_blocks_cols = (cols - 1 + block_size - 1) / block_size;

    #pragma omp parallel shared(S, X, Y)
    {
        #pragma omp single
        {
            for (unsigned int block_diagonal = 0; block_diagonal < num_blocks_rows + num_blocks_cols - 1; block_diagonal++) {

                unsigned int first_block_row = (block_diagonal < num_blocks_cols) ? 0 : block_diagonal - num_blocks_cols + 1;
                unsigned int last_block_row = (block_diagonal < num_blocks_rows) ? block_diagonal : num_blocks_rows - 1;

                #pragma omp taskgroup task_reduction(+:visited)
                {
                    #pragma omp taskloop grainsize(1) shared(S) firstprivate(block_diagonal, first_block_row, last_block_row, block_size) in_reduction(+:visited)
                    for (unsigned int block_row = first_block_row; block_row <= last_block_row; block_row++) {
                        unsigned int block_col = block_diagonal - block_row;

                        unsigned int row_start = block_row * block_size + 1;
                        unsigned int col_start = block_col * block_size + 1;

                        if (row_start >= (unsigned int)rows || col_start >= (unsigned int)cols) continue;

                        unsigned int row_end = (row_start + block_size < (unsigned int)rows) ? row_start + block_size : (unsigned int)rows;
                        unsigned int col_end = (col_start + block_size < (unsigned int)cols) ? col_start + block_size : (unsigned int)cols;

                        unsigned int local_count = 0;

                        for (unsigned int i = row_start; i < row_end; i++) {
                            for (unsigned int j = col_start; j < col_end; j++) {
                                float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
                                float del = S[i - 1][j] + gap_penalty;
                                float insert = S[i][j - 1] + gap_penalty;
                                S[i][j] = std::max({match, del, insert});

                                local_count++;
                            }
                        }

                        visited += local_count;
                    }
                }
            }
        }
    }

    return visited;
}

// Explicit tasks version grain size can be specified with grain_size, or block sizes. You can use both, or just one of them.
unsigned long SequenceInfo::gpsa_tasks(float** S, long grain_size=1, int block_size_x=1, int block_size_y=1) {
    unsigned long visited = 0;

    // I determine the appropriate block size from the given parameters
    unsigned int block_size = grain_size;
    if (block_size <= 1 && block_size_x > 1) {
        block_size = block_size_x;
    }
    if (block_size <= 1) {
        block_size = 64;
    }

    // Boundary
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        visited++;
    }

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        visited++;
    }

    // I process the matrix using block-based wavefront parallelization with explicit tasks
    unsigned int num_blocks_rows = (rows - 1 + block_size - 1) / block_size;
    unsigned int num_blocks_cols = (cols - 1 + block_size - 1) / block_size;

    #pragma omp parallel shared(S, X, Y)
    {
        #pragma omp single
        {
            for (unsigned int block_diagonal = 0; block_diagonal < num_blocks_rows + num_blocks_cols - 1; block_diagonal++) {

                unsigned int first_block_row = (block_diagonal < num_blocks_cols) ? 0 : block_diagonal - num_blocks_cols + 1;
                unsigned int last_block_row = (block_diagonal < num_blocks_rows) ? block_diagonal : num_blocks_rows - 1;

                #pragma omp taskgroup task_reduction(+:visited)
                {
                    for (unsigned int block_row = first_block_row; block_row <= last_block_row; block_row++) {
                        unsigned int block_col = block_diagonal - block_row;

                        unsigned int row_start = block_row * block_size + 1;
                        unsigned int col_start = block_col * block_size + 1;

                        if (row_start >= (unsigned int)rows || col_start >= (unsigned int)cols) continue;

                        unsigned int row_end = (row_start + block_size < (unsigned int)rows) ? row_start + block_size : (unsigned int)rows;
                        unsigned int col_end = (col_start + block_size < (unsigned int)cols) ? col_start + block_size : (unsigned int)cols;

                        #pragma omp task firstprivate(row_start, row_end, col_start, col_end) shared(S, X, Y) in_reduction(+:visited)
                        {
                            unsigned int local_count = 0;

                            for (unsigned int i = row_start; i < row_end; i++) {
                                for (unsigned int j = col_start; j < col_end; j++) {
                                    float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
                                    float del = S[i - 1][j] + gap_penalty;
                                    float insert = S[i][j - 1] + gap_penalty;
                                    S[i][j] = std::max({match, del, insert});

                                    local_count++;
                                }
                            }

                            visited += local_count;
                        }
                    }
                }
            }
        }
    }

    return visited;
}
