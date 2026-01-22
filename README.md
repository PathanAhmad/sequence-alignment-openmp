## Sequence Alignment with OpenMP (Global Pairwise Sequence Alignment)

This project implements **global pairwise sequence alignment** for long DNA-like sequences in C++, including:

- **Sequential dynamic programming** baseline
- **OpenMP taskloop**-based parallelization
- **OpenMP explicit tasks**-based parallelization

The core idea is to fill a dynamic programming matrix \( S \) using a scoring scheme (match, mismatch, gap) and then perform a traceback to recover the aligned sequences. The parallel versions use **wavefront (anti-diagonal) parallelization** over blocks of the matrix.

### Features

- **Three execution modes**
  - Sequential DP (`gpsa_sequential`)
  - Parallel DP using OpenMP **taskloop** (`gpsa_taskloop`)
  - Parallel DP using OpenMP **explicit tasks** (`gpsa_tasks`)
- **Configurable granularity**
  - `--grain_size` to tune task granularity
  - `--block_size_x`, `--block_size_y` for block-based processing
- **Multiple input datasets**
  - Tiny debug sequences
  - Medium sequences
  - Large sequences for stress testing and performance measurements
- **Validation**
  - Parallel implementations verify their output against the sequential result.

### Project Structure

- `main.cpp` – CLI, argument parsing, runs selected execution mode(s), measures runtime, prints stats.
- `implementation.hpp` – Implementations of:
  - `SequenceInfo::gpsa_sequential`
  - `SequenceInfo::gpsa_taskloop`
  - `SequenceInfo::gpsa_tasks`
- `helpers.hpp` – Utilities for allocation, deallocation, I/O, traceback, verification, and argument parsing.
- `README.txt` – Original assignment/usage notes from the course environment.
- `X*.txt`, `Y*.txt`, `simple*.txt`, `simple-longer-*.txt` – Example sequence files of different sizes.

### Build Instructions

This is a standard C++ project using **C++20** and **OpenMP**.

#### Build with `g++` (e.g., on Linux or WSL)

```bash
g++ -O2 -std=c++20 -fopenmp -o gpsa main.cpp
```

If `helpers.hpp` / `implementation.hpp` are in the same directory as `main.cpp` (as in this project), no extra include flags are needed.

#### Build on Windows (MSVC, from x64 Native Tools Command Prompt)

```bash
cl /O2 /std:c++20 /openmp main.cpp
```

This will produce `main.exe` (or `gpsa.exe` if you rename the output). Make sure the OpenMP flag is enabled (`/openmp` for MSVC).

### How to Run

By default, the program looks for `X.txt` and `Y.txt` in the working directory and writes a sequential alignment to `aligned-sequential.txt`.

```bash
./gpsa
```

You can also pass arguments (as supported in `main.cpp` and described originally in `README.txt`):

- **Custom input sequences**

```bash
./gpsa --x <sequence1_filename> --y <sequence2_filename>
```

- **Granularity modifiers**

```bash
./gpsa --x <sequence1_filename> --y <sequence2_filename> --grain_size <integer>
./gpsa --x <sequence1_filename> --y <sequence2_filename> --block_size_x <integer> --block_size_y <integer>
```

Depending on `exec_mode`, the program can run:

- Sequential only
- Taskloop only
- Explicit tasks only
- Or all three, comparing performance and verifying correctness

### Example Datasets

From the original assignment:

- `X.txt`, `Y.txt` – Random large sequences, size roughly `[51480 x 53640]`.
- `X2.txt`, `Y2.txt` – Large sequences with nicely divisible dimensions (`[32768 x 32768]`).
- `X3.txt`, `Y3.txt` – Medium sequences, also with nicely divisible dimensions (`[16384 x 16384]`).
- `simple1.txt`, `simple2.txt` – Very small sequences (`[3 x 5]`) for debugging and inspecting the DP matrix.
- `simple-longer-1.txt`, `simple-longer-2.txt` – Small, slightly longer sequences (`[20 x 20]`) for debugging.

### Output

For each execution mode, the program reports:

- Runtime in seconds
- Number of DP matrix entries visited
- Alignment score
- Similarity and identity scores
- Gap count and final aligned length
- (When applicable) Verification that parallel results match the sequential baseline

The aligned sequences are written to:

- `aligned-sequential.txt`
- `aligned-taskloop.txt`
- `aligned-tasks.txt`

### Parallelization Approach

The DP matrix is filled using a standard recurrence:

- \( \text{match} = S[i-1][j-1] + (\text{X[i-1]} == \text{Y[j-1]} ? \text{match\_score} : \text{mismatch\_score}) \)
- \( \text{del} = S[i-1][j] + \text{gap\_penalty} \)
- \( \text{insert} = S[i][j-1] + \text{gap\_penalty} \)
- \( S[i][j] = \max(\text{match}, \text{del}, \text{insert}) \)

The parallel implementations:

- Partition the matrix into **blocks**.
- Process blocks in **wavefront order** along anti-diagonals, ensuring dependencies are respected.
- Use OpenMP `taskloop` and explicit `task` constructs with a reduction on the visited entry counter.

This design makes it easier to experiment with **task granularity** and understand how OpenMP scheduling and overheads affect performance.

### Possible Extensions

If you want to extend this project, here are some ideas:

- Add **local alignment** (Smith–Waterman) as an additional mode.
- Implement **affine gap penalties** (separate open/extend costs).
- Add **profiling scripts** to sweep over `grain_size` / block sizes and plot performance.
- Add unit tests for small matrices to automatically validate correctness.

