#ifndef VEGAS_CUH
#define VEGAS_CUH
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

__device__ __forceinline__ float func(float *x, float params)
{
    // Example function: Gaussian centered at 0.5 in each dimension with width params[0]
    float exponent = 0;
    float scale = params;
    for (int i = 0; i < 4; ++i)
    {
        exponent += ((x[i]) * (x[i]));
    }
    exponent = sqrtf(exponent);
    return expf(-scale * (exponent) * (exponent)) * cosf(exponent) / ((exponent - 0.5f) * (exponent - 0.5f) + 0.001f);
}

struct vegas_opts
{
    float *x_0; // Lower bounds of integration
    float *x_1; // Upper bounds of integration

    int n_dim;             // Number of dimensions
    int n_st;              // Number of stratifications per dimension
    int num_blocks = 16;   // Number of blocks used in the monte carlo section of the code
    int num_threads = 64;  // Number of threads used in the monte carlo section of the code
    int n_g = 1024;        // Number of grid divisions
    int n_ev_grid = 25;    // Number of evaluations per thread (total evaluations = n_ev * num_threads * num_blocks) used to create the initial grid
    int n_ev_strat = 1000; // Number of evaluations per thread (total evaluations = n_ev * num_threads * num_blocks) used initially during stratified sampling. This is only used initially, but the total evaluations will change as the algorithm progresses

    int n_grid_iters_max = 15;   // Max num. iterations to adjust the grid
    float grid_tol = 0.01f;      // Tolerance for grid adjustment--if the grid changes by less than this amount, we stop adjusting it
    int n_stratified_iters = 10; // Number of itterations using adaptive hypercube stratifed sampling

    int num_init_threads = 128; // Number of threads used in some other kernels that can be easily parallelized

    float smoothing = 8.0f;
    float alpha = 1.5f;
    float beta = 0.75f;

    bool print_output = false;
};

struct vegas_environment
{
    curandState *curand_states; // curand states for each thread
    float *x_0;
    float *x_1;    // integrate over [x_0[0], x_1[0]]x[x_0[1], x_1[1]]x...x[x_0[d-1], x_1[d-1]]
    float *x_k;    // Grid points
    float *jf2;    // variance in each bin
    int *num_hits; // number of samples in each bin

    int *hypercube_num_ev; // Number of evaluations in each hypercube
    float *jf_h;           // monte carlo estimate of the integral in each hypercube
    float *jf2_h;          // monte carlo estimate of the variance in each hypercube
    int *threadWork;       // Number of hypercubes each thread will handle
    float *total_d;        // Total smoothed variance in each dimension
    float *I_est;          // Estimated integral value
    float *var_est;        // Estimated variance
    int *n_ev_tot_strat;   // total evaluations for stratified sampling

    float *smoothed_hypercube_variance; // Smoothed variance in each hypercube, never used outside kernels
};

void create_vegas_environment(vegas_environment &env, vegas_opts opts);

__global__ void initialize_vegas_environment(vegas_environment env, vegas_opts opts);

__global__ void adjust_and_sum(float *bins, int *num_hits, float *total_d, int n_g);

template <int n_dim, int n_g, typename params, float (*f)(float *, params)>
__global__ void fill_bins(params p, float *I_est, float *var_est, float *jf2, int *num_hits, curandState *cu_states, float *x_k, int n_ev)
{
    __shared__ float shared_jf2[n_dim * n_g];
    __shared__ float shared_num_hits[n_dim * n_g];
    for (int i = threadIdx.x; i < n_dim * n_g; i += blockDim.x)
    {
        shared_jf2[i] = 0;
        shared_num_hits[i] = 0;
    }
    __syncthreads();

    curandState localstate = cu_states[blockIdx.x * blockDim.x + threadIdx.x];

    float x_y[n_dim];
    int xy_idx[n_dim];

    float y;
    float Jac;
    float fval;
    float I_mc = 0;
    float var = 0;
    for (int i = 0; i < n_ev; ++i)
    { // add a single evluation to I_mc, var, and the corresponding bins
        Jac = 1;
        for (int dim = 0; dim < n_dim; ++dim)
        {
            y = 1 - curand_uniform(&localstate); // now y is in [0,1)
            xy_idx[dim] = (int)floor(n_g * y);   // 0<=xy_idx<n_g
            if (xy_idx[dim] == n_g)
            {
                xy_idx[dim] -= 1;
            }
            x_y[dim] = __ldg(&x_k[dim * (n_g + 1) + xy_idx[dim]]) + (__ldg(&x_k[xy_idx[dim] + 1]) - __ldg(&x_k[dim * (n_g + 1) + xy_idx[dim]])) * (n_g * y - xy_idx[dim]);
            Jac *= n_g * (__ldg(&x_k[dim * (n_g + 1) + xy_idx[dim] + 1]) - __ldg(&x_k[dim * (n_g + 1) + xy_idx[dim]]));
        }
        fval = f(x_y, p);
        if (fval != fval)
        {
            --i;
            continue;
        }
        I_mc += fval * Jac;
        var += fval * fval * Jac * Jac;
        for (int dim = 0; dim < n_dim; ++dim)
        {
            atomicAdd(&shared_jf2[dim * n_g + xy_idx[dim]], fval * fval * Jac * Jac);
            atomicAdd(&shared_num_hits[dim * n_g + xy_idx[dim]], 1);
        }
    }

    cu_states[blockIdx.x * blockDim.x + threadIdx.x] = localstate;
    atomicAdd(I_est, I_mc);
    atomicAdd(var_est, var);
    __syncthreads();
    for (int i = threadIdx.x; i < n_dim * n_g; i += blockDim.x)
    {
        atomicAdd(&jf2[i], shared_jf2[i]);
        atomicAdd(&num_hits[i], shared_num_hits[i]);
    }
}

template <int n_dim, int n_g, typename params, float (*f)(float *, params)>
__global__ void fill_bins_strat(params p, float *I_tot, float *I_h, float *var_h, float *x_k, int *hypercube_n_eval, int n_st, int *thread_tasks, curandState *cu_states, int num_cubes)
{
    /*
    __shared__ float shared_jf2[n_dim*n_g];
    __shared__ float shared_num_hits[n_dim*n_g];
    for(int i=threadIdx.x;i<n_dim*n_g;i+=blockDim.x)
    {
        shared_jf2[i]=0;
        shared_num_hits[i]=0;
    }
    __syncthreads();*/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localstate = cu_states[idx];
    float y;
    float x_y[n_dim];
    int xy_idx[n_dim];
    int total_thr = blockDim.x * gridDim.x;
    int thread_position = idx;
    int start_block = thread_tasks[3 * thread_position];
    int n_ev_start_block = thread_tasks[3 * thread_position + 1];
    int total_needed_evs = thread_tasks[3 * thread_position + 2];

    int num_hypercubes = num_cubes;
    float hypercube_vol = 1 / (1.0f * num_hypercubes);
    int hypercube_idxs[n_dim] = {};

    int remainder;
    int temp_idx = start_block;
    for (int i = 0; i < n_dim; ++i)
    {
        remainder = temp_idx % n_st;
        hypercube_idxs[i] = remainder;
        temp_idx = (temp_idx - remainder) / n_st;
    }

    float Jac;
    float fval;
    float I_mc = 0;
    float var = 0;
    float I_sum = 0;
    float var_sum = 0;

    int curr_hypercube = start_block;
    int num_hypercube_evals = n_ev_start_block;

    int needed_hypercube_evals = hypercube_n_eval[curr_hypercube];

    int total_evals = 0;
    while (true)
    { // add a single evluation to I_mc, var, and the corresponding bins
        if (num_hypercube_evals == needed_hypercube_evals || total_evals == total_needed_evs)
        {
            I_mc /= needed_hypercube_evals;
            var /= needed_hypercube_evals;

            I_mc *= hypercube_vol;
            var *= hypercube_vol;
            I_sum += I_mc;

            atomicAdd(&var_h[curr_hypercube], var); // need to subtract I_h^2 once all threads finish
            atomicAdd(&I_h[curr_hypercube], I_mc);

            //
            if (total_evals == total_needed_evs)
            {
                break;
            }
            num_hypercube_evals = 0; // reset counter
            var = 0;
            I_mc = 0;
            ++curr_hypercube;    // Now we move on to the next hypercube
            ++hypercube_idxs[0]; // The list of hypercubes can be indexed as h_(d-1)*n_st^(d-1)+h_(d-2)*n_st^(d-2)...h_0 where the h_i's are the position on the (i+1)'th dimension's grid

            for (int j = 0; j < n_dim - 1; ++j)
            {
                if (hypercube_idxs[j] == n_st) // if d0=n_st we've already traversed all the cubes in the first dimension, so we then increment to the next position in the 2nd dimension. If d1=n_st, same thing--continue through all dimensions
                {
                    hypercube_idxs[j] = 0;
                    ++hypercube_idxs[j + 1];
                }
            }
            needed_hypercube_evals = hypercube_n_eval[curr_hypercube];
        }
        Jac = 1;
        for (int dim = 0; dim < n_dim; ++dim)
        {
            y = 1 - curand_uniform(&localstate); // now y is in [0,1)
            // we want y to be in our current hypercube
            y = (hypercube_idxs[dim] + y) / n_st;
            // now we can just continue everything as normal
            xy_idx[dim] = (int)floor(n_g * y); // 0<=xy_idx<n_g
            if (xy_idx[dim] == n_g)
            {
                xy_idx[dim] -= 1;
            }
            x_y[dim] = __ldg(&x_k[dim * (n_g + 1) + xy_idx[dim]]) + (__ldg(&x_k[dim * (n_g + 1) + xy_idx[dim] + 1]) - __ldg(&x_k[dim * (n_g + 1) + xy_idx[dim]])) * (n_g * y - xy_idx[dim]);
            Jac *= n_g * (__ldg(&x_k[dim * (n_g + 1) + xy_idx[dim] + 1]) - __ldg(&x_k[dim * (n_g + 1) + xy_idx[dim]]));
        }
        fval = f(x_y, p);
        I_mc += fval * Jac;
        var += fval * fval * Jac * Jac;
        total_evals++;
        num_hypercube_evals++;
    }

    cu_states[idx] = localstate;
    atomicAdd(I_tot, I_sum);
}

template <int n_g>
__global__ void smooth_and_compress_bins(float *bins, float *total_d, float alpha, float weight)
{
    float d_tot = total_d[blockIdx.x];
    int idx = threadIdx.x;
    int stride = blockDim.x;
    int i = idx;

    __shared__ float shared_bins[n_g];
    for (int j = idx; j < n_g; j += stride)
    {
        shared_bins[j] = bins[blockIdx.x * n_g + j];
    }
    __syncthreads();
    float d_adjusted = 0;

    float local_sum = 0;

    if (i == 0)
    {
        d_adjusted = ((weight - 1) * shared_bins[0] + shared_bins[1]) / (weight * d_tot);
        d_adjusted = pow((d_adjusted - 1) / log(d_adjusted), alpha);
        bins[blockIdx.x * n_g + 0] = d_adjusted;
        local_sum += d_adjusted;
        i += stride;
    }
    for (; i < n_g - 1; i += stride)
    {
        d_adjusted = (shared_bins[i - 1] + (weight - 2) * shared_bins[i] + shared_bins[i + 1]) / (weight * d_tot);
        d_adjusted = pow((d_adjusted - 1) / log(d_adjusted), alpha);
        bins[blockIdx.x * n_g + i] = d_adjusted;
        local_sum += d_adjusted;
    }
    if (i == n_g - 1)
    {
        d_adjusted = ((weight - 1) * shared_bins[n_g - 1] + shared_bins[n_g - 2]) / (weight * d_tot);
        d_adjusted = pow((d_adjusted - 1) / log(d_adjusted), alpha);
        bins[blockIdx.x * n_g + n_g - 1] = d_adjusted;
        local_sum += d_adjusted;
    }
    total_d[blockIdx.x] = 0;
    __syncthreads();
    atomicAdd(&total_d[blockIdx.x], local_sum); // get sum of weights for each dimension
}

template <int n_dim, int n_g>
__global__ void update_grid(float *x_k, float *new_bins, float *total_d)
{
    __shared__ float x_k_shared[(n_g + 1) * n_dim];
    for (int i = threadIdx.x; i < (n_g + 1) * n_dim; i += blockDim.x)
    {
        x_k_shared[i] = x_k[i];
    }
    __syncthreads();
    float s_d = 0;
    float d_avg = total_d[threadIdx.x] / n_g;
    int j = 0;
    x_k[threadIdx.x * (n_g + 1)] = x_k_shared[threadIdx.x * (n_g + 1)];
    for (int i = 1; i < n_g; ++i) // exclude x_0, x_N_g
    {
        while (s_d < d_avg)
        {
            s_d += new_bins[threadIdx.x * n_g + (j++)];
        }
        s_d -= d_avg;
        x_k[threadIdx.x * (n_g + 1) + i] = x_k_shared[threadIdx.x * (n_g + 1) + j] - s_d / new_bins[threadIdx.x * n_g + j - 1] * (x_k_shared[threadIdx.x * (n_g + 1) + j] - x_k_shared[threadIdx.x * (n_g + 1) + j - 1]);
    }
    x_k[threadIdx.x * (n_g + 1) + n_g] = x_k_shared[threadIdx.x * (n_g + 1) + n_g];
}

void SetThreadWork(const int *numEvals, int *threadWork, int totalEvals, int numTasks, int numThreads);
void SetThreadWorkConst(int *threadWork, int evalsPerTask, int numTasks, int numThreads);

__global__ void strat_adjust(float beta, float *var_h, float *I_h, float *final_var, int *hypercube_n_eval, float *varsum, int num_hypercubes);

__global__ void adjust_hypercube_sampling(int n_ev_tot, int num_hypercubes, float *varsum, float *var_h, int *hypercube_n_eval, int *n_ev_ptr);

template <int n_dim, int n_g, int n_st, typename params, float (*func)(float *, params)>
void vegas_fast(params p, vegas_environment &env, vegas_opts vopts)
{
    int total_threads = vopts.num_threads * vopts.num_blocks;
    int num_hypercubes = pow(vopts.n_st, n_dim);
    float alpha = vopts.alpha;
    float beta = vopts.beta;

    float smoothing = vopts.smoothing;

    float *x_k = env.x_k;
    float *I_est = env.I_est;
    float *var_est = env.var_est;
    curandState *curand_states = env.curand_states;
    int *hypercube_num_ev = env.hypercube_num_ev;
    float *total_d = env.total_d;

    float I;
    float var;
    float I_tot = 0;
    float var_tot = 0;
    float prev_var = 0;
    float total_d_copy[n_dim];

    int n_grid_evals_tot = vopts.n_ev_grid * total_threads;
    for (int i = 0; i < vopts.n_grid_iters_max; ++i)
    {
        fill_bins<n_dim, n_g, params, func><<<vopts.num_blocks, vopts.num_threads>>>(p, I_est, var_est, env.jf2, env.num_hits, curand_states, x_k, vopts.n_ev_grid);
        cudaDeviceSynchronize();
        prev_var = var;
        cudaMemcpy(&I, I_est, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&var, var_est, sizeof(float), cudaMemcpyDeviceToHost);
        I /= (n_grid_evals_tot);
        var = (var - I * I) / n_grid_evals_tot;
        var /= n_grid_evals_tot;
        if (i > 1)
        {
            if (fabs(var - prev_var) / prev_var < vopts.grid_tol)
            {
                printf("Variance converged, stopping grid adaptation early\n");
                cudaMemset(env.jf2, 0, n_dim * vopts.n_g * sizeof(float));
                cudaMemset(env.num_hits, 0, n_dim * vopts.n_g * sizeof(int));
                cudaMemset(I_est, 0, sizeof(float));
                cudaMemset(var_est, 0, sizeof(float));
                cudaMemset(total_d, 0, n_dim * sizeof(float));
                break;
            }
        }
        printf("Iteration %d: I=%.6e, var=%.6e\n", i + 1, I, var);
        adjust_and_sum<<<n_dim, vopts.num_init_threads>>>(env.jf2, env.num_hits, total_d, vopts.n_g);
        cudaDeviceSynchronize();
        smooth_and_compress_bins<n_g><<<n_dim, vopts.num_init_threads>>>(env.jf2, total_d, alpha, smoothing);
        cudaDeviceSynchronize();
        cudaMemcpy(total_d_copy, total_d, n_dim * sizeof(float), cudaMemcpyDeviceToHost);

        update_grid<n_dim, n_g><<<1, n_dim>>>(x_k, env.jf2, total_d);
        cudaDeviceSynchronize();
        cudaMemset(env.jf2, 0, n_dim * vopts.n_g * sizeof(float));
        cudaMemset(env.num_hits, 0, n_dim * vopts.n_g * sizeof(int));
        cudaMemset(I_est, 0, sizeof(float));
        cudaMemset(var_est, 0, sizeof(float));
        cudaMemset(total_d, 0, n_dim * sizeof(float));
    }

    int n_ev_tot_copy = 0;
    cudaMemcpy(&n_ev_tot_copy, env.n_ev_tot_strat, sizeof(int), cudaMemcpyDeviceToHost);

    int thread_tasks_host[3 * total_threads]; // for each thread: starting hypercube, number of evals in starting hypercube, total evals needed
    int hypercube_num_ev_host[num_hypercubes];
    for (int i = 0; i < vopts.n_stratified_iters; ++i)
    {
        fill_bins_strat<n_dim, n_g, params, func><<<vopts.num_blocks, vopts.num_threads>>>(p, I_est, env.jf_h, env.jf2_h, x_k, hypercube_num_ev, vopts.n_st, env.threadWork, curand_states, num_hypercubes);
        cudaDeviceSynchronize();

        strat_adjust<<<64, vopts.num_init_threads>>>(beta, env.jf2_h, env.jf_h, var_est, hypercube_num_ev, env.smoothed_hypercube_variance, num_hypercubes);
        cudaMemset(env.n_ev_tot_strat, 0, sizeof(int));
        cudaDeviceSynchronize();
        adjust_hypercube_sampling<<<64, vopts.num_init_threads>>>(n_ev_tot_copy, num_hypercubes, env.smoothed_hypercube_variance, env.jf2_h, hypercube_num_ev, env.n_ev_tot_strat);
        cudaDeviceSynchronize();
        cudaMemcpy(&I, I_est, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&var, var_est, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Final iteration: I=%.6e, var=%.6e\n", I, var);

        I_tot += I / var;
        var_tot += 1 / var;

        cudaMemcpy(&n_ev_tot_copy, env.n_ev_tot_strat, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Total evaluations: %d\n", n_ev_tot_copy);
        cudaMemcpy(hypercube_num_ev_host, hypercube_num_ev, num_hypercubes * sizeof(int), cudaMemcpyDeviceToHost);
        SetThreadWork(hypercube_num_ev_host, thread_tasks_host, n_ev_tot_copy, num_hypercubes, total_threads);
        cudaMemcpy(env.threadWork, thread_tasks_host, 3 * total_threads * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(I_est, 0, sizeof(float));
        cudaMemset(var_est, 0, sizeof(float));
        cudaMemset(env.jf2, 0, n_dim * vopts.n_g * sizeof(float));
        cudaMemset(env.num_hits, 0, n_dim * vopts.n_g * sizeof(int));
        cudaMemset(env.jf2_h, 0, num_hypercubes * sizeof(float));
        cudaMemset(env.jf_h, 0, num_hypercubes * sizeof(float));
        cudaMemset(env.smoothed_hypercube_variance, 0, sizeof(float));
    }

    printf("Accumulated: I=%.6e, var=%.6e\n", I_tot / var_tot, 1 / var_tot);
}

#endif