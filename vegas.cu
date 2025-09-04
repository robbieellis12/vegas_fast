#include "vegas.cuh"

// Always launch with n_dim blocks
// I could change this to use more blocks but this only needs to be ran a single time so it doesn't really matter
__global__ 
void initialize_vegas_environment(vegas_environment env, vegas_opts opts)
{
    int idx = threadIdx.x;
    int stride = blockDim.x;
    int n_g = opts.n_g;
    int total_threads = opts.num_threads * opts.num_blocks;
    int n_ev_tot_strat=opts.n_ev_strat*total_threads;
    int num_hypercubes=pow(opts.n_st,opts.n_dim);
    float x0val = env.x_0[blockIdx.x];
    float del = (env.x_1[blockIdx.x] - x0val) / (float)n_g;
    
    for (int i = idx; i < n_g + 1; i += stride)
    {
        env.x_k[(blockIdx.x) * (n_g + 1) + i] = x0val + i * del;
    }
    idx=blockDim.x*blockIdx.x+threadIdx.x;  
    stride = blockDim.x*gridDim.x;
    for(int i=idx;i<total_threads;i+=stride)
    {
        curand_init(1234, i, 0, &env.curand_states[i]);
    }
    int n_ev_tot_updated_local=0;
    for (int i = idx; i < num_hypercubes; i += stride)
    {
        env.hypercube_num_ev[i] = (int)(n_ev_tot_strat / (num_hypercubes * 1.0f));
        n_ev_tot_updated_local+=env.hypercube_num_ev[i];
    }
    atomicAdd(env.n_ev_tot_strat,n_ev_tot_updated_local);
   // printf("Total evals assigned in thread %d: %d\n",idx,n_ev_tot_updated_local);
}


void create_vegas_environment(vegas_environment &env, vegas_opts opts)
{
    int n_dim = opts.n_dim;
    int n_g = opts.n_g;
    int total_threads = opts.num_threads * opts.num_blocks;
    int total_evals=opts.n_ev_strat*total_threads;
    int num_hypercubes=pow(opts.n_st,opts.n_dim);
    if(total_evals<2*num_hypercubes)
    {
       std::cerr<<"Warning: Total evaluations ("<<total_evals<<") is less than the number of hypercubes ("<<num_hypercubes<<"). This will lead to some hypercubes not being sampled. Consider increasing n_ev, num_threads, or num_blocks."<<std::endl;
    }
    cudaMalloc(&env.curand_states, total_threads * sizeof(curandState));
    cudaMalloc(&env.x_0, n_dim * sizeof(float));
    cudaMalloc(&env.x_1, n_dim * sizeof(float));
    cudaMalloc(&env.x_k, n_dim * (n_g + 1) * sizeof(float));
    cudaMemcpy(env.x_0, opts.x_0, n_dim * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(env.x_1, opts.x_1, n_dim * sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&env.hypercube_num_ev, num_hypercubes * sizeof(int));
    cudaMalloc(&env.jf_h, num_hypercubes * sizeof(float));
    cudaMalloc(&env.jf2_h, num_hypercubes * sizeof(float));
    cudaMemset(env.jf_h, 0, num_hypercubes * sizeof(float));
    cudaMemset(env.jf2_h, 0, num_hypercubes * sizeof(float));
    cudaMalloc(&env.threadWork, 3*total_threads * sizeof(int));
    cudaMemset(env.threadWork, 0, 3*total_threads * sizeof(int));
    cudaMalloc(&env.n_ev_tot_strat, sizeof(int));
    cudaMemset(env.n_ev_tot_strat, 0, sizeof(int)); //This will be set in initialize_vegas_environment
    cudaMalloc(&env.total_d, n_dim * sizeof(float));
    cudaMemset(env.total_d, 0, n_dim * sizeof(float));
    
    cudaMalloc(&env.I_est, sizeof(float));
    cudaMalloc(&env.var_est, sizeof(float));
    cudaMemset(env.I_est, 0, sizeof(float));
    cudaMemset(env.var_est, 0, sizeof(float));

    cudaMalloc(&env.jf2, n_dim * n_g * sizeof(float));
    cudaMalloc(&env.num_hits, n_dim * n_g * sizeof(int));
    cudaMemset(env.jf2, 0, n_dim * n_g * sizeof(float));
    cudaMemset(env.num_hits, 0, n_dim * n_g * sizeof(int));
    cudaMalloc(&env.smoothed_hypercube_variance, sizeof(float));
    cudaMemset(env.smoothed_hypercube_variance, 0, sizeof(float));

    initialize_vegas_environment<<<n_dim, opts.num_init_threads>>>(env, opts);
    cudaDeviceSynchronize();
    int *threadWorkHost;
    threadWorkHost=(int*)malloc(3*total_threads*sizeof(int));
    SetThreadWorkConst(threadWorkHost, total_evals/num_hypercubes, num_hypercubes, total_threads);
    cudaMemcpy(env.threadWork,threadWorkHost,3*total_threads*sizeof(int),cudaMemcpyHostToDevice);
    free(threadWorkHost);
}



//Normalize the bins and sum the total variance in each dimension
__global__
void adjust_and_sum(float *bins,int *num_hits, float* total_d, int n_g)
{
    float local_sum=0;
    int idx=threadIdx.x;
    int stride=blockDim.x;
    for(int i=idx;i<n_g;i+=stride)
    {
       if(num_hits[blockIdx.x*n_g+i]!=0)
        {
        bins[blockIdx.x*n_g+i]/=num_hits[blockIdx.x*n_g+i];
        local_sum+= bins[blockIdx.x*n_g+i];
        }
    }
    __syncthreads();
    atomicAdd(&total_d[blockIdx.x],local_sum);

}



//Normalize variance in each hypercube
__global__ 
void strat_adjust(float beta, float* var_h, float* I_h, float *final_var, int* hypercube_n_eval,float *varsum,int num_hypercubes)
{
    float local_sum=0;
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int stride=blockDim.x*gridDim.x;
    float local_var_tot=0;
    for(int i=idx;i<num_hypercubes;i+=stride)
    {
        var_h[i]=1.00001f*var_h[i]/num_hypercubes-I_h[i]*I_h[i];
        if(var_h[i]<0) //Floating point precision issues can lead to a small negative variance, which destroys the program--this helps mitigate this issue
        {
            var_h[i]=0;
        }
        local_var_tot+=var_h[i]/hypercube_n_eval[i];
        var_h[i]=pow(var_h[i],beta/2.0f);
        local_sum+=var_h[i];
    }
    atomicAdd(varsum,local_sum);
    atomicAdd(final_var,local_var_tot);


}

__global__
void adjust_hypercube_sampling(int n_ev_tot,int num_hypercubes,float* varsum, float *var_h,int* hypercube_n_eval,int* n_ev_ptr)
{
    int local_new_n_ev_sum=0;
    float varsum_local=*varsum;
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int stride=blockDim.x*gridDim.x;
    for(int i=idx;i<num_hypercubes;i+=stride)
    {
        hypercube_n_eval[i]=max(2,(int) (n_ev_tot*var_h[i]/varsum_local));
        local_new_n_ev_sum+=hypercube_n_eval[i];
    }
    atomicAdd(n_ev_ptr,local_new_n_ev_sum);
}



void SetThreadWork(const int* numEvals,int* threadWork,int totalEvals, int numTasks, int numThreads)
{
int numTasksInCurrentThread=0;
double evalsPerThread=totalEvals/(1.0*numThreads);
int accumulatedEvals=0;
int currentThreadIdx=0;
int currentTaskIdx=0;
int numEvalsInCurrentTask=numEvals[0];
int remainingEvalsInTask=numEvalsInCurrentTask;
int currentBenchmark=evalsPerThread;
int remainingEvalsForThread=currentBenchmark;
int evalsForThread=0;
threadWork[0]=0;
threadWork[1]=0;
threadWork[2]=currentBenchmark;
while(true)
{
       if(remainingEvalsInTask>=remainingEvalsForThread) //enough evals in the current task to completely occupy the current thread
       {
           accumulatedEvals+=remainingEvalsForThread;
           remainingEvalsInTask-=remainingEvalsForThread;
           currentThreadIdx++;
           if(currentThreadIdx==numThreads)
           {
               break;
           }
           currentBenchmark=(currentThreadIdx+1)*evalsPerThread;
           if(currentThreadIdx==numThreads-1)
           {
               //std::cout<<"Last thread benchmark: "<<currentBenchmark<<", total evals: "<<totalEvals<<std::endl;
           }
           remainingEvalsForThread=currentBenchmark-accumulatedEvals;
           evalsForThread=remainingEvalsForThread;
           if(remainingEvalsInTask==0)
           {
             //  std::cout<<"Task: "<<currentTaskIdx<<" looped"<<std::endl;
           threadWork[3*currentThreadIdx]=currentTaskIdx+1;
           threadWork[3*currentThreadIdx+1]=0;
            //threadWork[3*currentThreadIdx+1]=numEvals[currentTaskIdx+1];
           threadWork[3*currentThreadIdx+2]=evalsForThread;
           }
           else
           {
             //  std::cout<<"Task: "<<currentTaskIdx<<" not looped, remaining evals: "<<remainingEvalsInTask<<std::endl;
           threadWork[3*currentThreadIdx]=currentTaskIdx;
           threadWork[3*currentThreadIdx+1]=numEvalsInCurrentTask-remainingEvalsInTask;
           //threadWork[3*currentThreadIdx+1]=remainingEvalsInTask;
           threadWork[3*currentThreadIdx+2]=evalsForThread;
           }
       }
       else //less evals remaining in current task than the current thread needs, allocate all of them to the thread and move to the next task
       {
           accumulatedEvals+=remainingEvalsInTask;
           remainingEvalsForThread-=remainingEvalsInTask;


           currentTaskIdx++;
           if(currentTaskIdx==numTasks)
           {
               break;
           }
           numEvalsInCurrentTask=numEvals[currentTaskIdx];
           remainingEvalsInTask=numEvalsInCurrentTask;
       }
  


}
}

void SetThreadWorkConst(int* threadWork,int evalsPerTask, int numTasks, int numThreads)
{
int numTasksInCurrentThread=0;
double evalsPerThread=evalsPerTask*numTasks/(1.0*numThreads);
int accumulatedEvals=0;
int currentThreadIdx=0;
int currentTaskIdx=0;
int numEvalsInCurrentTask=evalsPerTask;
int remainingEvalsInTask=numEvalsInCurrentTask;
int currentBenchmark=evalsPerThread;
int remainingEvalsForThread=currentBenchmark;
int evalsForThread=0;
threadWork[0]=0;
threadWork[1]=0;
threadWork[2]=currentBenchmark;
while(true)
{
       if(remainingEvalsInTask>=remainingEvalsForThread) //enough evals in the current task to completely occupy the current thread
       {
           accumulatedEvals+=remainingEvalsForThread;
           remainingEvalsInTask-=remainingEvalsForThread;
           currentThreadIdx++;
           if(currentThreadIdx==numThreads)
           {
               break;
           }
           currentBenchmark=(currentThreadIdx+1)*evalsPerThread;
           remainingEvalsForThread=currentBenchmark-accumulatedEvals;
           evalsForThread=remainingEvalsForThread;
           if(remainingEvalsInTask==0)
           {
               //std::cout<<"Task: "<<currentTaskIdx<<" looped"<<std::endl;
               //std::cout<<"Thread "<<currentThreadIdx<<": Task "<<currentTaskIdx+1<<", start eval "<<0<<", num evals "<<evalsForThread<<std::endl;
           threadWork[3*currentThreadIdx]=currentTaskIdx+1;
           threadWork[3*currentThreadIdx+1]=0;
            //threadWork[3*currentThreadIdx+1]=numEvals[currentTaskIdx+1];
           threadWork[3*currentThreadIdx+2]=evalsForThread;
           }
           else
           {
               //std::cout<<"Task: "<<currentTaskIdx<<" not looped, remaining evals: "<<remainingEvalsInTask<<std::endl;
               //std::cout<<"Thread "<<currentThreadIdx<<": Task "<<currentTaskIdx<<", start eval "<<numEvalsInCurrentTask-remainingEvalsInTask<<", num evals "<<evalsForThread<<std::endl;
           threadWork[3*currentThreadIdx]=currentTaskIdx;
           threadWork[3*currentThreadIdx+1]=numEvalsInCurrentTask-remainingEvalsInTask;
           //threadWork[3*currentThreadIdx+1]=remainingEvalsInTask;
           threadWork[3*currentThreadIdx+2]=evalsForThread;
           }
       }
       else //less evals remaining in current task than the current thread needs, allocate all of them to the thread and move to the next task
       {
           accumulatedEvals+=remainingEvalsInTask;
           remainingEvalsForThread-=remainingEvalsInTask;


           currentTaskIdx++;
           if(currentTaskIdx==numTasks)
           {
               break;
           }
           numEvalsInCurrentTask=evalsPerTask;
           remainingEvalsInTask=numEvalsInCurrentTask;
       }
  


}
}
