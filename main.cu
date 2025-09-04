#include <iostream>
#include "vegas.cuh"
#include <chrono>
int main(int, char**){
    vegas_opts vopts;
    vopts.n_ev_strat=1000;
    vopts.n_ev_grid=25;
    vopts.num_blocks=170;
    vopts.num_threads=128;
    vopts.smoothing=8.0f;
    vopts.n_stratified_iters=3;
    vopts.print_output=true;
    vopts.n_grid_iters_max=15;
   vopts.beta=1.0f;
    vopts.alpha=0.5f;
    vopts.num_init_threads=128;
    constexpr int n_dim = 4;
    float x0[n_dim] = {0.0f, 0.0f,0.0f,0.0f}; // Starting point
    float x1[n_dim] = {10.0f, 10.0f,10.0f,10.0f}; // Ending point
    constexpr int n_g = 1024; // Number of grid divisions, must match vpopts.n_g
    constexpr int n_st = 32; // Number of grid divisions, must match vpopts.n_g
    vopts.n_g=n_g;
    vopts.n_st=n_st;
    vopts.n_dim=n_dim;
    vopts.x_0 = x0;
    vopts.x_1 = x1;

    vegas_environment env;

   create_vegas_environment(env,vopts);
    auto start = std::chrono::high_resolution_clock::now();
  vegas_fast<n_dim,n_g,n_st,float,func>(1.0f,env,vopts);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}
