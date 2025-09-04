# vegas_fast
Fast implementation of the VEGAS+ Monte Carlo integration algorithm


I could not find an implementation of this algorithm that provided quick Monte Carlo estimates (under the order of a second), so I created this simple program. The only other implementation I found (cuvegas) seemed to be optimized for use in python at the expense of speed. Since this algorithm has a lot of interesting applications (solving complicated integro-differential equations), I tried to speed it up as much as possible.

Some changes I made:
1: Using shared memory for storing variance accumulated during the grid adjustment phase: using several threads can lead to frequent collisions, creating severe serialization. Giving each block its own memory greatly reduces the number of collisions: saturing a RTX5090 (170 blocks, 128 threads/block) and using 25 evaluations/thread yields 124,399.8 ns/iteration without shared memory, and 46,509.3 ns/iteration with shared memory. In truth, this is a fairly trivial optimization as this phase is very quick. For integrands that take longer to converge, this may help save a few ms.

2: Seperating the grid adjustment phase and adaptive hypercube sampling: implementing the full VEGAS+ algorithm turned out to be very slow. Incrementing hypercube position requires a lot of slow integer arithmetic, which alone makes it much slower than the classic VEGAS algorithm. Writing to shared memory on top of this made it several times slower than a kernel that only adjusted the grid. Rather than adjust the grid and hypercube sampling strategy each iteration, the grid is first rapidly adjusted with a smaller number of samples until converging, then the samples are stratified. This routine could likely be sped up a tiny bit more as the stratification step more or less converges on the first iteration, but from testing it didnt make a significant enough difference for me to implement.

3: Templating kernels: using local memory and static shared memory in some kernels can lead to a significant speedup. For example, storing each x(y) in global memory can make the fill_bins kernel take >3x longer to execute. Of course shared memory could be used for this but this is unnecessary.

What I still need to do/test:
Updating the grid on the host: The update_grid kernel is one of the most time consuming operations in the algorithm. It is likely doing this on the host can trim the execution time by ~33%. I plan to create a second routine consisting of a vectorized version of this one--theres a lot of fixed overhead making this routine less than ideal for a single integral evaluation. In this case, the update_grid kernel may be faster than a host-side computation.
