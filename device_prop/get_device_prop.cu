#include <stdio.h> 

int main() {
  int nDevices;
 


//All CUDA C Runtime API functions have a return value which can be used to check for errors that occurr during their execution

  //cudaPeekAtLastError(): cuda maintain a single variable for error, which is updated everytime. This method will return the value of this variable
  //cudaGetLastError(): this does the same as above function do, but after fetching the value it resets it to cudaSuccess
  //cudaDeviceSynchronize(): this check the device async errors cause by the issuing command to the device from host, this can be achieved by doing folowing as well
  //    if (errAsync != cudaSuccess)
  //        printf("Async kernel error: %s\n", cudaGetErrorString(cudaGetLastError());

cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

//  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
