#ifndef ALEPHIUM_MINING_H
#define ALEPHIUM_MINING_H

// void worker_stream_callback(cudaStream_t stream, cudaError_t status, void *data);

void start_worker_mining(mining_worker_t *worker)
{
    reset_worker(worker);
    TRY(clEnqueueWriteBuffer(worker->queue, worker->device_hasher, CL_TRUE, 0, sizeof(blake3_hasher), worker->hasher, 0, NULL, NULL));
    TRY(clSetKernelArg(worker->kernel, 0, sizeof(cl_mem), &worker->device_hasher));
    printf("==== %d %d\n", CL_KERNEL_WORK_GROUP_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    printf("platform: %d, device: %d, grid: %d, block: %d\n", worker->platform_index, worker->device_index, worker->grid_size, worker->block_size);
    TRY(clEnqueueNDRangeKernel(worker->queue, worker->kernel, 1, NULL, &(worker->grid_size), &(worker->block_size), 0, NULL, NULL));

    // TRY( cudaEventRecord(startEvent, worker->stream) );
    // // blake3_hasher_mine<<<worker->grid_size, worker->block_size, 0, worker->stream>>>(worker->device_hasher);
    // blake3_hasher_mine<<<worker->grid_size, worker->block_size, 0, worker->stream>>>(worker->device_hasher);
    // TRY( cudaEventRecord(stopEvent, worker->stream) );

    // TRY( cudaMemcpyAsync(worker->hasher, worker->device_hasher, sizeof(blake3_hasher), cudaMemcpyDeviceToHost, worker->stream) );
    // TRY( cudaStreamAddCallback(worker->stream, worker_stream_callback, worker, 0) );

    // float time;
    // TRY( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    // printf(" === mining time: %f\n", time);
}

#endif // ALEPHIUM_MINING_H
