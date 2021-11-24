#ifndef ALEPHIUM_MINING_H
#define ALEPHIUM_MINING_H

void CL_CALLBACK worker_kernel_callback(cl_event event, cl_int status, void *data);

void CL_CALLBACK worker_write_callback(cl_event event, cl_int status, void *data)
{
    printf("==== Write is done %d\n", *(cl_int *)data);
}

void start_worker_mining(mining_worker_t *worker)
{
    reset_worker(worker);
    size_t hasher_size = sizeof(blake3_hasher);
    printf("==== context: %p %p %d %d\n", worker->context, worker->device_hasher, worker->platform_index, worker->device_index);
    cl_event write_completed;
    TRY(clEnqueueWriteBuffer(worker->queue, worker->device_hasher, CL_TRUE, 0, hasher_size, worker->hasher, 0, NULL, &write_completed));
    clSetEventCallback(write_completed, CL_COMPLETE, worker_write_callback, &worker->platform_index);
    TRY(clSetKernelArg(worker->kernel, 0, sizeof(cl_mem), &worker->device_hasher));
    printf("==== %d %d\n", CL_KERNEL_WORK_GROUP_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    printf("platform: %d, device: %d, grid: %d, block: %d\n", worker->platform_index, worker->device_index, worker->grid_size, worker->block_size);
    TRY(clEnqueueNDRangeKernel(worker->queue, worker->kernel, 1, NULL, &(worker->grid_size), &(worker->block_size), 0, NULL, NULL));

    cl_event worker_completed;
    TRY(clEnqueueReadBuffer(worker->queue, worker->device_hasher, CL_TRUE, 0, hasher_size, worker->device_hasher, 0, NULL, &worker_completed));
    TRY(clSetEventCallback(worker_completed, CL_COMPLETE, worker_kernel_callback, worker));

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
