/*
Vector-pool aggregation based local feature aggregation for point cloud.
PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection
https://arxiv.org/abs/2102.00463

Written by Shaoshuai Shi
All Rights Reserved 2020.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "vector_pool_gpu.h"
#include "cuda_utils.h"


__global__ void query_three_nn_by_stacked_local_idxs_kernel(
    const float *support_xyz, const float *new_xyz, const float *new_xyz_grid_centers,
    int *new_xyz_grid_idxs, float *new_xyz_grid_dist2,
    const int *stack_neighbor_idxs, const int *start_len,
    int M, int num_total_grids){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
    // new_xyz_grid_idxs: (M1 + M2 ..., num_total_grids, 3) three-nn
    // new_xyz_grid_dist2: (M1 + M2 ..., num_total_grids, 3) square of dist of three-nn
    // stack_neighbor_idxs: (max_length_of_neighbor_idxs)
    // start_len: (M1 + M2, 2)  [start_offset, neighbor_length]

    int grid_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt_idx >= M || grid_idx >= num_total_grids) return;

    new_xyz += pt_idx * 3;
    new_xyz_grid_centers += pt_idx * num_total_grids * 3 + grid_idx * 3;
    new_xyz_grid_idxs += pt_idx * num_total_grids * 3 + grid_idx * 3;
    new_xyz_grid_dist2 += pt_idx * num_total_grids * 3 + grid_idx * 3;

    start_len += pt_idx * 2;
    stack_neighbor_idxs += start_len[0];
    int neighbor_length = start_len[1];

    float center_x = new_xyz_grid_centers[0];
    float center_y = new_xyz_grid_centers[1];
    float center_z = new_xyz_grid_centers[2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = -1, besti2 = -1, besti3 = -1;
    for (int k = 0; k < neighbor_length; k++){
        int cur_neighbor_idx = stack_neighbor_idxs[k];

        float x = support_xyz[cur_neighbor_idx * 3 + 0];
        float y = support_xyz[cur_neighbor_idx * 3 + 1];
        float z = support_xyz[cur_neighbor_idx * 3 + 2];

        float d = (center_x - x) * (center_x - x) + (center_y - y) * (center_y - y) + (center_z - z) * (center_z - z);

        if (d < best1) {
            best3 = best2; besti3 = besti2;
            best2 = best1; besti2 = besti1;
            best1 = d; besti1 = cur_neighbor_idx;
        }
        else if (d < best2) {
            best3 = best2; besti3 = besti2;
            best2 = d; besti2 = cur_neighbor_idx;
        }
        else if (d < best3) {
            best3 = d; besti3 = cur_neighbor_idx;
        }
    }
    if (besti2 == -1){
        besti2 = besti1; best2 = best1;
    }
    if (besti3 == -1){
        besti3 = besti1; best3 = best1;
    }
    new_xyz_grid_dist2[0] = best1;
    new_xyz_grid_dist2[1] = best2;
    new_xyz_grid_dist2[2] = best3;
    new_xyz_grid_idxs[0] = besti1;
    new_xyz_grid_idxs[1] = besti2;
    new_xyz_grid_idxs[2] = besti3;
}


int query_three_nn_by_stacked_local_idxs_kernel_launcher_stack(
    const float *support_xyz, const float *new_xyz, const float *new_xyz_grid_centers,
    int *new_xyz_grid_idxs, float *new_xyz_grid_dist2,
    const int *stack_neighbor_idxs, const int *start_len,
    int M, int num_total_grids){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
    // new_xyz_grid_idxs: (M1 + M2 ..., num_total_grids, 3) three-nn
    // new_xyz_grid_dist2: (M1 + M2 ..., num_total_grids, 3) square of dist of three-nn
    // stack_neighbor_idxs: (max_length_of_neighbor_idxs)
    // start_len: (M1 + M2, 2)  [start_offset, neighbor_length]

    cudaError_t err;
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK), num_total_grids);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    query_three_nn_by_stacked_local_idxs_kernel<<<blocks, threads>>>(
        support_xyz, new_xyz, new_xyz_grid_centers,
        new_xyz_grid_idxs, new_xyz_grid_dist2, stack_neighbor_idxs, start_len,
        M, num_total_grids
    );

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}


__global__ void query_stacked_local_neighbor_idxs_kernel(
    const float *support_xyz, const int *xyz_batch_cnt, const float *new_xyz, const int *new_xyz_batch_cnt,
    int *stack_neighbor_idxs, int *start_len, int *cumsum, int avg_length_of_neighbor_idxs,
    float max_neighbour_distance, int batch_size, int M, int nsample, int neighbor_type){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // stack_neighbor_idxs: (max_length_of_neighbor_idxs)
    // start_len: (M1 + M2, 2)  [start_offset, neighbor_length]
    // cumsum: (1), max offset of current data in stack_neighbor_idxs
    // max_neighbour_distance: float
    // nsample: find all (-1), find limited number(>0)
    // neighbor_type: 1: ball, others: cube

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < batch_size; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

    support_xyz += xyz_batch_start_idx * 3;
    new_xyz += pt_idx * 3;
    start_len += pt_idx * 2;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx];

    float local_x, local_y, local_z;
    float radius2 = max_neighbour_distance * max_neighbour_distance;

    int temp_idxs[1000];

    int sample_cnt = 0;
    for (int k = 0; k < n; ++k) {
        local_x = support_xyz[k * 3 + 0] - new_x;
        local_y = support_xyz[k * 3 + 1] - new_y;
        local_z = support_xyz[k * 3 + 2] - new_z;

        if (neighbor_type == 1){
            // ball
            if (local_x * local_x + local_y * local_y + local_z * local_z > radius2){
                continue;
            }
        }
        else{
            // voxel
            if ((fabs(local_x) > max_neighbour_distance) |
                (fabs(local_y) > max_neighbour_distance) |
                (fabs(local_z) > max_neighbour_distance)){
                continue;
            }
        }
        if (sample_cnt < 1000){
            temp_idxs[sample_cnt] = k;
        }
        else{
            break;
        }
        sample_cnt++;
        if (nsample > 0 && sample_cnt >= nsample) break;
    }
    start_len[0] = atomicAdd(cumsum, sample_cnt);
    start_len[1] = sample_cnt;

    int max_thresh = avg_length_of_neighbor_idxs * M;
    if (start_len[0] >= max_thresh) return;

    stack_neighbor_idxs += start_len[0];
    if (start_len[0] + sample_cnt >= max_thresh) sample_cnt = max_thresh - start_len[0];

    for (int k = 0; k < sample_cnt; k++){
        stack_neighbor_idxs[k] = temp_idxs[k] + xyz_batch_start_idx;
    }
}


int query_stacked_local_neighbor_idxs_kernel_launcher_stack(
    const float *support_xyz, const int *xyz_batch_cnt, const float *new_xyz, const int *new_xyz_batch_cnt,
    int *stack_neighbor_idxs, int *start_len, int *cumsum, int avg_length_of_neighbor_idxs,
    float max_neighbour_distance, int batch_size, int M, int nsample, int neighbor_type){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // stack_neighbor_idxs: (max_length_of_neighbor_idxs)
    // start_len: (M1 + M2, 2)  [start_offset, neighbor_length]
    // cumsum: (1), max offset of current data in stack_neighbor_idxs
    // max_neighbour_distance: float
    // nsample: find all (-1), find limited number(>0)
    // neighbor_type: 1: ball, others: cube

    cudaError_t err;
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    query_stacked_local_neighbor_idxs_kernel<<<blocks, threads>>>(
        support_xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
        stack_neighbor_idxs, start_len, cumsum, avg_length_of_neighbor_idxs,
        max_neighbour_distance, batch_size, M, nsample, neighbor_type
    );

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}


__global__ void vector_pool_kernel_stack(
    const float *support_xyz, const float *support_features, const int *xyz_batch_cnt,
    const float *new_xyz, float *new_features, float *new_local_xyz, const int *new_xyz_batch_cnt,
    int num_grid_x, int num_grid_y, int num_grid_z, float max_neighbour_distance,
    int batch_size, int M, int num_c_in, int num_c_out,
    int num_c_each_grid, int num_total_grids, int *point_cnt_of_grid, int *grouped_idxs,
    int use_xyz, float grid_size_x, float grid_size_y,
    float grid_size_z, int *cum_sum, int num_max_sum_points, int nsample, int neighbor_type, int pooling_type){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // support_features: (N1 + N2 ..., C)
    // xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_features: (M1 + M2 ..., C), C = num_total_grids * num_c_each_grid
    // new_local_xyz: (M1 + M2 ..., 3 * num_total_grids)
    // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // num_grid_x, num_grid_y, num_grid_z: number of grids in each local area centered at new_xyz
    // point_cnt_of_grid: (M1 + M2 ..., num_total_grids)
    // grouped_idxs: (num_max_sum_points, 3)[idx of support_xyz, idx of new_xyz, idx of grid_idx in new_xyz]
    // use_xyz: whether to calculate new_local_xyz
    // neighbor_type: 1: ball, others: cube
    // pooling_type: 0: avg_pool, 1: random choice

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < batch_size; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

    support_xyz += xyz_batch_start_idx * 3;
    support_features += xyz_batch_start_idx * num_c_in;

    new_xyz += pt_idx * 3;
    new_features += pt_idx * num_c_out;
    point_cnt_of_grid += pt_idx * num_total_grids;
    new_local_xyz += pt_idx * 3 * num_total_grids;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx], grid_idx_x, grid_idx_y, grid_idx_z, grid_idx;
    float local_x, local_y, local_z;
    float radius2 = max_neighbour_distance * max_neighbour_distance;

    int sample_cnt = 0;
    for (int k = 0; k < n; ++k) {
        local_x = support_xyz[k * 3 + 0] - new_x;
        local_y = support_xyz[k * 3 + 1] - new_y;
        local_z = support_xyz[k * 3 + 2] - new_z;

        if (neighbor_type == 1){
            // ball
            if (local_x * local_x + local_y * local_y + local_z * local_z > radius2){
                continue;
            }
        }
        else{
            // voxel
            if ((fabs(local_x) > max_neighbour_distance) |
                (fabs(local_y) > max_neighbour_distance) |
                (fabs(local_z) > max_neighbour_distance)){
                continue;
            }
        }

        grid_idx_x = floorf((local_x + max_neighbour_distance) / grid_size_x);
        grid_idx_y = floorf((local_y + max_neighbour_distance) / grid_size_y);
        grid_idx_z = floorf((local_z + max_neighbour_distance) / grid_size_z);
        grid_idx = grid_idx_x * num_grid_y * num_grid_z + grid_idx_y * num_grid_z + grid_idx_z;
        grid_idx = min(max(grid_idx, 0), num_total_grids - 1);

        if (pooling_type == 0){
            // avg pooling
            point_cnt_of_grid[grid_idx] ++;

            for (int i = 0; i < num_c_in; i++){
                new_features[grid_idx * num_c_each_grid + i % num_c_each_grid] += support_features[k * num_c_in + i];
            }
            if (use_xyz){
                new_local_xyz[grid_idx * 3 + 0] += local_x;
                new_local_xyz[grid_idx * 3 + 1] += local_y;
                new_local_xyz[grid_idx * 3 + 2] += local_z;
            }

            int cnt = atomicAdd(cum_sum, 1);
            if (cnt >= num_max_sum_points) continue;  // continue to statistics the max number of points

            grouped_idxs[cnt * 3 + 0] = xyz_batch_start_idx + k;
            grouped_idxs[cnt * 3 + 1] = pt_idx;
            grouped_idxs[cnt * 3 + 2] = grid_idx;

            sample_cnt++;
            if(nsample > 0 && sample_cnt >= nsample) break;
        }
        else if (pooling_type == 1){
            // random choose one within sub-voxel
            // printf("new_xyz=(%.2f, %.2f, %.2f, ), find neighbor k=%d: support_xyz=(%.2f, %.2f, %.2f), local_xyz=(%.2f, %.2f, %.2f), neighbor=%.2f, grid_idx=%d, point_cnt_of_grid_idx=%d\n",
            // new_x, new_y, new_z, k, support_xyz[k * 3 + 0], support_xyz[k * 3 + 1], support_xyz[k * 3 + 2], local_x, local_y, local_z, max_neighbour_distance, grid_idx, point_cnt_of_grid[grid_idx]);

            if (point_cnt_of_grid[grid_idx] == 0){
                point_cnt_of_grid[grid_idx] ++;
                for (int i = 0; i < num_c_in; i++){
                    new_features[grid_idx * num_c_each_grid + i % num_c_each_grid] = support_features[k * num_c_in + i];
                }
                if (use_xyz){
                    new_local_xyz[grid_idx * 3 + 0] = local_x;
                    new_local_xyz[grid_idx * 3 + 1] = local_y;
                    new_local_xyz[grid_idx * 3 + 2] = local_z;
                }

                int cnt = atomicAdd(cum_sum, 1);
                if (cnt >= num_max_sum_points) continue;  // continue to statistics the max number of points

                grouped_idxs[cnt * 3 + 0] = xyz_batch_start_idx + k;
                grouped_idxs[cnt * 3 + 1] = pt_idx;
                grouped_idxs[cnt * 3 + 2] = grid_idx;

                sample_cnt++;
                if(nsample > 0 && sample_cnt >= nsample || sample_cnt >= num_total_grids) break;
            }

        }

    }
}


int vector_pool_kernel_launcher_stack(
    const float *support_xyz, const float *support_features, const int *xyz_batch_cnt,
    const float *new_xyz, float *new_features, float *new_local_xyz, const int *new_xyz_batch_cnt,
    int *point_cnt_of_grid, int *grouped_idxs,
    int num_grid_x, int num_grid_y, int num_grid_z, float max_neighbour_distance,
    int batch_size, int N, int M, int num_c_in, int num_c_out, int num_total_grids,
    int use_xyz, int num_max_sum_points, int nsample, int neighbor_type, int pooling_type){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // support_features: (N1 + N2 ..., C)
    // xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_features: (M1 + M2 ..., C)
    // new_local_xyz: (M1 + M2 ..., 3)
    // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // num_grid_x, num_grid_y, num_grid_z: number of grids in each local area centered at new_xyz
    // use_xyz: whether to calculate new_local_xyz
    // grouped_idxs: (num_max_sum_points, 3)[idx of support_xyz, idx of new_xyz, idx of grid_idx in new_xyz]
    // neighbor_type: 1: ball, others: cube
    // pooling_type: 0: avg_pool, 1: random choice


    cudaError_t err;
    int num_c_each_grid = num_c_out / num_total_grids;
    float grid_size_x = max_neighbour_distance * 2 / num_grid_x;
    float grid_size_y = max_neighbour_distance * 2 / num_grid_y;
    float grid_size_z = max_neighbour_distance * 2 / num_grid_z;

    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    int cum_sum = 0;
    int *p_cum_sum;
    cudaMalloc((void**)&p_cum_sum, sizeof(int));
    cudaMemcpy(p_cum_sum, &cum_sum, sizeof(int), cudaMemcpyHostToDevice);

    vector_pool_kernel_stack<<<blocks, threads>>>(
        support_xyz, support_features, xyz_batch_cnt,
        new_xyz, new_features, new_local_xyz, new_xyz_batch_cnt,
        num_grid_x, num_grid_y, num_grid_z, max_neighbour_distance,
        batch_size, M, num_c_in, num_c_out,
        num_c_each_grid, num_total_grids, point_cnt_of_grid, grouped_idxs,
        use_xyz, grid_size_x, grid_size_y, grid_size_z, p_cum_sum, num_max_sum_points,
        nsample, neighbor_type, pooling_type
    );

    cudaMemcpy(&cum_sum, p_cum_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return cum_sum;
}


__global__ void vector_pool_grad_kernel_stack(const float *grad_new_features,
    const int *point_cnt_of_grid, const int *grouped_idxs,
    float *grad_support_features, int N, int M, int num_c_out, int num_c_in,
    int num_c_each_grid, int num_total_grids, int num_max_sum_points){
    // grad_new_features: (M1 + M2 ..., C_out)
    // point_cnt_of_grid: (M1 + M2 ..., num_total_grids)
    // grouped_idxs: (num_max_sum_points, 3) [idx of support_xyz, idx of new_xyz, idx of grid_idx in new_xyz]
    // grad_support_features: (N1 + N2 ..., C_in)

    int channel_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= num_max_sum_points || channel_idx >= num_c_in) return;

    int idx_of_support_xyz = grouped_idxs[index * 3 + 0];
    int idx_of_new_xyz = grouped_idxs[index * 3 + 1];
    int idx_of_grid_idx = grouped_idxs[index * 3 + 2];

    int num_total_pts = point_cnt_of_grid[idx_of_new_xyz * num_total_grids + idx_of_grid_idx];
    grad_support_features += idx_of_support_xyz * num_c_in + channel_idx;

    grad_new_features += idx_of_new_xyz * num_c_out + idx_of_grid_idx * num_c_each_grid;
    int channel_idx_of_cin = channel_idx % num_c_each_grid;
    float cur_grad = 1 / fmaxf(float(num_total_pts), 1.0);
    atomicAdd(grad_support_features, grad_new_features[channel_idx_of_cin] * cur_grad);
}


void vector_pool_grad_kernel_launcher_stack(
    const float *grad_new_features, const int *point_cnt_of_grid, const int *grouped_idxs,
    float *grad_support_features, int N, int M, int num_c_out, int num_c_in, int num_total_grids,
    int num_max_sum_points){
    // grad_new_features: (M1 + M2 ..., C_out)
    // point_cnt_of_grid: (M1 + M2 ..., num_total_grids)
    // grouped_idxs: (num_max_sum_points, 3) [idx of support_xyz, idx of new_xyz, idx of grid_idx in new_xyz]
    // grad_support_features: (N1 + N2 ..., C_in)
    int num_c_each_grid = num_c_out / num_total_grids;

    cudaError_t err;

    dim3 blocks(DIVUP(num_max_sum_points, THREADS_PER_BLOCK), num_c_in);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    vector_pool_grad_kernel_stack<<<blocks, threads>>>(
        grad_new_features, point_cnt_of_grid, grouped_idxs, grad_support_features,
        N, M, num_c_out, num_c_in, num_c_each_grid, num_total_grids, num_max_sum_points
    );
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}