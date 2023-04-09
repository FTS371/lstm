# 导入PyOpenCL库
import pyopencl as cl
import numpy as np

# 创建两个随机矩阵A和B
A = np.random.rand(1000, 1000).astype(np.float32)
B = np.random.rand(1000, 1000).astype(np.float32)

# 创建一个空矩阵C，用于存放结果
C = np.empty_like(A)

# 获取平台和设备信息
platforms = cl.get_platforms()
devices = platforms[0].get_devices()

# 创建一个上下文和一个命令队列
ctx = cl.Context([devices[0]])
queue = cl.CommandQueue(ctx)

# 创建缓冲区对象，用于在主机和设备之间传输数据
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

# 定义一个OpenCL内核函数，用于执行矩阵乘法操作
prg = cl.Program(ctx,
"""
__kernel void matmul(const int N,
                     __global float* A,
                     __global float* B,
                     __global float* C) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0;
    for (int k=0; k<N; k++) {
        sum += A[i*N + k] * B[k*N + j];
    }
    C[i*N + j] = sum;
}
""").build()

# 调用内核函数，并指定全局和局部工作组大小
N = np.int32(1000)
prg.matmul(queue, (N,N), None, N, a_buf, b_buf, c_buf)

# 将结果从设备缓冲区复制到主机内存中
cl.enqueue_copy(queue, C, c_buf)