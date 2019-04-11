import os
import multiprocessing
import tensorflow as tf

from absl import logging


def setup_gpu_threadpool(num_gpus):
    cpu_count = multiprocessing.cpu_count()
    logging.info('Logical CPU cores: %d' % cpu_count)

    # Sets up thread pool for each GPU.
    per_gpu_thread_count = 1
    total_gpu_thread_count = per_gpu_thread_count * num_gpus
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    logging.info('TF_GPU_THREAD_COUNT: %s' % os.environ['TF_GPU_THREAD_COUNT'])
    logging.info('TF_GPU_THREAD_MODE: %s' % os.environ['TF_GPU_THREAD_MODE'])

    # Reduces general thread pool by number of threads used for GPU pool.
    main_thread_count = cpu_count - total_gpu_thread_count
    inter_op_parallelism_threads = main_thread_count

    # Sets thread count for tf.data. Logicalcores minus threads assigned
    # to private GPU pool along with 2 threads per GPU for event monitoring
    # and sending / receiving tensors.
    num_monitoring_threads = 2 * num_gpus
    dataset_workers = (
        cpu_count - total_gpu_thread_count - num_monitoring_threads)

    return inter_op_parallelism_threads, dataset_workers
