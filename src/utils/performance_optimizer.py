import time
import numpy as np
import threading
from queue import Queue
import multiprocessing
from functools import lru_cache

class PerformanceOptimizer:
    """
    A class for optimizing the performance of the dance evaluation system.
    """
    
    def __init__(self):
        """
        Initialize the performance optimizer.
        """
        self.profiling_data = {}
        self.start_times = {}
        self.frame_times = []
        self.max_frame_times = 100  # Keep track of the last 100 frame times
        
    def start_timer(self, operation_name):
        """
        Start a timer for an operation.
        
        Args:
            operation_name: The name of the operation.
        """
        self.start_times[operation_name] = time.time()
    
    def stop_timer(self, operation_name):
        """
        Stop a timer for an operation and record the elapsed time.
        
        Args:
            operation_name: The name of the operation.
            
        Returns:
            elapsed_time: The elapsed time in seconds.
        """
        if operation_name not in self.start_times:
            return 0.0
            
        elapsed_time = time.time() - self.start_times[operation_name]
        
        if operation_name not in self.profiling_data:
            self.profiling_data[operation_name] = []
            
        self.profiling_data[operation_name].append(elapsed_time)
        
        # Keep only the last 100 measurements
        if len(self.profiling_data[operation_name]) > 100:
            self.profiling_data[operation_name] = self.profiling_data[operation_name][-100:]
            
        return elapsed_time
        
    def get_timer_duration(self, operation_name):
        """
        Get the duration of the last timer for an operation.
        
        Args:
            operation_name: The name of the operation.
            
        Returns:
            duration: The duration of the timer in seconds.
        """
        if operation_name not in self.start_times:
            return 0.0
            
        return time.time() - self.start_times[operation_name]
    
    def record_frame_time(self, frame_time):
        """
        Record the time taken to process a frame.
        
        Args:
            frame_time: The time taken to process a frame in seconds.
        """
        self.frame_times.append(frame_time)
        
        # Keep only the last max_frame_times measurements
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times = self.frame_times[-self.max_frame_times:]
    
    def get_average_frame_time(self):
        """
        Get the average time taken to process a frame.
        
        Returns:
            average_frame_time: The average time taken to process a frame in seconds.
        """
        if not self.frame_times:
            return 0.0
            
        return sum(self.frame_times) / len(self.frame_times)
    
    def get_fps(self):
        """
        Get the frames per second.
        
        Returns:
            fps: The frames per second.
        """
        avg_frame_time = self.get_average_frame_time()
        if avg_frame_time == 0:
            return 0.0
            
        return 1.0 / avg_frame_time
    
    def get_operation_stats(self, operation_name):
        """
        Get statistics for an operation.
        
        Args:
            operation_name: The name of the operation.
            
        Returns:
            stats: A dictionary of statistics for the operation.
        """
        if operation_name not in self.profiling_data or not self.profiling_data[operation_name]:
            return {
                'min': 0.0,
                'max': 0.0,
                'avg': 0.0,
                'median': 0.0,
                'std': 0.0,
                'count': 0
            }
            
        times = self.profiling_data[operation_name]
        
        return {
            'min': min(times),
            'max': max(times),
            'avg': sum(times) / len(times),
            'median': sorted(times)[len(times) // 2],
            'std': np.std(times),
            'count': len(times)
        }
    
    def get_all_stats(self):
        """
        Get statistics for all operations.
        
        Returns:
            all_stats: A dictionary mapping operation names to their statistics.
        """
        all_stats = {}
        
        for operation_name in self.profiling_data:
            all_stats[operation_name] = self.get_operation_stats(operation_name)
            
        return all_stats
    
    def print_stats(self):
        """
        Print statistics for all operations.
        """
        all_stats = self.get_all_stats()
        
        print("Performance Statistics:")
        print(f"FPS: {self.get_fps():.2f}")
        print()
        
        for operation_name, stats in all_stats.items():
            print(f"{operation_name}:")
            print(f"  Min: {stats['min']:.6f} s")
            print(f"  Max: {stats['max']:.6f} s")
            print(f"  Avg: {stats['avg']:.6f} s")
            print(f"  Median: {stats['median']:.6f} s")
            print(f"  Std: {stats['std']:.6f} s")
            print(f"  Count: {stats['count']}")
            print()
    
    def identify_bottlenecks(self, threshold=0.1):
        """
        Identify operations that are taking too long.
        
        Args:
            threshold: The threshold for considering an operation a bottleneck (in seconds).
            
        Returns:
            bottlenecks: A list of tuples (operation_name, avg_time) for operations that are taking too long.
        """
        bottlenecks = []
        
        for operation_name in self.profiling_data:
            stats = self.get_operation_stats(operation_name)
            if stats['avg'] > threshold:
                bottlenecks.append((operation_name, stats['avg']))
                
        # Sort bottlenecks by average time (descending)
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        return bottlenecks
    
    def clear_data(self):
        """
        Clear all profiling data.
        """
        self.profiling_data = {}
        self.start_times = {}
        self.frame_times = []


class ThreadPool:
    """
    A simple thread pool for parallel processing.
    """
    
    def __init__(self, num_threads=None):
        """
        Initialize the thread pool.
        
        Args:
            num_threads: The number of threads in the pool. If None, use the number of CPU cores.
        """
        self.num_threads = num_threads or multiprocessing.cpu_count()
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.threads = []
        self.running = False
        
    def start(self):
        """
        Start the thread pool.
        """
        if self.running:
            return
            
        self.running = True
        
        # Create and start the worker threads
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self._worker)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
    
    def stop(self):
        """
        Stop the thread pool.
        """
        if not self.running:
            return
            
        self.running = False
        
        # Add None tasks to signal the workers to exit
        for _ in range(self.num_threads):
            self.task_queue.put(None)
            
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join()
            
        self.threads = []
    
    def _worker(self):
        """
        Worker function for the thread pool.
        """
        while self.running:
            # Get a task from the queue
            task = self.task_queue.get()
            
            # Exit if the task is None
            if task is None:
                break
                
            # Unpack the task
            task_id, func, args, kwargs = task
            
            try:
                # Execute the task
                result = func(*args, **kwargs)
                
                # Put the result in the result queue
                self.result_queue.put((task_id, result, None))
            except Exception as e:
                # Put the exception in the result queue
                self.result_queue.put((task_id, None, e))
    
    def submit(self, func, *args, **kwargs):
        """
        Submit a task to the thread pool.
        
        Args:
            func: The function to execute.
            *args: The positional arguments for the function.
            **kwargs: The keyword arguments for the function.
            
        Returns:
            task_id: The ID of the submitted task.
        """
        if not self.running:
            self.start()
            
        # Generate a task ID
        task_id = id(func) + id(args) + id(frozenset(kwargs.items()))
        
        # Put the task in the queue
        self.task_queue.put((task_id, func, args, kwargs))
        
        return task_id
    
    def get_result(self, task_id=None, block=True, timeout=None):
        """
        Get the result of a task.
        
        Args:
            task_id: The ID of the task. If None, get any result.
            block: Whether to block until a result is available.
            timeout: The timeout for blocking.
            
        Returns:
            result: The result of the task, or None if no result is available.
            exception: The exception raised by the task, or None if no exception was raised.
        """
        try:
            # Get a result from the queue
            result_task_id, result, exception = self.result_queue.get(block=block, timeout=timeout)
            
            # Check if the result is for the requested task
            if task_id is not None and result_task_id != task_id:
                # Put the result back in the queue
                self.result_queue.put((result_task_id, result, exception))
                return None, None
                
            return result, exception
        except:
            return None, None


class FrameSkipper:
    """
    A class for skipping frames to maintain real-time performance.
    """
    
    def __init__(self, target_fps=30, max_skip=2):
        """
        Initialize the frame skipper.
        
        Args:
            target_fps: The target frames per second.
            max_skip: The maximum number of frames to skip.
        """
        self.target_fps = target_fps
        self.max_skip = max_skip
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.skip_count = 0
        
    def should_process_frame(self):
        """
        Determine whether to process the current frame.
        
        Returns:
            should_process: Whether to process the current frame.
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_frame_time
        
        # Calculate the target frame interval
        target_interval = 1.0 / self.target_fps
        
        # Increment the frame count
        self.frame_count += 1
        
        # If we're processing fast enough, don't skip frames
        if elapsed_time <= target_interval:
            self.skip_count = 0
            self.last_frame_time = current_time
            return True
            
        # If we've skipped too many frames, process this one
        if self.skip_count >= self.max_skip:
            self.skip_count = 0
            self.last_frame_time = current_time
            return True
            
        # Skip this frame
        self.skip_count += 1
        return False
    
    def get_effective_fps(self):
        """
        Get the effective frames per second.
        
        Returns:
            effective_fps: The effective frames per second.
        """
        if self.frame_count == 0:
            return 0.0
            
        elapsed_time = time.time() - self.last_frame_time
        if elapsed_time == 0:
            return 0.0
            
        return self.frame_count / elapsed_time
    
    def reset(self):
        """
        Reset the frame skipper.
        """
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.skip_count = 0


@lru_cache(maxsize=1024)
def cached_euclidean_distance(a, b):
    """
    Calculate the Euclidean distance between two points with caching.
    
    Args:
        a: The first point.
        b: The second point.
        
    Returns:
        distance: The Euclidean distance between the points.
    """
    return np.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)))


def downsample_features(features, factor=2):
    """
    Downsample features to reduce computation.
    
    Args:
        features: The features to downsample.
        factor: The downsampling factor.
        
    Returns:
        downsampled_features: The downsampled features.
    """
    downsampled_features = {}
    
    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict):
            # Handle dictionary features (e.g., joint_positions)
            downsampled_dict = {}
            for key, values in feature_data.items():
                downsampled_values = values[::factor]
                downsampled_dict[key] = downsampled_values
            
            downsampled_features[feature_name] = downsampled_dict
        else:
            # Handle list features (e.g., center_of_mass)
            downsampled_values = feature_data[::factor]
            downsampled_features[feature_name] = downsampled_values
    
    return downsampled_features


def parallel_dtw(reference_features, learner_features, distance_func, radius, thread_pool=None):
    """
    Perform DTW comparison in parallel.
    
    Args:
        reference_features: The reference movement features.
        learner_features: The learner movement features.
        distance_func: The distance function to use.
        radius: The radius constraint for FastDTW.
        thread_pool: The thread pool to use for parallel processing.
        
    Returns:
        distance: The DTW distance between the features.
        path: The optimal warping path.
    """
    from fastdtw import fastdtw
    
    # Flatten and normalize features
    reference_flat = _flatten_features(reference_features)
    learner_flat = _flatten_features(learner_features)
    
    # Ensure both sequences have the same number of features
    min_features = min(reference_flat.shape[1], learner_flat.shape[1])
    reference_flat = reference_flat[:, :min_features]
    learner_flat = learner_flat[:, :min_features]
    
    # If no thread pool is provided, perform sequential DTW
    if thread_pool is None:
        return fastdtw(reference_flat, learner_flat, dist=distance_func, radius=radius)
    
    # Split the sequences into chunks for parallel processing
    chunk_size = max(1, len(reference_flat) // thread_pool.num_threads)
    reference_chunks = [reference_flat[i:i+chunk_size] for i in range(0, len(reference_flat), chunk_size)]
    learner_chunks = [learner_flat[i:i+chunk_size] for i in range(0, len(learner_flat), chunk_size)]
    
    # Submit DTW tasks for each chunk
    task_ids = []
    for ref_chunk, learner_chunk in zip(reference_chunks, learner_chunks):
        task_id = thread_pool.submit(fastdtw, ref_chunk, learner_chunk, dist=distance_func, radius=radius)
        task_ids.append(task_id)
    
    # Collect the results
    distances = []
    paths = []
    for task_id in task_ids:
        result, exception = thread_pool.get_result(task_id)
        if exception is not None:
            raise exception
        
        distance, path = result
        distances.append(distance)
        paths.append(path)
    
    # Combine the results
    total_distance = sum(distances)
    
    # Combine the paths (simplified)
    combined_path = []
    offset = 0
    for path in paths:
        for i, j in path:
            combined_path.append((i + offset, j + offset))
        offset += chunk_size
    
    return total_distance, combined_path


def _flatten_features(features):
    """
    Flatten features into a 2D numpy array.
    
    Args:
        features: A dictionary of extracted features.
        
    Returns:
        flat_features: A 2D numpy array of flattened features.
    """
    flat_features = []
    
    # Get the number of frames
    num_frames = 0
    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict):
            # Handle dictionary features (e.g., joint_positions)
            for key, values in feature_data.items():
                num_frames = max(num_frames, len(values))
        else:
            # Handle list features (e.g., center_of_mass)
            num_frames = max(num_frames, len(feature_data))
    
    # Flatten features for each frame
    for i in range(num_frames):
        frame_features = []
        
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, dict):
                # Handle dictionary features (e.g., joint_positions)
                for key, values in feature_data.items():
                    if i < len(values):
                        # Flatten the values for this frame
                        if isinstance(values[i], list):
                            frame_features.extend(values[i])
                        else:
                            frame_features.append(values[i])
            else:
                # Handle list features (e.g., center_of_mass)
                if i < len(feature_data):
                    # Flatten the values for this frame
                    if isinstance(feature_data[i], list):
                        frame_features.extend(feature_data[i])
                    else:
                        frame_features.append(feature_data[i])
        
        flat_features.append(frame_features)
    
    return np.array(flat_features)
