"""
Performance optimization utilities for Philosophical Concept Map Generator.

This module provides utilities for improving application performance, including:
- Caching mechanisms
- Async processing helpers
- Memory usage optimization
- Performance monitoring
"""
import os
import time
import functools
import threading
import asyncio
import concurrent.futures
import pickle
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from logging_utils import get_logger

# Initialize logger
logger = get_logger("performance")


class MemoryCache:
    """In-memory cache for expensive function results."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items to store in cache
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached item or None if not found
        """
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store an item in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        with self._lock:
            # Check if cache is full and remove least recently used item
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._remove_oldest()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _remove_oldest(self) -> None:
        """Remove the least recently accessed item from cache."""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
    
    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self.cache)


# Global memory cache instance
memory_cache = MemoryCache()


class DiskCache:
    """Disk-based cache for expensive function results."""
    
    def __init__(self, cache_dir: str = 'cache', max_size_mb: int = 500):
        """
        Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.RLock()
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        # Use hash of key as filename to avoid invalid characters
        hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.cache")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached item or None if not found
        """
        with self._lock:
            cache_path = self._get_cache_path(key)
            if not os.path.exists(cache_path):
                return None
                
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
                return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store an item in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        with self._lock:
            # Check available space
            self._ensure_space()
            
            cache_path = self._get_cache_path(key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.warning(f"Error writing cache file: {e}")
    
    def _ensure_space(self) -> None:
        """Ensure cache stays under the maximum size by removing old files."""
        total_size = 0
        cache_files = []
        
        # Get all cache files with their sizes and last modified times
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                file_path = os.path.join(self.cache_dir, filename)
                size = os.path.getsize(file_path)
                mtime = os.path.getmtime(file_path)
                total_size += size
                cache_files.append((file_path, size, mtime))
        
        # Convert to MB
        total_size_mb = total_size / (1024 * 1024)
        
        # If we're over the limit, remove oldest files until under limit
        if total_size_mb > self.max_size_mb:
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x[2])
            
            # Remove files until we're under the limit
            for file_path, size, _ in cache_files:
                if total_size_mb <= self.max_size_mb:
                    break
                    
                try:
                    os.remove(file_path)
                    total_size_mb -= size / (1024 * 1024)
                except OSError:
                    pass
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except OSError:
                        pass


# Global disk cache instance
disk_cache = DiskCache()


def memoize(memory: bool = True, disk: bool = False, key_prefix: str = ''):
    """
    Decorator to cache function results in memory and/or on disk.
    
    Args:
        memory: Whether to cache in memory
        disk: Whether to cache on disk
        key_prefix: Prefix for cache keys
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a cache key from function name, args, and kwargs
            key_parts = [key_prefix, func.__name__]
            
            # Add args to key
            for arg in args:
                key_parts.append(str(arg))
                
            # Add kwargs to key (sorted for consistency)
            for k in sorted(kwargs.keys()):
                key_parts.append(f"{k}:{kwargs[k]}")
                
            cache_key = ":".join(key_parts)
            
            # Try memory cache first
            if memory:
                result = memory_cache.get(cache_key)
                if result is not None:
                    return result
            
            # Try disk cache next
            if disk:
                result = disk_cache.get(cache_key)
                if result is not None:
                    # Also store in memory for faster access next time
                    if memory:
                        memory_cache.set(cache_key, result)
                    return result
            
            # Cache miss, call the function
            result = func(*args, **kwargs)
            
            # Store result in caches
            if memory:
                memory_cache.set(cache_key, result)
            if disk:
                disk_cache.set(cache_key, result)
                
            return result
        return wrapper
    return decorator


class PerformanceMonitor:
    """Utility for monitoring and profiling function performance."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.timings = {}
        self.counts = {}
        self._lock = threading.RLock()
    
    def record_timing(self, func_name: str, execution_time: float) -> None:
        """
        Record function execution time.
        
        Args:
            func_name: Name of the function
            execution_time: Execution time in seconds
        """
        with self._lock:
            if func_name not in self.timings:
                self.timings[func_name] = []
                self.counts[func_name] = 0
                
            self.timings[func_name].append(execution_time)
            self.counts[func_name] += 1
    
    def get_average_timing(self, func_name: str) -> Optional[float]:
        """
        Get the average execution time for a function.
        
        Args:
            func_name: Name of the function
            
        Returns:
            Average execution time or None if no data
        """
        with self._lock:
            if func_name not in self.timings or not self.timings[func_name]:
                return None
                
            return sum(self.timings[func_name]) / len(self.timings[func_name])
    
    def get_function_stats(self, func_name: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a function.
        
        Args:
            func_name: Name of the function
            
        Returns:
            Dictionary of function statistics
        """
        with self._lock:
            if func_name not in self.timings or not self.timings[func_name]:
                return {
                    'name': func_name,
                    'count': 0,
                    'average': None,
                    'min': None,
                    'max': None,
                    'total': 0
                }
                
            times = self.timings[func_name]
            return {
                'name': func_name,
                'count': self.counts[func_name],
                'average': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'total': sum(times)
            }
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all monitored functions.
        
        Returns:
            List of function statistics dictionaries
        """
        with self._lock:
            return [self.get_function_stats(func) for func in self.timings.keys()]
    
    def clear(self) -> None:
        """Clear all performance data."""
        with self._lock:
            self.timings.clear()
            self.counts.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def profile(func):
    """
    Decorator to profile function execution time.
    
    Args:
        func: Function to profile
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        performance_monitor.record_timing(func.__name__, execution_time)
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper


async def run_in_executor(func, *args, **kwargs):
    """
    Run a blocking function in an executor pool.
    
    Args:
        func: Function to run
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: func(*args, **kwargs)
    )


def run_in_thread(func, *args, **kwargs):
    """
    Run a function in a separate thread.
    
    Args:
        func: Function to run
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        threading.Thread object
    """
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.daemon = True
    thread.start()
    return thread


def run_in_process_pool(func, *args, **kwargs):
    """
    Run a function in a process pool executor.
    
    Args:
        func: Function to run
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Future object
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        return executor.submit(func, *args, **kwargs)


def limit_memory_usage(func):
    """
    Decorator to limit memory usage of a function.
    
    This is a simple wrapper that triggers garbage collection before and after.
    For more advanced memory management, consider using libraries like memory_profiler.
    
    Args:
        func: Function to wrap
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import gc
        
        # Force garbage collection before function call
        gc.collect()
        
        result = func(*args, **kwargs)
        
        # Force garbage collection after function call
        gc.collect()
        
        return result
    return wrapper