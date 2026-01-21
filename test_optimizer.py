from perf_takehome import KernelBuilder, do_kernel_test, Tests
from my_optimizer import OptimizedKernelBuilder
import unittest

# Patch KernelBuilder
import perf_takehome
perf_takehome.KernelBuilder = OptimizedKernelBuilder

if __name__ == "__main__":
    unittest.main()
