from perf_takehome import KernelBuilder
from my_optimizer import OptimizedKernelBuilder

# Patch Perf Takehome
KernelBuilder.build_kernel = OptimizedKernelBuilder.build_kernel
KernelBuilder.__init__ = OptimizedKernelBuilder.__init__
KernelBuilder.debug_info = OptimizedKernelBuilder.debug_info

# Patch Submission Tests
try:
    import tests.submission_tests as submission_tests
    submission_tests.KernelBuilder = KernelBuilder
except ImportError:
    pass
try:
    import submission_tests
    submission_tests.KernelBuilder = KernelBuilder
except ImportError:
    pass
