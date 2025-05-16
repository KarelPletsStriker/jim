#import numpy as np

import jax.numpy as np
#xp = np
gpu = False
'''try:
    import cupy as xp
    from pyresponse import get_response_wrap as get_response_wrap_gpu
    from pyresponse import get_tdi_delays_wrap as get_tdi_delays_wrap_gpu

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu = False
'''




def pointer_adjust(func):
    def func_wrapper(*args, **kwargs):
        targs = []
        for arg in args:
            if gpu:
                if isinstance(arg, cp.ndarray):
                    targs.append(arg.data.mem.ptr)
                    continue

            if isinstance(arg, np.ndarray):
                
                # this is what copilot told me to do. they told me it was unsafe and only meant for use if you need it for CUDA code or so
                device_buffer = arg.addressable_data(0)
                memory_pointer = device_buffer.unsafe_buffer_pointer()
                
                targs.append(memory_pointer)
                continue

            try:
                targs.append(arg.ptr)
                continue
            except AttributeError:
                targs.append(arg)

        return func(*targs, **kwargs)

    return func_wrapper
