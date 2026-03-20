import numpy as np

def _derivative_inplace(state, acc_func, out_k):
    """
    Zero-allocation derivative. Writes directly into out_k.
    out_k[:3] = velocity, out_k[3:] = acceleration.
    """
    # Slicing creates a view, not a copy
    r = state[:, :3]
    v = state[:, 3:]
    
    # Write velocity directly into the first 3 columns of the output buffer
    out_k[:, :3] = v
    # Calculate acceleration and write directly into the last 3 columns
    out_k[:, 3:] = acc_func(r)

def rk4_step_hardened(state, dt, acc_func, buffers):
    """
    Hardened RK4 using a pre-allocated buffer dictionary:
    buffers = {'k1': arr, 'k2': arr, 'k3': arr, 'k4': arr, 'temp': arr}
    """
    k1, k2, k3, k4, temp = buffers['k1'], buffers['k2'], buffers['k3'], buffers['k4'], buffers['temp']
    
    # Step 1: k1 = f(t, y)
    _derivative_inplace(state, acc_func, k1)
    
    # Step 2: k2 = f(t + 0.5dt, y + 0.5dt*k1)
    np.multiply(k1, 0.5 * dt, out=temp)
    np.add(state, temp, out=temp)
    _derivative_inplace(temp, acc_func, k2)
    
    # Step 3: k3 = f(t + 0.5dt, y + 0.5dt*k2)
    np.multiply(k2, 0.5 * dt, out=temp)
    np.add(state, temp, out=temp)
    _derivative_inplace(temp, acc_func, k3)
    
    # Step 4: k4 = f(t + dt, y + dt*k3)
    np.multiply(k3, dt, out=temp)
    np.add(state, temp, out=temp)
    _derivative_inplace(temp, acc_func, k4)
    
    # Final Accumulation: state = state + (dt/6)(k1 + 2k2 + 2k3 + k4)
    # Re-using k2 and k3 as intermediate accumulators to save space
    np.add(k2, k3, out=k2) 
    np.multiply(k2, 2.0, out=k2)
    np.add(k1, k4, out=k1)
    np.add(k1, k2, out=temp) # temp now holds (k1 + 2k2 + 2k3 + k4)
    np.multiply(temp, dt / 6.0, out=temp)
    np.add(state, temp, out=state) # In-place update of the state