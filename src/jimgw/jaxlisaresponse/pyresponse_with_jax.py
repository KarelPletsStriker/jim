import jax
from jax.scipy.special import factorial
import jax.numpy as jnp
from functools import partial

#jax.config.update("jax_enable_x64", False)

# half_order * 2 + 1 is the polynomial order for your lagrange interpolation
half_order = 12
polyarr = jnp.arange(-half_order,+half_order+1)

# this variable determines how large each chunk of your computations will be. 
# this is important so that you don't demand too much memory
#block_size = 100000

@jax.jit
def denominator(i):
    
    return (-1)**(i+half_order) * factorial(half_order+i) * factorial(half_order-i)

den_array = jax.vmap(denominator)(jnp.arange(-half_order,+half_order+1))


@jax.checkpoint#jit
def lagrange_interp(input_in, integer_delay, e, start_input_ind):
    
    integer_delay = jnp.clip(integer_delay, start_input_ind , len(input_in)-half_order).astype(int)
    
    def lagrange_polynom(i):
        # spits out the unnormed lagrange polynomial
        # l_i(e) = (\prod_{j=-13}^{12}(e-j) ) / (e-i)
        return jnp.prod(jax.vmap(lambda k: e-k)(polyarr).at[i+half_order].set(1)) 
    
    polynoms = jax.vmap(lagrange_polynom)(polyarr).astype(jnp.float32)
    
    y_arr = []#jnp.zeros(25)
    
    for i in range(-half_order,half_order+1):
        
        #jnp.sum(jnp.asarray(hp_arr) * polynoms / den_array)#trunked_den)
        y_arr.append((input_in[(i  + integer_delay)].real)* polynoms[i+half_order] / den_array[i+half_order])# hp_arr.at[i].add(input_in[ind].real)
    
    res = jnp.sum(jnp.asarray(y_arr) )
    
    
    return res
        
    

@jax.checkpoint#jit
def complex_lagrange_interp(input_in, integer_delay, e, start_input_ind):
    
    integer_delay = jnp.clip(integer_delay, start_input_ind , len(input_in)-half_order).astype(int)
    
    def lagrange_polynom(i):
        # spits out the unnormed lagrange polynomial
        # l_i(e) = (\prod_{j=-13}^{12}(e-j) ) / (e-i) / (weights)
        return jnp.prod(jax.vmap(lambda k: e-k)(polyarr).at[i+half_order].set(1)) / den_array[i+half_order]
    
    #polynoms = jax.vmap(lagrange_polynom)(polyarr).astype(jnp.float32)
    
    #hp_arr = []#jnp.zeros(25)
    #hc_arr = []#jnp.zeros(25)
    
    res_hp = 0.0
    res_hc = 0.0
    
    for i in range(-half_order,+half_order+1):
        
        polynom = lagrange_polynom(i) 
        #jnp.sum(jnp.asarray(hp_arr) * polynoms / den_array)#trunked_den)
        res_hp += (input_in[(i  + integer_delay).astype(int)].real) * polynom
        
        res_hc += (input_in[(i  + integer_delay).astype(int)].imag) * polynom
    
    return res_hp, res_hc



#@jax.jit
@partial(jax.jit, static_argnums = -5)
def response(y_gw, t_data, k, u, v,
             link_space_craft_0, link_space_craft_1,
             input_in, sampling_frequency, buffer_integer,
             projections_start_ind,
             x_in_receiver, x_in_emitter, L_in, num_orbit_inputs):
    
    '''
    This function computes the 6 single-link response functions for a given waveform's output
    
    Parameters:
    y_gw  - empty array on which the new data will be inserted, has shape (num_links, num_pts)
    k,u,v - 3D base vectors in , denotes GW's direction (k is direction of propagation) 
    
    '''
    num_links, num_blocks, block_size = y_gw.shape
    num_pts = num_blocks * block_size
    
    k_dot_x0 = jnp.einsum('i,jik->jk', k, x_in_receiver)
    k_dot_x1 = jnp.einsum('i,jik->jk', k, x_in_emitter)

    delay0 = - k_dot_x0 / 299792458.0
    delay0 = delay0.at[:].add(t_data)
    delay1 = - L_in - k_dot_x1 / 299792458.0
    delay1 = delay1.at[:].add(t_data)
    
    

    integer_delay0 = jnp.floor(delay0 * sampling_frequency).astype(int)
    fraction0 = -integer_delay0 + delay0 * sampling_frequency

    integer_delay1 = jnp.floor(delay1 * sampling_frequency).astype(int) 
    fraction1 = - integer_delay1 + delay1 * sampling_frequency
    
    def xi_projections(u0, v0, n0):
        u_dot_n = jnp.dot(u0, n0)
        v_dot_n = jnp.dot(v0, n0)
        return 0.5 * (u_dot_n**2 - v_dot_n**2), u_dot_n * v_dot_n

    #@jax.checkpoint
    def compute_delay(link_i, i):
        
        # (time-dependent) coordinate location of the receiving spacecraft
        #x0 = jnp.zeros(3)
        #x1 = jnp.zeros(3)
        
        x0 = (x_in_receiver[link_i, :, i]) # x0.at[:].add
        # (time-dependent) coordinate location of the emitting spacecraft
        x1 = (x_in_emitter[link_i, :, i]) # x1.at[:].add
        
        
        # normed vector of their direction
        n = x0 - x1
        n /= jnp.linalg.norm(n)
        
        
        # armlength at approximate time
        L = L_in[link_i , i] #8.339518627606857#
        
        # GW projection coefficients
        xi_p, xi_c = xi_projections(u, v, n)
        
        k_dot_n = jnp.dot(k, n)
        

        start_input_ind = buffer_integer
        
        interp = jax.jit(complex_lagrange_interp)
        
        hp_del0, hc_del0 = interp(input_in, integer_delay0[link_i, i], fraction0[link_i, i], start_input_ind)
        hp_del1, hc_del1 = interp(input_in, integer_delay1[link_i, i], fraction1[link_i, i], start_input_ind)

        pre_factor = 1.0 / (1.0 - k_dot_n)
        large_factor = (hp_del0 - hp_del1) * xi_p + (hc_del0 - hc_del1) * xi_c
        
        
        return (pre_factor * large_factor)

    # Vectorize over delays for each link
    '''def compute_link(i):
            return jax.vmap(lambda link_i: compute_delay(link_i, i))(jnp.arange(num_links))

    # Vectorize over all blocks
    updated =[]
    for idx in range(projections_start_ind, num_pts, block_size):
        idx_max = min((idx+block_size, num_pts))
        updated.append(jax.vmap(compute_link)(jnp.arange(idx, idx_max)))
        
    y_gw = y_gw.at[:, projections_start_ind:].set(jnp.concatenate(updated).T)'''
    
    
    def compute_link(y,i):
        def compute_block(k):
            response = jax.vmap(lambda link_i: compute_delay(link_i, k))(jnp.arange(num_links))
            return response
        res = jax.vmap(compute_block)(i*block_size + jnp.arange(block_size))
        y = y.at[:,i].set(res.T)
        
        return y, None
    
    #y_gw, _ = jax.lax.scan(scan_step, y_gw, , block_size))
    
    y_gw, _ = jax.lax.scan(compute_link, y_gw, jnp.arange(projections_start_ind // block_size, num_blocks))

    # Update y_gw from projections_start_ind onward
    return y_gw.reshape(num_links, num_pts)




@partial(jax.jit, static_argnums = (-1,) )
def TDI_delay(delayed_links, input_links, delays, link_inds, tdi_signs,
              sampling_frequency, buffer_integer,
              tdi_start_ind):
    """
    Vectorized TDI_delay using jax.vmap for parallelism.
    
    Parameters:
    - input_links: shape (num_links, num_inputs)
    - delays: shape (num_channels, num_units, num_pts)
    - delayed_links: shape (num_channels, num_pts // block_size, block_size)
    """

    num_channels, num_units, num_pts = delays.shape
    num_inputs = input_links.shape[1]
    block_size = delayed_links.shape[-1]
    #print(input_links)

    def compute_delayed(channel_i, unit_i, i):
        link_i = link_inds[channel_i,unit_i]
            
        sign = tdi_signs[unit_i]
            
        link_input = input_links[link_i]
        delay_arr = delays[channel_i, unit_i]
        
        delay = delay_arr[i] # i feel like there should be an extra term here
        integer_delay = jnp.floor(delay * sampling_frequency).astype(int)
                
        e = ( -integer_delay + delay * sampling_frequency)
                
        start_input_ind = integer_delay - buffer_integer

        delayed_val = lagrange_interp(link_input, integer_delay, e, start_input_ind)
                
                
        return sign * delayed_val

    
    def scan_step(y,block_i):
        
        def compute_units(channel_i,k):
            return jax.vmap(lambda unit_i: compute_delayed(channel_i,unit_i,k))(jnp.arange(num_units)).sum(axis=0)
        def compute_channel(k):
            return jax.vmap(lambda channel_i: compute_units(channel_i, k))(jnp.arange(num_channels))
        
        y = y.at[:,block_i].set(
            jax.vmap(compute_channel)(block_i * block_size + jnp.arange(block_size)).T
        )
        
        
        return y, None
    

    #updated = jax.vmap(compute_delayed)(jnp.arange(num_channels))
    #print(jnp.max(jnp.abs(updated)))
    #
    
    delayed_links, _ = jax.lax.scan(scan_step, delayed_links, jnp.arange(tdi_start_ind // block_size, num_pts//block_size))

    return delayed_links.reshape(num_channels, num_pts)
