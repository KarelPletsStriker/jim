import jax
from jax.scipy.special import factorial
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

half_order = 12
polyarr = jnp.arange(-half_order,+half_order+1)

@jax.jit
def denominator(i):
    
    return (-1)**(i+half_order) * factorial(half_order+i) * factorial(half_order-i)

den_array = jax.vmap(denominator)(jnp.arange(-half_order,+half_order+1))

@jax.jit
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
        y_arr.append((input_in[(i  + integer_delay)].real))# hp_arr.at[i].add(input_in[ind].real)
    
    res = jnp.sum(jnp.asarray(y_arr) * polynoms / den_array)
    
    
    return res
        
    

@jax.jit
def complex_lagrange_interp(input_in, integer_delay, e, start_input_ind):
    
    integer_delay = jnp.clip(integer_delay, start_input_ind , len(input_in)-half_order).astype(int)
    
    def lagrange_polynom(i):
        # spits out the unnormed lagrange polynomial
        # l_i(e) = (\prod_{j=-13}^{12}(e-j) ) / (e-i)
        return jnp.prod(jax.vmap(lambda k: e-k)(polyarr).at[i+half_order].set(1)) 
    
    polynoms = jax.vmap(lagrange_polynom)(polyarr).astype(jnp.float32)
    
    hp_arr = []#jnp.zeros(25)
    hc_arr = []#jnp.zeros(25)
    
    for i in range(-half_order,+half_order+1):
        
        #jnp.sum(jnp.asarray(hp_arr) * polynoms / den_array)#trunked_den)
        hp_arr.append((input_in[(i  + integer_delay).astype(int)].real))
        
        hc_arr.append((input_in[(i  + integer_delay).astype(int)].imag))
    
    #ind_down = (-12 + integer_delay - start_input_ind).astype(int)
    
    res_hp = jnp.sum(jnp.asarray(hp_arr) * polynoms / den_array)#trunked_den) #jax.vmap(lambda i: input_in[i+ind_down].real*polynoms[i]/den_array[i])(jnp.arange(25)).sum
    res_hc = jnp.sum(jnp.asarray(hc_arr) * polynoms / den_array)
    
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
    num_links, num_pts = y_gw.shape
    
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
        
        
        hp_del0, hc_del0 = complex_lagrange_interp(input_in, integer_delay0[link_i, i], fraction0[link_i, i], start_input_ind)         
        hp_del1, hc_del1 = complex_lagrange_interp(input_in, integer_delay1[link_i, i], fraction1[link_i, i], start_input_ind)

        pre_factor = 1.0 / (1.0 - k_dot_n)
        large_factor = (hp_del0 - hp_del1) * xi_p + (hc_del0 - hc_del1) * xi_c

        return pre_factor * large_factor

    # Vectorize over delays for each link
    def compute_link(link_i):
        return jax.vmap(lambda i: compute_delay(link_i, i))(jnp.arange(projections_start_ind, num_pts))

    # Vectorize over all links
    for link in range(num_links):
        updated = compute_link(link)
        y_gw = y_gw.at[link, projections_start_ind:].set(updated)
    
    # Update y_gw from projections_start_ind onward
    return y_gw




'''
delayed_links, y_gw, num_inputs, num_orbit_info, delays, num_pts, dt, link_inds, tdi_signs, num_units, num_channels,
               order, sampling_frequency, buffer_integer, A_in, deps, num_A, E_in, tdi_start_ind)
'''

@partial(jax.jit, static_argnums = (-1,) )
def TDI_delay(delayed_links, input_links, delays, link_inds, tdi_signs,
              sampling_frequency, buffer_integer,
              tdi_start_ind):
    """
    Vectorized TDI_delay using jax.vmap for parallelism.
    
    Parameters:
    - input_links: shape (num_links, num_inputs)
    - delays: shape (num_channels, num_units, num_pts)
    - delayed_links: shape (num_channels, num_pts)
    """

    num_channels, num_units, num_pts = delays.shape
    num_inputs = input_links.shape[1]
    
    #print(input_links)

    def compute_delayed(channel_i):
        def compute_unit(unit_i):
            
            link_i = link_inds[channel_i,unit_i]
            
            sign = tdi_signs[unit_i]
            
            link_input = input_links[link_i]
            delay_arr = delays[channel_i, unit_i]

            def compute_delay(i):
                delay = delay_arr[i] # i feel like there should be an extra term here
                integer_delay = jnp.floor(delay * sampling_frequency).astype(int)
                
                e = ( -integer_delay + delay * sampling_frequency)
                
                start_input_ind = integer_delay - buffer_integer

                delayed_val = lagrange_interp(link_input, integer_delay, e, start_input_ind)
                
                
                return sign * delayed_val

            return jax.vmap(compute_delay)(jnp.arange(tdi_start_ind, num_pts))

        return jax.vmap(compute_unit)(jnp.arange(num_units)).sum(axis=0)

    #updated = jax.vmap(compute_delayed)(jnp.arange(num_channels))
    #print(jnp.max(jnp.abs(updated)))
    #updated = updated
    for chan in range(num_channels):
        updated = compute_delayed(chan)
        delayed_links = delayed_links.at[chan, tdi_start_ind:].set(updated)

    return delayed_links
