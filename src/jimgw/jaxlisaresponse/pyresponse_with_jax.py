import jax
import jax.numpy as jnp
from functools import partial



@jax.jit
def interp_single(input_in, integer_delay, fraction, A_arr, deps, E_arr, start_input_ind):
    ind = jnp.floor_divide(fraction, deps).astype(int)
    frac = (fraction - ind * deps) / deps
    A = A_arr[ind] * (1.0 - frac) + A_arr[ind + 1] * frac

    B = 1.0 - fraction
    C = fraction
    D = fraction * (1.0 - fraction)
    
    def body_fn(j, val):
        sum_val = val
        E = E_arr[j - 1]
        F = j + fraction
        G = j + (1.0 - fraction)
        temp_up = input_in[jnp.asarray(integer_delay + 1 + j - start_input_ind).astype(int)]
        temp_down = input_in[jnp.asarray(integer_delay - j - start_input_ind).astype(int)]
        sum_val += E * (temp_up / F + temp_down.real / G)
        return sum_val
    
    sum_val = jax.lax.fori_loop(1, E_arr.shape[0] + 1, body_fn, 0.0)

    temp_up = input_in[jnp.asarray(integer_delay + 1 - start_input_ind).astype(int)]
    temp_down = input_in[jnp.asarray(integer_delay - start_input_ind).astype(int)]
    result = A * (B * temp_up + C * temp_down + D * sum_val)
    
    return result

#@partial(jax.jit, static_argnums = (5)) 
@jax.jit
def interp(input_in, integer_delay, fraction, A_arr, deps, E_arr, start_input_ind):
    ind = jnp.floor_divide(fraction, deps).astype(int)
    frac = (fraction - ind * deps) / deps
    A = A_arr[ind] * (1.0 - frac) + A_arr[ind + 1] * frac

    B = 1.0 - fraction
    C = fraction
    D = fraction * (1.0 - fraction)

    def body_fn(j, sums):
        sum_hp, sum_hc = sums
        E = E_arr[j - 1]
        F = j + fraction
        G = j + (1.0 - fraction)
        temp_up = input_in[jnp.asarray(integer_delay + 1 + j - start_input_ind).astype(int)]
        temp_down = input_in[jnp.asarray(integer_delay - j - start_input_ind).astype(int)]
        sum_hp += E * (temp_up.real / F + temp_down.real / G)
        sum_hc += E * (temp_up.imag / F + temp_down.imag / G)
        return sum_hp, sum_hc

    sum_hp, sum_hc = jax.lax.fori_loop(1, E_arr.shape[0] + 1, body_fn, (0.0, 0.0))

    temp_up = input_in[jnp.asarray(integer_delay + 1 - start_input_ind).astype(int)]
    temp_down = input_in[jnp.asarray(integer_delay - start_input_ind).astype(int)]

    result_hp = A * (B * temp_up.real + C * temp_down.real + D * sum_hp)
    result_hc = A * (B * temp_up.imag + C * temp_down.imag + D * sum_hc)

    return result_hp, result_hc






'''
double *y_gw, double* t_data, double *k_in, double *u_in, double *v_in, double dt,
              int num_delays, int *link_space_craft_0_in, int *link_space_craft_1_in,
              cmplx *input_in, int num_inputs, int order, double sampling_frequency,
              int buffer_integer, double* A_in, double deps, int num_A, double* E_in, int projections_start_ind,
              double* x_in_receiver, double* x_in_emitter, double* L_in, int num_orbit_inputs
'''

#@partial(jax.jit, static_argnums = 13)
@jax.jit
def response(y_gw, t_data, k, u, v,
             link_space_craft_0, link_space_craft_1,
             input_in, buffer_integer, sampling_frequency,
             A_arr, deps, E_arr, projections_start_ind,
             x_in_receiver, x_in_emitter, L_in, num_orbit_inputs):

    num_links = link_space_craft_0.shape[0]
    num_delays = t_data.shape[0]

    def xi_projections(u0, v0, n0):
        u_dot_n = jnp.dot(u0, n0)
        v_dot_n = jnp.dot(v0, n0)
        return 0.5 * (u_dot_n**2 - v_dot_n**2), u_dot_n * v_dot_n

    def compute_delay(link_i, i):
        x0 = jnp.array([x_in_receiver[(link_i * 3 + coord) * num_orbit_inputs + i] for coord in range(3)])
        x1 = jnp.array([x_in_emitter[(link_i * 3 + coord) * num_orbit_inputs + i] for coord in range(3)])
        n = x0 - x1
        n /= jnp.linalg.norm(n)

        L = L_in[link_i * num_orbit_inputs + i]
        xi_p, xi_c = xi_projections(u, v, n)
        k_dot_n = jnp.dot(k, n)
        k_dot_x0 = jnp.dot(k, x0)
        k_dot_x1 = jnp.dot(k, x1)

        delay0 = t_data[i] - k_dot_x0 / 299792458.0
        delay1 = t_data[i] - L - k_dot_x1 / 299792458.0

        integer_delay0 = jnp.ceil(delay0 * sampling_frequency).astype(int) - 1
        fraction0 = 1.0 + integer_delay0 - delay0 * sampling_frequency

        integer_delay1 = jnp.ceil(delay1 * sampling_frequency).astype(int) - 1
        fraction1 = 1.0 + integer_delay1 - delay1 * sampling_frequency

        start_input_ind = jnp.minimum(integer_delay0, integer_delay1) - buffer_integer

        hp_del0, hc_del0 = interp(input_in, integer_delay0, fraction0, A_arr, deps, E_arr, start_input_ind)
        hp_del1, hc_del1 = interp(input_in, integer_delay1, fraction1, A_arr, deps, E_arr, start_input_ind)

        pre_factor = 1.0 / (1.0 - k_dot_n)
        large_factor = (hp_del0 - hp_del1) * xi_p + (hc_del0 - hc_del1) * xi_c

        return pre_factor * large_factor

    # Vectorize over delays for each link
    def compute_link(link_i):
        return jax.vmap(lambda i: compute_delay(link_i, i))(jnp.arange(projections_start_ind, num_delays))

    # Vectorize over all links
    updated = jax.vmap(compute_link)(jnp.arange(num_links))

    # Update y_gw from projections_start_ind onward
    y_gw = y_gw.at[:, projections_start_ind:].set(updated)
    return y_gw




'''
delayed_links, y_gw, num_inputs, num_orbit_info, delays, num_delays, dt, link_inds, tdi_signs, num_units, num_channels,
               order, sampling_frequency, buffer_integer, A_in, deps, num_A, E_in, tdi_start_ind)
'''

#@partial(jax.jit, static_argnums = (2,4) )
import jax
import jax.numpy as jnp

@jax.jit
def TDI_delay(delayed_links, input_links, delays, link_inds, tdi_signs,
              num_units, num_channels, sampling_frequency, buffer_integer,
              A_arr, deps, E_arr, tdi_start_ind):
    """
    Vectorized TDI_delay using jax.vmap for parallelism.
    
    Parameters:
    - input_links: shape (num_links, num_inputs)
    - delays: shape (num_units * num_channels, num_delays)
    - delayed_links: shape (num_units * num_channels, num_delays)
    """

    num_delays = delays.shape[1]
    num_inputs = input_links.shape[1]

    def compute_delayed(unit_i):
        link_i = link_inds[unit_i]
        sign = tdi_signs[unit_i % num_units]
        link_input = input_links[link_i]

        def compute_delay(i):
            delay = delays[unit_i, i]
            integer_delay = jnp.ceil(delay * sampling_frequency).astype(int) - 1
            fraction = 1.0 + integer_delay - delay * sampling_frequency
            start_input_ind = integer_delay - buffer_integer

            delayed_val = interp_single(link_input, integer_delay, fraction,
                                        A_arr, deps, E_arr, start_input_ind)
            return sign * delayed_val

        return jax.vmap(compute_delay)(jnp.arange(tdi_start_ind, num_delays))

    updated = jax.vmap(compute_delayed)(jnp.arange(num_units * num_channels))

    # Update delayed_links only from tdi_start_ind onward
    delayed_links = delayed_links.at[:, tdi_start_ind:].set(updated)
    return delayed_links





'''
def TDI_delay(delayed_links, input_links, delays, link_inds, tdi_signs,
              A_arr, deps, E_arr, num_units, num_channels, buffer_integer, sampling_frequency):

    num_delays = delays.shape[1]

    def unit_body(unit_i, delayed_links):
        link_i = link_inds[unit_i]
        sign = tdi_signs[unit_i % num_units]

        def delay_body(i, delayed_links):
            delay_ind = unit_i * num_delays + i
            delay = delays[delay_ind]
            integer_delay = jnp.ceil(delay * sampling_frequency).astype(int) - 1
            fraction = 1.0 + integer_delay - delay * sampling_frequency
            start_input_ind = integer_delay - buffer_integer

            # Assumes input_links[link_i] is a 1D array of waveform data
            link_input = input_links[link_i]

            link_delayed_out = interp_single(link_input, integer_delay, fraction, A_arr, deps, E_arr, start_input_ind)
            return delayed_links.at[unit_i, i].add(sign * link_delayed_out)

        return lax.fori_loop(0, num_delays, delay_body, delayed_links)

    return lax.fori_loop(0, num_units * num_channels, unit_body, delayed_links)
'''