# From the Jax MD examples and code
from typing import Callable
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import jax.numpy as np
from jax.api import jit
from jax_md import space, smap
from jax_md.util import Array, f32, f64

def ProgressIter(iter_fun, iter_len=0):
  if not iter_len:
    iter_len = len(iter_fun)
  out = display(progress(0, iter_len), display_id=True)
  for i, it in enumerate(iter_fun):
    yield it
    out.update(progress(i + 1, iter_len))

def progress(value, max):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 45%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))


def params_to_matrices(
    e_aa, e_bb, e_ab,
    s_aa, s_bb, s_ab):
    epsilon=np.array([
        [e_aa, e_ab],
        [e_ab, e_bb]], dtype=np.float32)
    sigma = np.array([
        [s_aa, s_ab],
        [s_ab, s_bb]
    ], dtype=np.float32)
    return epsilon, sigma

  
def draw_system(state, *args, **kwargs):
  return draw_lattice(state.position, *args, **kwargs)

def draw_lattice(positions, box_size=5., marker_size=100, N_A=25, ax=None):
    R_A = positions[:N_A]
    R_B = positions[N_A:]
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    ms = marker_size / box_size
    color = 'black'
    styles = (
         dict( #A
              linestyle='none', 
              markeredgewidth=3,
              marker='o', 
              markersize=ms, 
              color=color, 
         fillstyle='none'),
         dict( #B
              linestyle='none', 
              markeredgewidth=3,
              marker='o', 
              markersize=ms, 
              color=color, 
         fillstyle='full'))
    
    for R, marker_style in zip((R_A, R_B), styles):
        R = np.array(R)
        ax.plot(R[:, 0], R[:, 1], **marker_style)

    ax.axis('off')
    return ax


# https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
from matplotlib.collections import LineCollection
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:
def colorline(x, y, z=None,
              cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
              linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)


def lennard_jones(dr: Array,
                  sigma: Array=1,
                  epsilon: Array=1,
                  **unused_kwargs) -> Array:
  """Lennard-Jones interaction between particles with a minimum at sigma.

  Args:
    dr: An ndarray of shape [n, m] of pairwise distances between particles.
    sigma: Distance between particles where the energy has a minimum. Should
      either be a floating point scalar or an ndarray whose shape is [n, m].
    epsilon: Interaction energy scale. Should either be a floating point scalar
      or an ndarray whose shape is [n, m].
    unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.
  Returns:
    Matrix of energies of shape [n, m].
  """
  # TODO(kazeevn) nan avoidance and correctness check
  bad_dr = (~np.isfinite(dr)) | (dr == 0)
  dr = np.nan_to_num(dr)
  # dr == 0 routinely gets passed to the function by smap.pair
  dr = np.where(dr==0, 1, dr)
  idr = sigma / dr
  idr = idr * idr
  idr6 = idr * idr * idr
  idr12 = idr6 * idr6
  # TODO(schsam): This seems potentially dangerous. We should do ErrorChecking
  # here.
  return np.where(bad_dr, 0, f32(4) * epsilon * (idr12 - idr6))


def lennard_jones_pair_no_cutoff(
                       displacement_or_metric: space.DisplacementOrMetricFn,
                       species: Array=None,
                       sigma: Array=1.0,
                       epsilon: Array=1.0,
                       per_particle: bool=False) -> Callable[[Array], Array]:
  """Convenience wrapper to compute Lennard-Jones energy over a system."""
  sigma = np.array(sigma, dtype=f32)
  epsilon = np.array(epsilon, dtype=f32)
  return smap.pair(
    lennard_jones,
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    reduce_axis=(1,) if per_particle else None)


def normalise_phase(phase):
    return np.abs(np.pi*0.5 - phase)
