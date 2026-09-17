"""Three Simpson squares; merge weights, never discard integral contributions."""
import numpy as np
from numerics import jnp  # enables float64 before arrays are constructed

PROBLEM = 'heat-l-shape-sine-v1'


def initial_condition(x):
    return jnp.sin(x[..., 0]) * jnp.sin(x[..., 1])


def exact_solution(time, x):
    return jnp.exp(-2*time) * initial_condition(x)


def build_quadrature(main_nodes=5):
    """main_nodes endpoints + main_nodes-1 midpoints per patch side.

    Integer coordinates guarantee exact duplicate detection. Each dictionary
    accumulates physical weights; shared nodes carry the sum of incident rules.
    Boundary keys include tangent direction so corners retain both derivatives.
    """
    if type(main_nodes) is not int or main_nodes < 2:
        raise ValueError('main_nodes must be an integer >= 2')
    m = 2*(main_nodes-1)  # intervals on a half-axis; always even
    w = np.where(np.arange(m+1) % 2, 4., 2.)
    w[[0,-1]] = 1.
    w *= (np.pi/m)/3
    bulk = {}
    for ox, oy in ((-m,-m), (-m,0), (0,-m)):
        for i in range(m+1):
            for j in range(m+1):
                key = (ox+i, oy+j)
                bulk[key] = bulk.get(key, 0.) + w[i]*w[j]
    boundary = {}
    # Eight half-edges form six polygon edges. Shared collinear endpoints merge.
    edges = [((-m,-m),(1,0)), ((0,-m),(1,0)),
             ((m,-m),(0,1)), ((0,0),(1,0)),
             ((0,0),(0,1)), ((-m,m),(1,0)),
             ((-m,-m),(0,1)), ((-m,0),(0,1))]
    for (x,y), (tx,ty) in edges:
        for i in range(m+1):
            key = (x+i*tx, y+i*ty, tx, ty)
            boundary[key] = boundary.get(key, 0.) + w[i]
    coordinates = np.array(list(bulk))
    entries = np.array(list(boundary))
    return dict(points=jnp.asarray(coordinates*np.pi/m), weights=jnp.array(list(bulk.values())),
                boundary_points=jnp.asarray(entries[:,:2]*np.pi/m),
                boundary_weights=jnp.array(list(boundary.values())),
                tangents=jnp.asarray(entries[:,2:], dtype=jnp.float64))


def quadrature_description(main_nodes=5, validation_main_nodes=17):
    def description(k):
        n = 2*k-1
        return f'{k} main ({n} incl. midpoints)/patch side; {3*n*n-2*n} bulk, {8*(n-1)+6} boundary; Δx=π/{n-1}'
    return 'Simpson 1/3, three squares: '+description(main_nodes)+'\nValidation: '+description(validation_main_nodes)
