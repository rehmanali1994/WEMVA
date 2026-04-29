# Import Timing, Plotting, and Debugging Tools + NumPy
from HelperFunctions import * 

## Line Search Algorithm
def line_search(func, alpha_init, expansion=2.0, max_iter_bracketing=50, max_iter_golden_section=50):
    # Initial Bracketing Step
    a, b = bracket_minimum(func, alpha_init, expansion=expansion, max_iter=max_iter_bracketing)
    # Golden-Section Search Over Initial Bracket
    return golden_section_search(func, a, b, max_iter=max_iter_golden_section)

## Determine Bracket for Golden-Section Search
def bracket_minimum(phi, alpha_init, expansion=2.0, max_iter=50):
    """
    Brackets the minimum of a unimodal function phi starting from alpha=0.
    Returns interval [a, b] such that a local minimum lies between [a, b].
    Parameters:
        phi       : function to minimize (1D scalar function)
        alpha_init: initial step size guess (> 0)
        expansion : expansion factor for step size (default: 2.0)
        max_iter  : maximum iterations for bracketing (default: 50)
    Returns:
        (a, b) : tuple of floats, interval containing a local minimum
    """
    # Minimum at alpha >= 0
    alpha_prev = 0.0
    phi_prev = phi(alpha_prev)
    # Current alpha
    alpha_curr = alpha_init
    phi_curr = phi(alpha_curr)
    # Already found interval
    if phi_curr >= phi_prev:
        return alpha_prev, alpha_curr
    else: 
        # Iterate until interval found
        for _ in range(max_iter):
            # Interval expansion
            alpha_next = alpha_curr * expansion
            phi_next = phi(alpha_next)
            # We found a minimum at alpha_curr between alpha_prev and alpha_next
            if phi_curr <= phi_prev and phi_curr <= phi_next:
                return alpha_prev, alpha_next
            # Slide window forward
            alpha_prev, phi_prev = alpha_curr, phi_curr
            alpha_curr, phi_curr = alpha_next, phi_next
        # If unable to bracket a minimum (minimum might not exist!)
        pdb.set_trace(); 
        raise RuntimeError("Failed to bracket a minimum from zero")

## Golden-Section Search
def golden_section_search(func, a, b, max_iter=50):
    """
    Golden-section search to find the minimum of a unimodal function func on [a, b].
    Parameters:
        func     : function to minimize (1D scalar function)
        a, b     : interval [a, b] (must satisfy a < b)
        max_iter : maximum number of iterations (default: 100)
    Returns:
        xmin     : estimated location of the minimum
    """
    # If [a, b] ordered incorrectly
    if a > b:
        a, b = b, a
    # Golden ratio (~0.618)
    gr = (np.sqrt(5) - 1) / 2  
    # Initial sectioning
    x1 = a + (1 - gr) * (b - a)
    x2 = a + gr * (b - a)
    f1, f2 = func(x1), func(x2)
    # Iterating over Golden-Section Search
    for _ in range(max_iter):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1; 
            x1 = a + (1 - gr) * (b - a)
            f1 = func(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + gr * (b - a)
            f2 = func(x2)
    # Take midpoint of interval as minimum
    return (a + b) / 2