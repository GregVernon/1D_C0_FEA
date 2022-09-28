import sympy

def taylorExpansion( fun, a, order ):
    x = list( fun.atoms( sympy.Symbol ) )[0]
    t = 0
    for i in range( 0, order + 1 ):
        df = sympy.diff( fun, x, i )
        term = ( df.subs( x, a ) / sympy.factorial( i ) ) * ( x - a )**i
        t += term
    return t
