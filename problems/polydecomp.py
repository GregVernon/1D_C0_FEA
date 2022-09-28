import sympy
import numpy

def naiveDecomp( poly_fun ):
    degree = poly_fun.degree()
    x = sympy.symbols('x')
    p_d = [ [] for p in range( 0, degree + 1 ) ]
    decomp_coeffs = [ [] for p in range( 0, degree + 1 ) ]
    recon_fun = sympy.Poly( 0 * x, x )
    residual = poly_fun - recon_fun
    for term in range( degree, -1, -1 ):
        residual_coeffs = residual.all_coeffs()
        basis = symLegendreBasis( term, x )
        basis_coeffs = basis.all_coeffs()
        if ( len(residual_coeffs) == len(basis_coeffs) ):
            decomp_coeffs[term] = residual_coeffs[0] / basis_coeffs[0]
            p_d[term] = decomp_coeffs[term] * basis
            recon_fun = recon_fun + p_d[term]
            residual = poly_fun - recon_fun
    return decomp_coeffs

def operatorDecomp( poly_fun, decomp_poly_family ):
    degree = poly_fun.degree()
    x = sympy.symbols( 'x' )
    poly_fun_coeffs = poly_fun.coeffs()[::-1]
    # M = sympy.diag( *poly_fun_coeffs ) #
    M = sympy.eye( degree + 1 ) 
    D = sympy.zeros( degree + 1 )
    decomp_poly_basis = decomp_poly_family( degree, x )
    for p in range( 0, degree + 1 ):
        coeffs = decomp_poly_basis[p].all_coeffs()
        for i in range( 0, len( coeffs ) ):
            D[p, i] = coeffs[::-1][i]
    T = M.solve( D )
    return D, T

def symMonomialBasis( degree, variate ):
    m = variate**degree
    return sympy.Poly( m, variate )

def symMonomialBasisFamily( degree, variate ):
    m = [ symMonomialBasis( p, variate ) for p in range( 0, degree + 1 ) ]
    return m

def symLagrangeBasis( degree, basis_idx, variate ):
    node = [ -x**0 + i*(x**0 - (-x**0))/(degree) for i in range(0, degree+1)]
    p = variate**0
    for j in range( 0, degree + 1 ):
        if ( j != basis_idx ):
            p *= ( variate - node[j] ) / ( node[basis_idx] - node[j])
    return sympy.Poly( p, variate )

def symLagrangeBasisFamily( degree, variate ):
    m = [ symLagrangeBasis( degree, p, variate ) for p in range( 0, degree + 1 ) ]
    return m

def symLegendreBasis( degree, variate ):
    term_1 = 1 / ( ( 2 ** degree ) * sympy.factorial( degree ) )
    term_2 = ( ( variate**2) - 1 ) ** degree 
    term_3 = sympy.diff( term_2, variate, degree )
    p = term_1 * term_3
    p = sympy.simplify( p )
    return sympy.Poly( p, variate )

def symLegendreBasisFamily( degree, variate ):
    p = [ symLegendreBasis( p, variate ) for p in range( 0, degree + 1 ) ]
    return p


x =  sympy.symbols('x')
poly_fun = sympy.Poly( 1*x**0 + 1*x**1 + 2*x**2 )
P, TP = operatorDecomp( poly_fun, symLegendreBasisFamily )
L, TL = operatorDecomp( poly_fun, symLagrangeBasisFamily )
C = sympy.Matrix( poly_fun.all_coeffs()[::-1])
p = symLegendreBasisFamily(2,x)
l = symLagrangeBasisFamily(2,x)
f = (T * C) * numpy.array( p )

fp = ( sympy.diag(*C) * P.inv() ) * numpy.array( p )
fl = ( sympy.diag(*C) * L.inv() ) * numpy.array( l )

sympy.plot( poly_fun.as_expr(), fp.sum().as_expr(), fl.sum().as_expr(), (x, -1, 1) )



