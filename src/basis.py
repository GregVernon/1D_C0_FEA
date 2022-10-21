import unittest
import math
import numpy
import sympy

def affine_mapping_1D( domain, target_domain, x ):
    A = numpy.array( [ [ 1.0, domain[0] ], [ 1.0, domain[1] ] ] )
    b = numpy.array( [target_domain[0], target_domain[1] ] )
    c = numpy.linalg.solve( A, b )
    fx = c[0] + c[1] * x
    return fx

def changeOfBasis( b1, b2, x1 ):
    T = numpy.linalg.solve( b2, b1 )
    x2 = numpy.dot( T, x1 )
    return x2, T

def evalMonomialBasis1D( degree, variate ):
    return variate ** degree

def evalLagrangeBasis1D( degree, basis_idx, variate ):
    nodes = numpy.linspace( -1.0, 1.0, degree + 1 )
    basis_val = 1
    for i in range( 0, degree + 1 ):
        if ( i != basis_idx ):
            basis_val *= ( variate - nodes[i] ) / ( nodes[basis_idx] - nodes[i] )
    return basis_val

def evalBernsteinBasis1D( degree, basis_idx, variate ):
    if ( variate < 0.0 ) or ( variate > +1.0 ):
        raise Exception( "NOT_IN_DOMAIN" )
    term_1 = math.comb( degree, basis_idx )
    term_2 = variate ** basis_idx
    term_3 = ( 1.0 - variate ) ** ( degree - basis_idx )
    basis_val = term_1 * term_2 * term_3
    return basis_val

def evalBernsteinBasis1DVector( degree, variate ):
    basis_val_vector = numpy.zeros( shape = ( degree + 1, 1 ) )
    for basis_idx in range( 0, degree + 1 ):
        basis_val_vector[basis_idx] = evalBernsteinBasis1D( degree, basis_idx, variate )
    return basis_val_vector

def evalBernsteinBasisDeriv( degree, basis_idx, deriv, variate ):
    if ( variate < 0.0 ) or ( variate > +1.0 ):
        raise Exception( "NOT_IN_DOMAIN" )
    if deriv > 0:
        if basis_idx == 0:
            term_1 = evalBernsteinBasisDeriv( degree = degree - 1, basis_idx = basis_idx - 1, deriv = deriv - 1, variate = variate )
            term_2 = evalBernsteinBasisDeriv( degree = degree - 1, basis_idx = basis_idx, deriv = deriv - 1, variate = variate )
            basis_val = degree * ( term_1 - term_2 )
        else:
            term_1 = evalBernsteinBasisDeriv( degree = degree - 1, basis_idx = basis_idx - 1, deriv = deriv - 1, variate = variate )
            term_2 = evalBernsteinBasisDeriv( degree = degree - 1, basis_idx = basis_idx, deriv = deriv - 1, variate = variate )
            basis_val = degree * ( term_1 - term_2 )
    else:
        if ( basis_idx < 0 ) or ( basis_idx > degree ):
            basis_val = 0.0
        else:
            basis_val = evalBernsteinBasis1D( degree, basis_idx, variate )
    return basis_val

def evalLegendreBasis1D( degree, variate ):
    if ( degree == 0 ):
        basis_val = 1.0
    elif ( degree == 1 ):
        basis_val = variate
    else:
        i = degree - 1
        term_1 = ( ( 2 * i ) + 1 ) * variate * evalLegendreBasis1D( degree = i, variate = variate )
        term_2 = i * evalLegendreBasis1D( degree = i - 1, variate = variate )
        basis_val = ( term_1 - term_2 ) / ( i + 1 )
    return basis_val 

def symLegendreBasis( degree ):
    x = sympy.symbols( 'x', real = True )
    if degree == 0:
        p = sympy.Poly( 1, x )
    else:
        term_1 = 1.0 / ( ( 2.0 ** degree ) * sympy.factorial( degree ) )
        term_2 = ( ( x**2) - 1.0 ) ** degree 
        term_3 = sympy.diff( term_2, x, degree )
        p = term_1 * term_3
        p = sympy.poly( sympy.simplify( p ) )
    return p

def evalSymLegendreBasis( degree, variate ):
    p = symLegendreBasis( degree )
    basis_val = float( numpy.real( sympy.N( p( variate ) ) ) )
    return basis_val

def rootsLegendreBasis( degree ):
    if ( degree <= 0 ):
        raise Exception( "DEGREE_MUST_BE_NATURAL_NUMBER" )
    p, x = symLegendreBasis( degree )
    roots = sympy.roots( p, x )
    roots = list( roots.keys() )
    roots.sort()
    return roots

def eigenvaluesLegendreBasis( degree ):
    poly_fun = sympy.poly( symLegendreBasis( degree ) )
    comp_matrix = computeCompanionMatrix( poly_fun )
    eig_vals = numpy.sort( numpy.linalg.eigvals( comp_matrix ) )
    return eig_vals

def computeCompanionMatrix( poly_fun ):
    coeffs = poly_fun.all_coeffs()
    coeffs.reverse()
    coeffs = [ float( val / coeffs[-1] ) for val in coeffs ]
    coeffs = numpy.array( coeffs[0:-1] )
    comp_matrix = numpy.zeros( shape = ( len( coeffs ) , len( coeffs ) ) )
    comp_matrix[:,-1] = -1 * coeffs
    comp_matrix[1:, 0:-1] = numpy.eye( ( len( coeffs ) - 1 ) )
    return comp_matrix

class Test_affine_mapping_1D( unittest.TestCase ):
    def test_unit_to_biunit( self ):
        unit_domain = numpy.array( [ 0.0, 1.0 ] )
        biunit_domain = numpy.array( [ -1.0, 1.0 ] )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = unit_domain, target_domain = biunit_domain, x = 0.0 ), second = -1.0 )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = unit_domain, target_domain = biunit_domain, x = 0.5 ), second =  0.0 )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = unit_domain, target_domain = biunit_domain, x = 1.0 ), second = +1.0 )
    
    def test_biunit_to_unit( self ):
        unit_domain = numpy.array( [ 0.0, 1.0 ] )
        biunit_domain = numpy.array( [ -1.0, 1.0 ] )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = biunit_domain, target_domain = unit_domain, x = -1.0 ), second = 0.0 )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = biunit_domain, target_domain = unit_domain, x =  0.0 ), second = 0.5 )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = biunit_domain, target_domain = unit_domain, x = +1.0 ), second = 1.0 )
    
    def test_unit_to_biunit( self ):
        unit_domain = numpy.array( [ 0.0, 1.0 ] )
        biunit_domain = numpy.array( [ -1.0, 1.0 ] )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = unit_domain, target_domain = unit_domain, x = 0.0 ), second = 0.0 )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = unit_domain, target_domain = unit_domain, x = 0.5 ), second = 0.5 )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = unit_domain, target_domain = unit_domain, x = 1.0 ), second = 1.0 )
    
    def test_biunit_to_biunit( self ):
        unit_domain = numpy.array( [ 0.0, 1.0 ] )
        biunit_domain = numpy.array( [ -1.0, 1.0 ] )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = biunit_domain, target_domain = biunit_domain, x = -1.0 ), second = -1.0 )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = biunit_domain, target_domain = biunit_domain, x =  0.0 ), second =  0.0 )
        self.assertAlmostEqual( first = affine_mapping_1D( domain = biunit_domain, target_domain = biunit_domain, x = +1.0 ), second = +1.0 )
    
class Test_changeOfBasis( unittest.TestCase ):
    def test_standardR2BasisRotate( self ):
        b1 = numpy.eye(2)
        b2 = numpy.array([ [0, 1], [-1, 0] ] ).T
        x1 = numpy.array( [0.5, 0.5] ).T
        x2, T = changeOfBasis( b1, b2, x1 )
        v1 = b1 @ x1
        v2 = b2 @ x2
        self.assertTrue( numpy.allclose( v1, v2 ) )
    
    def test_standardR2BasisSkew( self ):
        b1 = numpy.eye(2)
        b2 = numpy.array([ [0, 1], [0.5, 0.5] ] ).T
        x1 = numpy.array( [0.5, 0.5] ).T
        x2, T = changeOfBasis( b1, b2, x1 )
        v1 = b1 @ x1
        v2 = b2 @ x2
        self.assertTrue( numpy.allclose( x2, numpy.array( [0.0, 1.0] ) ) )
        self.assertTrue( numpy.allclose( v1, v2 ) )
    
class Test_evalMonomialBasis1D( unittest.TestCase ):
    def test_basisAtBounds( self ):
        self.assertAlmostEqual( first = evalMonomialBasis1D( degree = 0, variate = 0 ), second = 1.0, delta = 1e-12 )
        for p in range( 1, 11 ):
            self.assertAlmostEqual( first = evalMonomialBasis1D( degree = p, variate = 0 ), second = 0.0, delta = 1e-12 )
            self.assertAlmostEqual( first = evalMonomialBasis1D( degree = p, variate = 1 ), second = 1.0, delta = 1e-12 )

    def test_basisAtMidpoint( self ):
        for p in range( 0, 11 ):
            self.assertAlmostEqual( first = evalMonomialBasis1D( degree = p, variate = 0.5 ), second = 1 / ( 2**p ), delta = 1e-12 )


class Test_evalBernsteinBasisDeriv( unittest.TestCase ):
    def test_outside_domain( self ):
        with self.assertRaises( Exception ) as context:
            evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, variate = -0.5 )
        self.assertEqual( "NOT_IN_DOMAIN", str( context.exception ) )
    
        with self.assertRaises( Exception ) as context:
            evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, variate = +1.5 )
        self.assertEqual( "NOT_IN_DOMAIN", str( context.exception ) )
    
    def test_constant_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, variate = 0.0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, variate = 1.0 ), second = 1.0, delta = 1e-12 )
    
    def test_constant_1st_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, variate = 0.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, variate = 1.0 ), second = 0.0, delta = 1e-12 )

    def test_constant_2nd_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, variate = 0.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, variate = 1.0 ), second = 0.0, delta = 1e-12 )

    def test_linear_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, variate = 0.0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, variate = 1.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, variate = 0.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, variate = 1.0 ), second = 1.0, delta = 1e-12 )
    
    def test_linear_at_gauss_pts( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, variate =  0.5 ), second = 0.5, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, variate =  0.5 ), second = 0.5, delta = 1e-12 )
    
    def test_quadratic_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, variate = 0.0 ), second = 1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, variate = 0.5 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, variate = 1.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, variate = 0.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, variate = 0.5 ), second = 0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, variate = 1.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, variate = 0.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, variate = 0.5 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, variate = 1.0 ), second = 1.00, delta = 1e-12 )
    
    def test_quadratic_at_gauss_pts( self ):
        x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
        x = [ affine_mapping_1D( [-1, 1], [0, 1], xi ) for xi in x ]
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, variate = x[0] ), second = 0.62200846792814620, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, variate = x[1] ), second = 0.04465819873852045, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, variate = x[0] ), second = 0.33333333333333333, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, variate = x[1] ), second = 0.33333333333333333, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, variate = x[0] ), second = 0.04465819873852045, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, variate = x[1] ), second = 0.62200846792814620, delta = 1e-12 )
    
    def test_linear_1st_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, variate = 0.0 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, variate = 1.0 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, variate = 0.0 ), second = +1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, variate = 1.0 ), second = +1.0, delta = 1e-12 )
    
    def test_linear_1st_deriv_at_gauss_pts( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, variate = 0.5 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, variate = 0.5 ), second = +1.0, delta = 1e-12 )
    
    def test_linear_2nd_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, variate = 0.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, variate = 1.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, variate = 0.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, variate = 1.0 ), second = 0, delta = 1e-12 )
    
    def test_linear_2nd_deriv_at_gauss_pts( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, variate = 0.5 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, variate = 0.5 ), second = 0, delta = 1e-12 )

    def test_quadratic_1st_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, variate = 0.0 ), second = -2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, variate = 0.5 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, variate = 1.0 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, variate = 0.0 ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, variate = 0.5 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, variate = 1.0 ), second = -2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, variate = 0.0 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, variate = 0.5 ), second =  1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, variate = 1.0 ), second = +2.0, delta = 1e-12 )
    
    def test_quadratic_1st_deriv_at_gauss_pts( self ):
        x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
        x = [ affine_mapping_1D( [-1, 1], [0, 1], xi ) for xi in x ]
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, variate = x[0] ), second = -1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, variate = x[1] ), second = -1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, variate = x[0] ), second = +2.0 / math.sqrt(3), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, variate = x[1] ), second = -2.0 / math.sqrt(3), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, variate = x[0] ), second = +1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, variate = x[1] ), second = +1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )

    def test_quadratic_2nd_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, variate = 0.0 ), second = -2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, variate = 0.5 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, variate = 1.0 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, variate = 0.0 ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, variate = 0.5 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, variate = 1.0 ), second = -2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, variate = 0.0 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, variate = 0.5 ), second =  1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, variate = 1.0 ), second = +2.0, delta = 1e-12 )
    
    def test_quadratic_2nd_deriv_at_gauss_pts( self ):
        x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
        x = [ affine_mapping_1D( [-1, 1], [0, 1], xi ) for xi in x ]
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, variate = x[0] ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, variate = x[1] ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, variate = x[0] ), second = -4.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, variate = x[1] ), second = -4.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, variate = x[0] ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, variate = x[1] ), second = +2.0, delta = 1e-12 )

class Test_evalLegendreBasis1D( unittest.TestCase ):
    def test_basisAtBounds( self ):
        for p in range( 0, 2 ):
            if ( p % 2 == 0 ):
                self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = -1 ), second = +1.0, delta = 1e-12 )
            else:
                self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = -1 ), second = -1.0, delta = 1e-12 )
            self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = +1 ), second = 1.0, delta = 1e-12 )
    
    def test_constant( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 0, variate = x ), second = 1.0, delta = 1e-12 )
    
    def test_linear( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 1, variate = x ), second = x, delta = 1e-12 )
    
    def test_quadratic_at_roots( self ):
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 2, variate = -1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 2, variate = +1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )
    
    def test_cubic_at_roots( self ):
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = -math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = +math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )


class Test_symLegendreBasis( unittest.TestCase ):
    def test_get_symbol( self ):
        p, x = symLegendreBasis( 2 )
        self.assertEqual( first = x, second = sympy.symbols( 'x' ) )
    
    def test_orthogonality( self ):
        max_degree = 4
        for degree_1 in range( 0, max_degree ):
            p1, x = symLegendreBasis( degree_1 )
            for degree_2 in range( 0, max_degree ):
                p2, _ = symLegendreBasis( degree_2 )
                if ( degree_1 == degree_2 ):
                    self.assertTrue( sympy.integrate( p1 * p2, ( x, -1, 1) ) != 0 )
                else:
                    self.assertAlmostEqual( first = sympy.integrate( p1 * p2, ( x, -1, 1) ), second = 0, delta = 1e-12 )

class Test_rootsLegendreBasis( unittest.TestCase ):  
    def test_non_natural_number( self ):
        with self.assertRaises( Exception ) as context:
            r = rootsLegendreBasis( 0 )
        self.assertEqual( "DEGREE_MUST_BE_NATURAL_NUMBER", str( context.exception ) )
    
    def test_finds_roots( self ):
        x = sympy.symbols( 'x' )
        max_degree = 10
        for degree in range( 1, max_degree ):
            r = rootsLegendreBasis( degree )
            self.assertIsInstance( r, list )
            self.assertTrue( len( r ) == degree )
        