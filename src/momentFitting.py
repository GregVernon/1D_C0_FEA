import scipy
from scipy import optimize
import sympy
import numpy
import unittest

if __name__ == "src.momentFitting":
    from src import basis
elif __name__ == "momentFitting":
    import basis

def computeQuadrature( n, generating_basis, basis_domain ):
    M = computeMomentVector( n, generating_basis, basis_domain )
    x0 = numpy.linspace( basis_domain[0], basis_domain[1], n )
    sol = scipy.optimize.least_squares( lambda x : objFun( M, generating_basis, basis_domain, x ), x0, bounds = (-1, 1), ftol = 1e-14, xtol = 1e-14, gtol = 1e-14 )
    qp = sol.x
    w = solveLinearMomentFit( M, generating_basis, basis_domain, qp )
    return qp, w 

def computeMomentVector( n, generating_basis, basis_domain ):
    moment_vector = numpy.zeros( 2*n )
    for i in range( 0, 2*n ):
        # moment_vector[i], _ = scipy.integrate.quad( lambda x: generating_basis( i, i, basis_domain, x ), basis_domain[0], basis_domain[1], epsrel = 1e-12, limit = 1000 )
        x = sympy.symbols( "x", real = True )
        basis_fun = generating_basis( i, i, basis_domain, x )
        moment_vector[i] = float( sympy.integrate( basis_fun.as_expr(), ( x, -1, 1 ) ).evalf() )
    return moment_vector

def assembleLinearMomentFitSystem( degree, generating_basis, basis_domain, pts ):
    A = numpy.zeros( shape = ( degree + 1, len( pts ) ), dtype = "double" )
    x = sympy.symbols( "x", real = True )
    for p in range( 0, degree + 1 ):
        basis_fun = generating_basis( p, p, basis_domain, x )
        for i in range( 0, len( pts ) ):
            A[p,i] = float( basis_fun( pts[i]) )
    return A

def solveLinearMomentFit( M, generating_basis, basis_domain, pts ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, generating_basis, basis_domain, pts )
    sol = scipy.optimize.lsq_linear( A, M )
    w = sol.x
    return w

def objFun( M, generating_basis, basis_domain, pts ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, generating_basis, basis_domain, pts )
    w = solveLinearMomentFit( M, generating_basis, basis_domain, pts )
    obj = ( M - A @ w )
    obj = obj.squeeze()
    return obj

class Test_computeMomentVector( unittest.TestCase ):
    def test_legendre( self ):
        for n in range( 1, 10 ):
            test_moment_vector = computeMomentVector( n, basis.symLegendreBasis, [-1, 1] )
            gold_moment_vector = numpy.concatenate( [[2.0], [0.0]*(2*n-1) ] )
            self.assertTrue( numpy.allclose( test_moment_vector, gold_moment_vector, atol = 1e-6 ) )

class Test_computeGaussLegendreQuadrature( unittest.TestCase ):
    def test_1_pt( self ):
        qp_gold = numpy.array( [ 0.0 ] )
        w_gold = numpy.array( [ 2.0 ] )
        [ qp, w ] = computeQuadrature( 1, basis.symLegendreBasis, [-1, 1] )
        self.assertAlmostEqual( first = qp, second = qp_gold, delta = 1e-12 )
        self.assertAlmostEqual( first = w, second = w_gold, delta = 1e-12 )
        
    def test_2_pt( self ):
        qp_gold = numpy.array( [ -1.0/numpy.sqrt(3), 1.0/numpy.sqrt(3) ] )
        w_gold = numpy.array( [ 1.0, 1.0 ] )
        [ qp, w ] = computeQuadrature( 2, basis.symLegendreBasis, [-1, 1] )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )
    
    def test_3_pt( self ):
        qp_gold = numpy.array( [ -1.0 * numpy.sqrt( 3.0 / 5.0 ), 
                                  0.0, 
                                 +1.0 * numpy.sqrt( 3.0 / 5.0 ) ] )
        w_gold = numpy.array( [ 5.0 / 9.0, 
                                8.0 / 9.0, 
                                5.0 / 9.0 ] )
        [ qp, w ] = computeQuadrature( 3, basis.symLegendreBasis, [-1, 1] )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )
    
    def test_4_pt( self ):
        qp_gold = numpy.array( [ -1.0 * numpy.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ),
                                 -1.0 * numpy.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ),
                                 +1.0 * numpy.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ),
                                 +1.0 * numpy.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ) ] )
        w_gold = numpy.array( [ ( 18.0 - numpy.sqrt( 30.0 ) ) / 36.0,
                                ( 18.0 + numpy.sqrt( 30.0 ) ) / 36.0,
                                ( 18.0 + numpy.sqrt( 30.0 ) ) / 36.0,
                                ( 18.0 - numpy.sqrt( 30.0 ) ) / 36.0 ] )
        [ qp, w ] = computeQuadrature( 4, basis.symLegendreBasis, [-1, 1] )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )
    
    def test_5_pt( self ):
        qp_gold = numpy.array( [ -1.0 / 3.0 * numpy.sqrt( 5.0 + 2.0 * numpy.sqrt( 10.0 / 7.0 ) ),
                                 -1.0 / 3.0 * numpy.sqrt( 5.0 - 2.0 * numpy.sqrt( 10.0 / 7.0 ) ),
                                  0.0,
                                 +1.0 / 3.0 * numpy.sqrt( 5.0 - 2.0 * numpy.sqrt( 10.0 / 7.0 ) ),
                                 +1.0 / 3.0 * numpy.sqrt( 5.0 + 2.0 * numpy.sqrt( 10.0 / 7.0 ) ) ] )
        w_gold = numpy.array( [ ( 322.0 - 13.0 * numpy.sqrt( 70.0 ) ) / 900.0,
                                ( 322.0 + 13.0 * numpy.sqrt( 70.0 ) ) / 900.0,
                                  128.0 / 225.0,
                                ( 322.0 + 13.0 * numpy.sqrt( 70.0 ) ) / 900.0,
                                ( 322.0 - 13.0 * numpy.sqrt( 70.0 ) ) / 900.0, ] )
        [ qp, w ] = computeQuadrature( 5, basis.symLegendreBasis, [-1, 1] )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )
