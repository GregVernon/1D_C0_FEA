import scipy
from scipy import optimize
import numpy
import unittest

if __name__ == "src.momentFitting":
    from src import basis
elif __name__ == "momentFitting":
    import basis

def computeQuadrature( n, domain, generating_basis ):
    M = computeMomentVector( n, domain, generating_basis )
    x0 = numpy.linspace( domain[0], domain[1], n )
    sol = scipy.optimize.least_squares( lambda x : objFun( M, generating_basis, x ), x0, bounds = (-1, 1), ftol = 1e-14, xtol = 1e-14, gtol = 1e-14 )
    qp = sol.x
    w = solveLinearMomentFit( M, generating_basis, qp )
    return qp, w 

def computeMomentVector( n, domain, generating_basis ):
    moment_vector = numpy.zeros( 2*n )
    for i in range( 0, 2*n ):
        moment_vector[i], _ = scipy.integrate.quad( lambda x: generating_basis( i, x ), domain[0], domain[1], epsrel = 1e-12, limit = 1000 )
    return moment_vector

def assembleLinearMomentFitSystem( degree, generating_basis, pts ):
    A = numpy.zeros( shape = ( degree + 1, len( pts ) ), dtype = "double" )
    for p in range( 0, degree + 1 ):
        for i in range( 0, len( pts ) ):
            A[p,i] = generating_basis( p,  pts[i] )
    return A

def solveLinearMomentFit( M, generating_basis, pts ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, generating_basis, pts )
    sol = scipy.optimize.lsq_linear( A, M )
    w = sol.x
    return w

def objFun( M, generating_basis, pts ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, generating_basis, pts )
    w = solveLinearMomentFit( M, generating_basis, pts )
    obj = ( M - A @ w )
    obj = obj.squeeze()
    return obj

class Test_computeGaussLegendreQuadrature( unittest.TestCase ):
    def test_1_pt( self ):
        qp_gold = numpy.array( [ 0.0 ] )
        w_gold = numpy.array( [ 2.0 ] )
        [ qp, w ] = computeQuadrature( 1, [-1, 1], basis.evalLegendreBasis1D )
        self.assertAlmostEqual( first = qp, second = qp_gold, delta = 1e-12 )
        self.assertAlmostEqual( first = w, second = w_gold, delta = 1e-12 )
        
    def test_2_pt( self ):
        qp_gold = numpy.array( [ -1.0/numpy.sqrt(3), 1.0/numpy.sqrt(3) ] )
        w_gold = numpy.array( [ 1.0, 1.0 ] )
        [ qp, w ] = computeQuadrature( 2, [-1, 1], basis.evalLegendreBasis1D )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )
    
    def test_3_pt( self ):
        qp_gold = numpy.array( [ -1.0 * numpy.sqrt( 3.0 / 5.0 ), 
                                  0.0, 
                                 +1.0 * numpy.sqrt( 3.0 / 5.0 ) ] )
        w_gold = numpy.array( [ 5.0 / 9.0, 
                                8.0 / 9.0, 
                                5.0 / 9.0 ] )
        [ qp, w ] = computeQuadrature( 3, [-1, 1], basis.evalLegendreBasis1D )
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
        [ qp, w ] = computeQuadrature( 4, [-1, 1], basis.evalLegendreBasis1D )
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
        [ qp, w ] = computeQuadrature( 5, [-1, 1], basis.evalLegendreBasis1D )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )
