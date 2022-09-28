import scipy
from scipy import optimize
import numpy
import unittest

if __name__ == "problems.moment_fitting":
    from problems import basis
elif __name__ == "moment_fitting":
    import basis

def computeGaussLegendreQuadrature( n ):
    M = numpy.zeros( 2*n, dtype = "double" )
    M[0] = 2.0
    x0 = numpy.linspace( -1, 1, n )
    sol = scipy.optimize.least_squares( lambda x : objFun( M, x ), x0, bounds = (-1, 1), ftol = 1e-14, xtol = 1e-14, gtol = 1e-14 )
    qp = sol.x
    w = solveLinearMomentFit( M, qp )
    return qp, w 

def assembleLinearMomentFitSystem( degree, pts ):
    A = numpy.zeros( shape = ( degree + 1, len( pts ) ), dtype = "double" )
    for p in range( 0, degree + 1 ):
        for i in range( 0, len( pts ) ):
            A[p,i] = basis.evalLegendreBasis1D( p,  pts[i] )
    return A

def solveLinearMomentFit( M, pts ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, pts )
    sol = scipy.optimize.lsq_linear( A, M )
    w = sol.x
    return w

def objFun( M, pts ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, pts )
    w = solveLinearMomentFit( M, pts )
    obj = ( M - A @ w )
    obj = obj.squeeze()
    print(A.shape, w.shape, obj.shape)
    return obj

class Test_computeGaussLegendreQuadrature( unittest.TestCase ):
    def test_1_pt( self ):
        qp_gold = numpy.array( [ 0.0 ] )
        w_gold = numpy.array( [ 2.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 1 )
        self.assertAlmostEqual( first = qp, second = qp_gold, delta = 1e-12 )
        self.assertAlmostEqual( first = w, second = w_gold, delta = 1e-12 )
        
    def test_2_pt( self ):
        qp_gold = numpy.array( [ -1.0/numpy.sqrt(3), 1.0/numpy.sqrt(3) ] )
        w_gold = numpy.array( [ 1.0, 1.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 2 )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )
    
    def test_3_pt( self ):
        qp_gold = numpy.array( [ -1.0 * numpy.sqrt( 3.0 / 5.0 ), 
                                  0.0, 
                                 +1.0 * numpy.sqrt( 3.0 / 5.0 ) ] )
        w_gold = numpy.array( [ 5.0 / 9.0, 
                                8.0 / 9.0, 
                                5.0 / 9.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 3 )
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
        [ qp, w ] = computeGaussLegendreQuadrature( 4 )
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
        [ qp, w ] = computeGaussLegendreQuadrature( 5 )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )
