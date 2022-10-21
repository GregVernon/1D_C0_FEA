import unittest
import math
import sympy
import numpy
import joblib

if __name__ == "src.quadrature":
    from src import basis
    from src import momentFitting
elif __name__ == "quadrature":
    import basis
    import momentFitting

def quad( fun, domain, num_points ):
    jacobian = ( domain[1] - domain[0] ) / ( 1 - (-1) )
    x_qp, w_qp = getGaussLegendreQuadrature( num_points )
    integral = 0.0
    for i in range( 0, len( x_qp ) ):
        integral += jacobian * ( fun( x_qp[i] ) * w_qp[i] )
    return integral

@joblib.Memory("cachedir").cache()
def getGaussLegendreQuadrature( num_points ):
    if num_points == 1:
        x = [ 0.0 ]
        w = [ 2.0 ]
    elif num_points == 2:
        x = [ -1.0 / math.sqrt(3), 
              +1.0 / math.sqrt(3) ]

        w = [ 1.0, 
              1.0  ]
    elif num_points == 3:
        x = [ -1.0 * math.sqrt( 3.0 / 5.0 ), 
               0.0, 
              +1.0 * math.sqrt( 3.0 / 5.0 ) ]

        w = [ 5.0 / 9.0, 
              8.0 / 9.0, 
              5.0 / 9.0 ]
    elif num_points == 4:
        x = [ -1.0 * math.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ),
              -1.0 * math.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ),
              +1.0 * math.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ),
              +1.0 * math.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ) ]
        
        w = [ ( 18.0 - math.sqrt( 30.0 ) ) / 36.0,
              ( 18.0 + math.sqrt( 30.0 ) ) / 36.0,
              ( 18.0 + math.sqrt( 30.0 ) ) / 36.0,
              ( 18.0 - math.sqrt( 30.0 ) ) / 36.0 ]
    elif num_points == 5:
        x = [ -1.0 / 3.0 * math.sqrt( 5.0 + 2.0 * math.sqrt( 10.0 / 7.0 ) ),
              -1.0 / 3.0 * math.sqrt( 5.0 - 2.0 * math.sqrt( 10.0 / 7.0 ) ),
               0.0,
              +1.0 / 3.0 * math.sqrt( 5.0 - 2.0 * math.sqrt( 10.0 / 7.0 ) ),
              +1.0 / 3.0 * math.sqrt( 5.0 + 2.0 * math.sqrt( 10.0 / 7.0 ) ) ]
        
        w = [ ( 322.0 - 13.0 * math.sqrt( 70.0 ) ) / 900.0,
              ( 322.0 + 13.0 * math.sqrt( 70.0 ) ) / 900.0,
                128.0 / 225.0,
              ( 322.0 + 13.0 * math.sqrt( 70.0 ) ) / 900.0,
              ( 322.0 - 13.0 * math.sqrt( 70.0 ) ) / 900.0, ]
    elif num_points > 5:
        # x, w = momentFitting.computeQuadrature( num_points, [-1, 1], basis.evalLegendreBasis1D )
        # x, w = computeGaussLegendreQuadratureRule( num_points )
        x = basis.eigenvaluesLegendreBasis( num_points )
        M = momentFitting.computeMomentVector( num_points, [ -1, 1 ], basis.evalSymLegendreBasis )
        A = momentFitting.assembleLinearMomentFitSystem( num_points, basis.evalSymLegendreBasis, x )
        w = momentFitting.solveLinearMomentFit( M, basis.evalSymLegendreBasis, x )
    else:
        raise( Exception( "num_points_MUST_BE_POSITIVE_INTEGER" ) )
    return x, w

def computeGaussLegendreQuadratureRule( num_points ):
    r = basis.rootsLegendreBasis( num_points )
    M = sympy.zeros( rows = num_points, cols = 1 )
    for row in range( 0, num_points ):
        p, x = basis.symLegendreBasis( row )
        M[ row ] = sympy.integrate( p, (x, -1, +1 ) )

    E = sympy.zeros( rows = num_points, cols = num_points )
    for row in range( 0, num_points ):
        p, x = basis.symLegendreBasis( row )
        for col in range( 0, len( r ) ):
            E[ row, col ] = p.subs( x, r[ col ] )
    w = list( E.LUsolve( M ) )
    r = [ float(val) for val in r ]
    w = [ float(val) for val in w ]
    return r, w

def getRiemannQuadrature( num_points ):
    if num_points < 1:
        raise( Exception( "num_points_MUST_BE_INTEGER_GEQ_1" ) )
    num_box_bounds = num_points + 1
    x = numpy.linspace( -1, 1, num_points + num_box_bounds )
    x_qp = x[1::2]
    w_qp = numpy.diff( x[0::2] )
    return x_qp, w_qp

def riemannQuadrature( fun, num_points ):
    x_qp, w_qp = getRiemannQuadrature( num_points = num_points )
    integral = 0.0
    for i in range( 0, num_points ):
        integral += fun( x_qp[i] ) * w_qp[i]
    return integral

def getNewtonCotesQuadrature( num_points ):
    if ( num_points < 1 ) or ( num_points > 6 ):
        raise( Exception( "num_points_MUST_BE_INTEGER_IN_[1,6]" ) )
    if num_points == 1:
        x_qp = numpy.array( [0.0] )
        w_qp = numpy.array( [2.0] )
    elif num_points == 2:
        x_qp = numpy.array( [-1.0, +1.0] )
        w_qp = numpy.array( [1.0, 1.0] )
    elif num_points == 3:
        x_qp = numpy.array( [-1.0, 0.0, +1.0] )
        w_qp = numpy.array( [1.0, 4.0, 1.0] ) / 3.0
    elif num_points == 4:
        x_qp = numpy.array( [-1.0, -1.0/3.0, +1.0/3.0, +1.0] )
        w_qp = numpy.array( [1.0, 3.0, 3.0, 1.0] ) / 4.0
    elif num_points == 5:
        x_qp = numpy.array( [-1.0, -0.5, 0.0, +0.5, +1.0] )
        w_qp = numpy.array( [7.0, 32.0, 12.0, 32.0, 7.0] ) / 45.0
    elif num_points == 6:
        x_qp = numpy.array( [-1.0, -0.6, -0.2, +0.2, +0.6, +1.0] )
        w_qp = numpy.array( [19.0, 75.0, 50.0, 50.0, 75.0, 19.0] ) / 144.0
    return x_qp, w_qp

def computeNewtonCotesQuadrature( fun, num_points ):
    x_qp, w_qp = getNewtonCotesQuadrature( num_points = num_points )
    integral = 0.0
    for i in range( 0, len( x_qp ) ):
        integral += fun( x_qp[i] ) * w_qp[i]
    return integral

class Test_computeNewtonCotesQuadrature( unittest.TestCase ):
    def test_integrate_constant_one( self ):
        constant_one = lambda x : 1 * x**0
        for degree in range( 1, 6 ):
            num_points = degree + 1
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = constant_one, num_points = num_points ), second = 2.0, delta = 1e-12 )
            
    def test_exact_poly_int( self ):
        for degree in range( 1, 6 ):
            num_points = degree + 1
            poly_fun = lambda x : ( x + 1.0 ) ** degree
            indef_int = lambda x : ( ( x + 1 ) ** ( degree + 1) ) / ( degree + 1 )
            def_int = indef_int(1.0) - indef_int(-1.0)
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = poly_fun, num_points = num_points ), second = def_int, delta = 1e-12 )

    def test_integrate_sin( self ):
        sin = lambda x : math.sin(x)
        for num_points in range( 1, 7 ):
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = sin, num_points = num_points ), second = 0.0, delta = 1e-12 )
    
    def test_integrate_cos( self ):
        cos = lambda x : math.cos(x)
        self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = cos, num_points = 6 ), second = 2*math.sin(1), delta = 1e-4 )

class Test_getNewtonCotesQuadrature( unittest.TestCase ):
    def test_incorrect_num_points( self ):
        with self.assertRaises( Exception ) as context:
            getNewtonCotesQuadrature( num_points = 0 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1,6]", str( context.exception ) )
        with self.assertRaises( Exception ) as context:
            getNewtonCotesQuadrature( num_points = 7 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1,6]", str( context.exception ) )
                
    def test_return_types( self ):
        for num_points in range( 1, 7 ):
            x, w = getNewtonCotesQuadrature( num_points = num_points )
            self.assertIsInstance( obj = x, cls = numpy.ndarray )
            self.assertIsInstance( obj = w, cls = numpy.ndarray )
            self.assertTrue( len( x ) == num_points )
            self.assertTrue( len( w ) == num_points )

class Test_computeRiemannQuadrature( unittest.TestCase ):
    def test_integrate_constant_one( self ):
        constant_one = lambda x : 1
        for num_points in range( 1, 100 ):
            self.assertAlmostEqual( first = riemannQuadrature( fun = constant_one, num_points = num_points ), second = 2.0, delta = 1e-12 )
    
    def test_integrate_linear( self ):
        linear = lambda x : x
        for num_points in range( 1, 100 ):
            self.assertAlmostEqual( first = riemannQuadrature( fun = linear, num_points = num_points ), second = 0.0, delta = 1e-12 )

    def test_integrate_quadratic( self ):
        linear = lambda x : x**2
        error = []
        for num_points in range( 1, 100 ):
            error.append( abs( (2.0 / 3.0) - riemannQuadrature( fun = linear, num_points = num_points ) ) )
        self.assertTrue( numpy.all( numpy.diff( error ) <= 0.0 ) )
    
    def test_integrate_sin( self ):
        sin = lambda x : math.sin(x)
        error = []
        for num_points in range( 1, 100 ):
            self.assertAlmostEqual( first = riemannQuadrature( fun = sin, num_points = num_points ), second = 0.0, delta = 1e-12 )
    
    def test_integrate_cos( self ):
        cos = lambda x : math.cos(x)
        error = []
        for num_points in range( 1, 100 ):
            error.append( abs( (2.0 / 3.0) - riemannQuadrature( fun = cos, num_points = num_points ) ) )
        self.assertTrue( numpy.all( numpy.diff( error ) <= 0.0 ) )

class Test_getRiemannQuadrature( unittest.TestCase ):
    def test_zero_points( self ):
        with self.assertRaises( Exception ) as context:
            getRiemannQuadrature( num_points = 0 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_GEQ_1", str( context.exception ) )
    
    def test_one_point( self ):
        x, w = getRiemannQuadrature( num_points = 1 )
        self.assertAlmostEqual( first = x, second = 0.0 )
        self.assertAlmostEqual( first = w, second = 2.0 )
        self.assertIsInstance( obj = x, cls = numpy.ndarray )
        self.assertIsInstance( obj = w, cls = numpy.ndarray )
    
    def test_two_point( self ):
        x, w = getRiemannQuadrature( num_points = 2 )
        self.assertTrue( numpy.allclose( x, [ -0.50, 0.50 ] ) )
        self.assertTrue( numpy.allclose( w, [ 1.0, 1.0 ] ) )
        self.assertIsInstance( obj = x, cls = numpy.ndarray )
        self.assertIsInstance( obj = w, cls = numpy.ndarray )

    def test_three_point( self ):
        x, w = getRiemannQuadrature( num_points = 3 )
        self.assertTrue( numpy.allclose( x, [ -2.0/3.0, 0.0, 2.0/3.0 ] ) )
        self.assertTrue( numpy.allclose( w, [ 2.0/3.0, 2.0/3.0, 2.0/3.0 ] ) )
        self.assertIsInstance( obj = x, cls = numpy.ndarray )
        self.assertIsInstance( obj = w, cls = numpy.ndarray )

    def test_many_points( self ):
        for num_points in range( 1, 100 ):
            x, w = getRiemannQuadrature( num_points = num_points )
            self.assertTrue( len( x ) == num_points )
            self.assertTrue( len( w ) == num_points )
            self.assertIsInstance( obj = x, cls = numpy.ndarray )
            self.assertIsInstance( obj = w, cls = numpy.ndarray )

class Test_getGaussLegendreQuadrature( unittest.TestCase ):
    def test_num_points_out_of_range( self ):
        with self.assertRaises( Exception ) as context:
            getGaussLegendreQuadrature( num_points = 0 )
        self.assertEqual( "num_points_MUST_BE_POSITIVE_INTEGER", str( context.exception ) )
    
    def test_num_points_non_int_float( self ):
        with self.assertRaises( Exception ) as context:
            getGaussLegendreQuadrature( num_points = 3.5 )
        self.assertEqual( "num_points_MUST_BE_POSITIVE_INTEGER", str( context.exception ) )
    
    def test_num_points_int_float( self ):
        x, w = getGaussLegendreQuadrature( num_points = 4.0 )
        self.assertIsInstance( obj = x, cls = list )
        self.assertIsInstance( obj = w, cls = list )
        self.assertTrue( len( x ) == 4 )
        self.assertTrue( len( w ) == 4 )
    
    def test_integrate_poly_0( self ):
        f = lambda xi : xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 1 )
        integral = f( x[0] ) * w[0]
        self.assertAlmostEqual( first = integral, second = 2.0, delta = 1e-12 )
    
    def test_integrate_poly_1( self ):
        f = lambda xi : xi ** 1.0 + xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 1 )
        integral = f( x[0] ) * w[0]
        self.assertAlmostEqual( first = integral, second = 2.0, delta = 1e-12 )

    def test_integrate_poly_2( self ):
        f = lambda xi : xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 2 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] 
        self.assertAlmostEqual( first = integral, second = 8.0 / 3.0, delta = 1e-12 )
    
    def test_integrate_poly_3( self ):
        f = lambda xi : xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 2 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] 
        self.assertAlmostEqual( first = integral, second = 8.0 / 3.0 , delta = 1e-12 )
    
    def test_integrate_poly_4( self ):
        f = lambda xi : xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 3 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] 
        self.assertAlmostEqual( first = integral, second = 46.0 / 15.0 , delta = 1e-12 )
    
    def test_integrate_poly_5( self ):
        f = lambda xi : xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 3 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] 
        self.assertAlmostEqual( first = integral, second = 46.0 / 15.0 , delta = 1e-12 )
    
    def test_integrate_poly_6( self ):
        f = lambda xi : xi ** 6.0 + xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 4 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2]  + f( x[3] ) * w[3]
        self.assertAlmostEqual( first = integral, second = 352.0 / 105.0 , delta = 1e-12 )
    
    def test_integrate_poly_7( self ):
        f = lambda xi : xi ** 7.0 + xi ** 6.0 + xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 4 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] + f( x[3] ) * w[3]
        self.assertAlmostEqual( first = integral, second = 352.0 / 105.0 , delta = 1e-12 )
    
    def test_integrate_poly_8( self ):
        f = lambda xi : xi ** 8.0 + xi ** 7.0 + xi ** 6.0 + xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 5 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] + f( x[3] ) * w[3] + f( x[4] ) * w[4]
        self.assertAlmostEqual( first = integral, second = 1126.0 / 315.0 , delta = 1e-12 )
    
    def test_integrate_poly_9( self ):
        f = lambda xi : xi ** 9.0 + xi ** 8.0 + xi ** 7.0 + xi ** 6.0 + xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = getGaussLegendreQuadrature( num_points = 5 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] + f( x[3] ) * w[3] + f( x[4] ) * w[4]
        self.assertAlmostEqual( first = integral, second = 1126.0 / 315.0 , delta = 1e-12 )

class Test_computeGaussLegendreQuadratureRule( unittest.TestCase ):
    def test_matches_hardcoded( self ):
        for num_points in range( 1, 5 ):
            x_computed, w_computed = computeGaussLegendreQuadratureRule( num_points = num_points )
            x_actual, w_actual = getGaussLegendreQuadrature( num_points = num_points )
            self.assertIsInstance( obj = x_computed, cls = list )
            self.assertIsInstance( obj = w_computed, cls = list )
            self.assertTrue( len( x_computed ) == len( x_actual ) )
            self.assertTrue( len( w_computed ) == len( w_actual ) )
            for i in range( 0, len( x_computed ) ):
                self.assertAlmostEqual( first = x_computed[i], second = x_actual[i], delta = 1e-12 )
                self.assertAlmostEqual( first = w_computed[i], second = w_actual[i], delta = 1e-12 )

    def test_integrate_poly_0( self ):
        f = lambda xi : xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 1 )
        integral = f( x[0] ) * w[0]
        self.assertAlmostEqual( first = integral, second = 2.0, delta = 1e-12 )
    
    def test_integrate_poly_1( self ):
        f = lambda xi : xi ** 1.0 + xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 1 )
        integral = f( x[0] ) * w[0]
        self.assertAlmostEqual( first = integral, second = 2.0, delta = 1e-12 )

    def test_integrate_poly_2( self ):
        f = lambda xi : xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 2 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] 
        self.assertAlmostEqual( first = integral, second = 8.0 / 3.0, delta = 1e-12 )
    
    def test_integrate_poly_3( self ):
        f = lambda xi : xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 2 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] 
        self.assertAlmostEqual( first = integral, second = 8.0 / 3.0 , delta = 1e-12 )
    
    def test_integrate_poly_4( self ):
        f = lambda xi : xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 3 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] 
        self.assertAlmostEqual( first = integral, second = 46.0 / 15.0 , delta = 1e-12 )
    
    def test_integrate_poly_5( self ):
        f = lambda xi : xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 3 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] 
        self.assertAlmostEqual( first = integral, second = 46.0 / 15.0 , delta = 1e-12 )
    
    def test_integrate_poly_6( self ):
        f = lambda xi : xi ** 6.0 + xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 4 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2]  + f( x[3] ) * w[3]
        self.assertAlmostEqual( first = integral, second = 352.0 / 105.0 , delta = 1e-12 )
    
    def test_integrate_poly_7( self ):
        f = lambda xi : xi ** 7.0 + xi ** 6.0 + xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 4 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] + f( x[3] ) * w[3]
        self.assertAlmostEqual( first = integral, second = 352.0 / 105.0 , delta = 1e-12 )
    
    def test_integrate_poly_8( self ):
        f = lambda xi : xi ** 8.0 + xi ** 7.0 + xi ** 6.0 + xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 5 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] + f( x[3] ) * w[3] + f( x[4] ) * w[4]
        self.assertAlmostEqual( first = integral, second = 1126.0 / 315.0 , delta = 1e-12 )
    
    def test_integrate_poly_9( self ):
        f = lambda xi : xi ** 9.0 + xi ** 8.0 + xi ** 7.0 + xi ** 6.0 + xi ** 5.0 + xi ** 4.0 + xi ** 3.0 + xi ** 2.0 + xi ** 1.0 + xi ** 0.0
        x, w = computeGaussLegendreQuadratureRule( num_points = 5 )
        integral = f( x[0] ) * w[0] + f( x[1] ) * w[1] + f( x[2] ) * w[2] + f( x[3] ) * w[3] + f( x[4] ) * w[4]
        self.assertAlmostEqual( first = integral, second = 1126.0 / 315.0 , delta = 1e-12 )
