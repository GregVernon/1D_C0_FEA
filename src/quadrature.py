import unittest
import math

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
    else:
        raise( Exception( "num_points_MUST_BE_INTEGER_IN_[1-5]" ) )
    return x, w

class Test_getGaussLegendreQuadrature( unittest.TestCase ):
    def test_num_points_out_of_range( self ):
        with self.assertRaises( Exception ) as context:
            getGaussLegendreQuadrature( num_points = 0 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1-5]", str( context.exception ) )
    
        with self.assertRaises( Exception ) as context:
            getGaussLegendreQuadrature( num_points = 6 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1-5]", str( context.exception ) )
    
    def test_num_points_non_int_float( self ):
        with self.assertRaises( Exception ) as context:
            getGaussLegendreQuadrature( num_points = 3.5 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1-5]", str( context.exception ) )
    
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
