import unittest
import math
import sympy

def evalBernsteinBasis( degree, basis_idx, deriv, variate ):
    if ( variate < -1.0 ) or ( variate > +1.0 ):
        raise Exception( "NOT_IN_DOMAIN" )
    if deriv > 0:
        if basis_idx == 0:
            term_1 = evalBernsteinBasis( degree = degree - 1, basis_idx = basis_idx - 1, deriv = deriv - 1, variate = variate )
            term_2 = evalBernsteinBasis( degree = degree - 1, basis_idx = basis_idx, deriv = deriv - 1, variate = variate )
            basis_val = (1/2) * degree * ( term_1 - term_2 )
        else:
            term_1 = evalBernsteinBasis( degree = degree - 1, basis_idx = basis_idx - 1, deriv = deriv - 1, variate = variate )
            term_2 = evalBernsteinBasis( degree = degree - 1, basis_idx = basis_idx, deriv = deriv - 1, variate = variate )
            basis_val = (1/2) * degree * ( term_1 - term_2 )
    else:
        if ( basis_idx < 0 ) or ( basis_idx > degree ):
            basis_val = 0.0
        else:
            param_variate = (1/2) * ( variate + 1.0 )
            term_1 = math.comb( degree, basis_idx )
            term_2 = param_variate ** basis_idx
            term_3 = ( 1.0 - param_variate ) ** ( degree - basis_idx )
            basis_val = term_1 * term_2 * term_3
    return basis_val

def symLegendreBasis( degree ):
    x = sympy.symbols( 'x' )
    term_1 = 1.0 / ( ( 2.0 ** degree ) * sympy.factorial( degree ) )
    term_2 = ( ( x**2) - 1.0 ) ** degree 
    term_3 = sympy.diff( term_2, x, degree )
    p = term_1 * term_3
    p = sympy.simplify( p )
    return p, x
class Test_evalBernsteinBasis( unittest.TestCase ):
    def test_outside_domain( self ):
        with self.assertRaises( Exception ) as context:
            evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 0, variate = -1.5 )
        self.assertEqual( "NOT_IN_DOMAIN", str( context.exception ) )
    
        with self.assertRaises( Exception ) as context:
            evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 0, variate = +1.5 )
        self.assertEqual( "NOT_IN_DOMAIN", str( context.exception ) )
    
    def test_constant_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 0, basis_idx = 0, deriv = 0, variate = -1.0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 0, basis_idx = 0, deriv = 0, variate = +1.0 ), second = 1.0, delta = 1e-12 )
    
    def test_constant_1st_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 0, basis_idx = 0, deriv = 1, variate = -1.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 0, basis_idx = 0, deriv = 1, variate = +1.0 ), second = 0.0, delta = 1e-12 )

    def test_constant_2nd_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 0, basis_idx = 0, deriv = 2, variate = -1.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 0, basis_idx = 0, deriv = 2, variate = +1.0 ), second = 0.0, delta = 1e-12 )

    def test_linear_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 0, variate = -1.0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 0, variate = +1.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 0, variate = -1.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 0, variate = +1.0 ), second = 1.0, delta = 1e-12 )
    
    def test_linear_at_gauss_pts( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 0, variate =  0.0 ), second = 0.5, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 0, variate =  0.0 ), second = 0.5, delta = 1e-12 )
    
    def test_quadratic_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 0, variate = -1.0 ), second = 1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 0, variate =  0.0 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 0, variate = +1.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 0, variate = -1.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 0, variate =  0.0 ), second = 0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 0, variate = +1.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 0, variate = -1.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 0, variate =  0.0 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 0, variate = +1.0 ), second = 1.00, delta = 1e-12 )
    
    def test_quadratic_at_gauss_pts( self ):
        x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 0, variate = x[0] ), second = 0.62200846792814620, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 0, variate = x[1] ), second = 0.04465819873852045, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 0, variate = x[0] ), second = 0.33333333333333333, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 0, variate = x[1] ), second = 0.33333333333333333, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 0, variate = x[0] ), second = 0.04465819873852045, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 0, variate = x[1] ), second = 0.62200846792814620, delta = 1e-12 )
    
    def test_linear_1st_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 1, variate = -1.0 ), second = -1/2, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 1, variate = +1.0 ), second = -1/2, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 1, variate = -1.0 ), second = +1/2, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 1, variate = +1.0 ), second = +1/2, delta = 1e-12 )
    
    def test_linear_1st_deriv_at_gauss_pts( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 1, variate = 0.0 ), second = -1/2, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 1, variate = 0.0 ), second = +1/2, delta = 1e-12 )
    
    def test_linear_2nd_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 2, variate = -1.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 2, variate = +1.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 2, variate = -1.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 2, variate = +1.0 ), second = 0, delta = 1e-12 )
    
    def test_linear_2nd_deriv_at_gauss_pts( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 0, deriv = 2, variate = 0.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 1, basis_idx = 1, deriv = 2, variate = 0.0 ), second = 0, delta = 1e-12 )

    def test_quadratic_1st_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 1, variate = -1.0 ), second = -1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 1, variate =  0.0 ), second = -0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 1, variate = +1.0 ), second =  0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 1, variate = -1.0 ), second = +1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 1, variate =  0.0 ), second =  0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 1, variate = +1.0 ), second = -1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 1, variate = -1.0 ), second =  0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 1, variate =  0.0 ), second =  0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 1, variate = +1.0 ), second = +1.00, delta = 1e-12 )
    
    def test_quadratic_1st_deriv_at_gauss_pts( self ):
        x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 1, variate = x[0] ), second = -1/2 - 1/( 2*math.sqrt(3) ), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 1, variate = x[1] ), second = -1/2 + 1/( 2*math.sqrt(3) ), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 1, variate = x[0] ), second = +1 / math.sqrt(3), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 1, variate = x[1] ), second = -1 / math.sqrt(3), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 1, variate = x[0] ), second = +1/2 - 1/( 2*math.sqrt(3) ), delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 1, variate = x[1] ), second = +1/2 + 1/( 2*math.sqrt(3) ), delta = 1e-12 )

    def test_quadratic_2nd_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 1, variate = -1.0 ), second = -1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 1, variate =  0.0 ), second = -0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 1, variate = +1.0 ), second =  0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 1, variate = -1.0 ), second = +1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 1, variate =  0.0 ), second =  0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 1, variate = +1.0 ), second = -1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 1, variate = -1.0 ), second =  0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 1, variate =  0.0 ), second =  0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 1, variate = +1.0 ), second = +1.00, delta = 1e-12 )
    
    def test_quadratic_2nd_deriv_at_gauss_pts( self ):
        x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 2, variate = x[0] ), second = +0.5, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 0, deriv = 2, variate = x[1] ), second = +0.5, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 2, variate = x[0] ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 1, deriv = 2, variate = x[1] ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 2, variate = x[0] ), second = +0.5, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis( degree = 2, basis_idx = 2, deriv = 2, variate = x[1] ), second = +0.5, delta = 1e-12 )
    
class Test_symLegendreBasis( unittest.TestCase ):
    def test_get_symbol( self ):
        p, x = symLegendreBasis( 2 )
        self.assertEqual( first = x, second = sympy.symbols( 'x' ) )
    
    def test_orthogonality( self ):
        x = sympy.symbols( 'x' )
        max_degree = 4
        for degree_1 in range( 0, max_degree ):
            p1 = symLegendreBasis( degree_1 )
            for degree_2 in range( 0, max_degree ):
                p2 = symLegendreBasis( degree_2 )
                if ( degree_1 == degree_2 ):
                    self.assertTrue( sympy.integrate( p1 * p2, ( x, -1, 1) ) != 0 )
                else:
                    self.assertAlmostEqual( first = sympy.integrate( p1 * p2, ( x, -1, 1) ), second = 0, delta = 1e-12 )
