import numpy
import matplotlib.pyplot as plt
import unittest
from . import basis

def generateMesh( xmin, xmax, num_elems, degree ):
    ien_array = [ ]
    node_id = 0
    for i in range( 0, num_elems ):
        elem_nodes = [ ]
        for j in range( 0, degree + 1 ):
            if ( j > 0 ):
                node_id += 1
            elem_nodes.append( node_id )
        ien_array.append( elem_nodes )
    ien_array = numpy.array( ien_array, dtype = int )
    num_nodes = node_id + 1
    node_coords = numpy.linspace( xmin, xmax, num = num_nodes )
    return node_coords, ien_array

def plotMeshBasis( ien_array, node_coords, coeff, basisEval, color_by ):
    fig, ax = plt.subplots()
    colors = plt.cm.Dark2_r.colors
    xi = numpy.linspace( -1.0, 1.0, 100 )
    x = numpy.zeros( len( xi ) )
    y = numpy.zeros( len( xi ) )
    num_elems = ien_array.shape[0]
    for e in range( 0, num_elems ):
        elem_nodes = ien_array[e]
        degree = len( elem_nodes ) - 1
        elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
        for n in range( 0, len( elem_nodes ) ):
            curr_node = elem_nodes[n]
            for i in range( 0, len( xi ) ):
                x[i] = paramToRefCoords( xi[i], elem_domain )
                y[i] = coeff[curr_node] * basis.evalLagrangeBasis1D( degree = degree, basis_idx = n, variate = xi[i] )
            if ( color_by == "GLOBAL_ID" ):
                color_idx = curr_node % len( colors )
            elif ( color_by == "ELEMENT_ID" ):
                color_idx = e % len( colors )
            else:
                color_idx = n % len( colors )
            ax.plot(x, y, linewidth=2.0, color = colors[ color_idx ] )
    plt.show()
    return fig, ax, plt

def refToParamCoords( x_ref, reference_domain ):
    A = numpy.array( [ [ 1.0, reference_domain[0] ], [ 1.0, reference_domain[1] ] ] )
    b = numpy.array( [ -1.0, 1.0 ] )
    c = numpy.linalg.solve( A, b )
    x_ref = c[0] + c[1] * x_ref
    return x_ref

def paramToRefCoords( x_param, reference_domain ):
    A = numpy.array( [ [ 1.0, -1.0 ], [ 1.0, 1.0 ] ] )
    b = numpy.array( [ reference_domain[0], reference_domain[1] ] )
    c = numpy.linalg.solve( A, b )
    x_ref = c[0] + c[1] * x_param
    return x_ref

class Test_refToParamCoords( unittest.TestCase ):
    def test_unit_to_biunit( self ):
        unit_domain = numpy.array( [ 0.0, 1.0 ] )
        self.assertAlmostEqual( first = refToParamCoords( x_ref =  0.0, reference_domain = unit_domain ), second = -1.0 )
        self.assertAlmostEqual( first = refToParamCoords( x_ref =  0.5, reference_domain = unit_domain ), second =  0.0 )
        self.assertAlmostEqual( first = refToParamCoords( x_ref = +1.0, reference_domain = unit_domain ), second = +1.0 )
    
    def test_biunit_to_biunit( self ):
        biunit_domain = numpy.array( [ -1.0, 1.0 ] )
        self.assertAlmostEqual( first = refToParamCoords( x_ref = -1.0, reference_domain = biunit_domain ), second = -1.0 )
        self.assertAlmostEqual( first = refToParamCoords( x_ref =  0.0, reference_domain = biunit_domain ), second =  0.0 )
        self.assertAlmostEqual( first = refToParamCoords( x_ref = +1.0, reference_domain = biunit_domain ), second = +1.0 )
    
class Test_paramToRefCoords( unittest.TestCase ):
    def test_biunit_to_unit( self ):
        unit_domain = numpy.array( [ 0.0, 1.0 ] )
        self.assertAlmostEqual( first = paramToRefCoords( x_param = -1.0, reference_domain = unit_domain ), second = 0.0 )
        self.assertAlmostEqual( first = paramToRefCoords( x_param =  0.0, reference_domain = unit_domain ), second = 0.5 )
        self.assertAlmostEqual( first = paramToRefCoords( x_param = +1.0, reference_domain = unit_domain ), second = 1.0 )
    
    def test_biunit_to_biunit( self ):
        biunit_domain = numpy.array( [ -1.0, 1.0 ] )
        self.assertAlmostEqual( first = paramToRefCoords( x_param = -1.0, reference_domain = biunit_domain ), second = -1.0 )
        self.assertAlmostEqual( first = paramToRefCoords( x_param =  0.0, reference_domain = biunit_domain ), second =  0.0 )
        self.assertAlmostEqual( first = paramToRefCoords( x_param = +1.0, reference_domain = biunit_domain ), second = +1.0 )
    
class Test_generateMesh( unittest.TestCase ):
    def test_make_1_linear_elem( self ):
        gold_node_coords = numpy.array( [ 0.0, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1 ] ], dtype = int )
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, num_elems = 1, degree = 1 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )

    def test_make_1_quadratic_elem( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1, 2 ] ], dtype = int )
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, num_elems = 1, degree = 2 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )

    def test_make_2_linear_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1 ], [ 1, 2 ] ], dtype = int )
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, num_elems = 2, degree = 1 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
        
    def test_make_2_quadratic_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1, 2 ], [ 2, 3, 4 ] ], dtype = int )
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, num_elems = 2, degree = 2 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
    def test_make_4_linear_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 4 ] ], dtype = int )
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, num_elems = 4, degree = 1 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
        
    def test_make_4_quadratic_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1, 2 ], [ 2, 3, 4 ], [ 4, 5, 6 ], [ 6, 7, 8 ] ], dtype = int )
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, num_elems = 4, degree = 2 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )

class Test_plotBasisMesh( unittest.TestCase ):
    def test_3_linear_lagrange( self ):
        node_coords, ien_array = generateMesh( 0, 4, 3, 1 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = [], color_by = "GLOBAL_ID" )

    def test_3_quadratic_lagrange( self ):
        node_coords, ien_array = generateMesh( 0, 4, 3, 2 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = [], color_by = "GLOBAL_ID" )

    def test_10_linear_lagrange( self ):
        node_coords, ien_array = generateMesh( 0, 11, 10, 1 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = [], color_by = "GLOBAL_ID" )

    def test_10_quadratic_lagrange( self ):
        node_coords, ien_array = generateMesh( 0, 11, 10, 2 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = [], color_by = "GLOBAL_ID" )