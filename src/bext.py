import json
import numpy
import matplotlib
import matplotlib.pyplot as plt
import unittest

if __name__ == "src.bext":
    from src import basis
elif __name__ == "bext":
    import basis

#matplotlib.use('TkAgg')

def readBEXT( filename ):
    f = open( filename, "r" )
    uspline = json.load( f )
    f.close()
    return uspline

def getNumElems( uspline ):
    return uspline["num_elems"]

def getNumVertices( uspline ):
    return uspline["num_vertices"]

def getNumNodes( uspline ):
    return getSplineNodes( uspline ).shape[0]

def getNumBezierNodes( uspline ):
    num_elems = getNumElems( uspline )
    num_bez_nodes = 0
    for elem_idx in range( 0, num_elems ):
        elem_id = elemIdFromElemIdx( uspline, elem_idx )
        elem_degree = getElementDegree( uspline, elem_id )
        num_bez_nodes += elem_degree + 1
    return num_bez_nodes

def getDomain( uspline ):
    nodes = getSplineNodes( uspline )
    return [ min( nodes[:,0] ), max( nodes[:,0] ) ]

def elemIdFromElemIdx( uspline, elem_idx ):
    element_blocks = uspline["elements"]["element_blocks"]
    elem_id = element_blocks[ elem_idx ]["us_cid"]
    return elem_id

def elemIdxFromElemId( uspline, elem_id ):
    element_blocks = uspline["elements"]["element_blocks"]
    for elem_idx in range( 0, len( element_blocks ) ):
        if element_blocks[ elem_idx ]["us_cid"] == elem_id:
            return elem_idx

def getElementDegree( uspline, elem_id ):
    elem_idx = elemIdxFromElemId( uspline, elem_id )
    return len( uspline["elements"]["element_blocks"][elem_idx]["node_ids"] ) - 1

def getElementDomain( uspline, elem_id ):
    elem_bezier_nodes = getElementBezierNodes( uspline, elem_id )
    elem_domain = [ min( elem_bezier_nodes[:,0] ), max( elem_bezier_nodes[:,0] ) ]
    return elem_domain

def getElementNodeIds( uspline, elem_id ):
    elem_idx = elemIdxFromElemId( uspline, elem_id )
    elem_node_ids = numpy.array( uspline["elements"]["element_blocks"][elem_idx]["node_ids"] )
    return elem_node_ids

def getElementBezierNodeIds( uspline, elem_id ):
    num_elems = getNumElems( uspline )
    num_bez_nodes = 0
    for elem_idx in range( 0, num_elems ):
        curr_elem_id = elemIdFromElemIdx( uspline, elem_idx )
        elem_degree = getElementDegree( uspline, curr_elem_id )
        if elem_id == curr_elem_id:
            elem_bez_node_ids = list( range( num_bez_nodes, num_bez_nodes + elem_degree + 1 ) )
            return elem_bez_node_ids
        num_bez_nodes += elem_degree + 1

def getElementNodes( uspline, elem_id ):
    elem_node_ids = getElementNodeIds( uspline, elem_id )
    spline_nodes = getSplineNodes( uspline )
    elem_nodes = spline_nodes[elem_node_ids, 0:-1]
    return elem_nodes

def getSplineNodes( uspline ):
    return numpy.array( uspline["nodes"] )

def getCoefficientVectors( uspline ):
    coeff_vectors_list = uspline["coefficients"]["dense_coefficient_vectors"]
    coeff_vectors = {}
    for i in range( 0, len( coeff_vectors_list ) ):
        coeff_vectors[i] = coeff_vectors_list[i]["components"]
    return coeff_vectors

def getElementCoefficientVectorIds( uspline, elem_id ):
    elem_idx = elemIdxFromElemId( uspline, elem_id )
    return uspline["elements"]["element_blocks"][elem_idx]["coeff_vector_ids"]

def getVertexConnectivity( uspline ):
    return uspline["vertex_connectivity"]

def getElementExtractionOperator( uspline, elem_id ):
    coeff_vectors = getCoefficientVectors( uspline )    
    coeff_vector_ids = getElementCoefficientVectorIds( uspline, elem_id )
    C = numpy.zeros( shape = (len( coeff_vector_ids ), len( coeff_vector_ids ) ), dtype = "double" )
    for n in range( 0, len( coeff_vector_ids ) ):
        C[n,:] = coeff_vectors[ coeff_vector_ids[n] ]
    return C

def getGlobalExtractionOperator( uspline ):
    num_elems = getNumElems( uspline )
    num_nodes = getNumNodes( uspline )
    num_bez_nodes = getNumBezierNodes( uspline )
    glob_extraction_operator = numpy.zeros( shape = (num_nodes, num_bez_nodes ) )
    for elem_idx in range( 0, num_elems ):
        elem_id = elemIdFromElemIdx( uspline, elem_idx )
        elem_node_ids = getElementNodeIds( uspline, elem_id )
        elem_bez_node_ids = getElementBezierNodeIds( uspline, elem_id )
        elem_extraction_operator = getElementExtractionOperator( uspline, elem_id )
        for i in range( 0, len( elem_bez_node_ids ) ):
            I = elem_bez_node_ids[i]
            for j in range( 0, len( elem_node_ids ) ):
                J = elem_node_ids[j]
                glob_extraction_operator[J,I] = elem_extraction_operator[j, i]
    return glob_extraction_operator

def getElementBezierNodes( uspline, elem_id ):
    elem_nodes = getElementNodes( uspline, elem_id )
    C = getElementExtractionOperator( uspline, elem_id )
    element_bezier_node_coords = C.T @ elem_nodes
    return element_bezier_node_coords

def getElementBezierVertices( uspline, elem_id ):
    element_bezier_node_coords = getElementBezierNodes( uspline, elem_id )
    vertex_connectivity = getVertexConnectivity( uspline )
    vertex_coords = numpy.array( [ element_bezier_node_coords[0], element_bezier_node_coords[-1] ] )
    return vertex_coords

def getBezierNodes( uspline ):
    bezier_nodes = []
    for elem_idx in range( 0, getNumElems( uspline ) ):
        elem_id = elemIdFromElemIdx( uspline, elem_idx )
        elem_bezier_nodes = getElementBezierNodes( uspline, elem_id )
        bezier_nodes.append( elem_bezier_nodes )
    bezier_nodes = uniquetol( bezier_nodes, 1e-12 )
    return bezier_nodes

def getElementIdContainingPoint( uspline, point ):
    num_elems = getNumElems( uspline )
    for elem_idx in range( 0, num_elems ):
        elem_id = elemIdFromElemIdx( uspline, elem_idx )
        elem_domain = getElementDomain( uspline, elem_id )
        if ( ( point >= elem_domain[0] ) and ( point <= elem_domain[1] ) ):
            return elem_id
    raise Exception( "ELEMENT_CONTAINING_POINT_NOT_FOUND" )

def uniquetol( input_array, tol ):
    equalityArray = numpy.zeros( len( input_array ), dtype="bool" )
    for i in range( 0, len( input_array) ):
        for j in range( i+1, len( input_array ) ):
            if abs( input_array[ i ] - input_array[ j ] ) <= tol:
                equalityArray[i] = True
    return input_array[ ~equalityArray ]

class Test_two_element_quadratic_bspline( unittest.TestCase ):
    def setUp( self ):
        self.uspline = readBEXT( "data/two_element_quadratic_bspline.json" )
    
    def test_getNumElems( self ):
        self.assertEqual( getNumElems( self.uspline ), 2 )
    
    def test_getNumVertices( self ):
        self.assertEqual( getNumVertices( self.uspline ), 3 )
    
    def test_getElementDegree( self ):
        self.assertEqual( getElementDegree( self.uspline, 0 ), 2 )
        self.assertEqual( getElementDegree( self.uspline, 1 ), 2 )
    
    def test_getSplineNodes( self ):
        gold_spline_nodes = numpy.array( [[0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], [1.5, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]] )
        test_spline_nodes = getSplineNodes( self.uspline )
        self.assertTrue( numpy.allclose( test_spline_nodes, gold_spline_nodes ) )
    
    def test_getCoefficientVectors( self ):
        gold_spline_nodes = {0: [0, 0, 0.5], 1: [0, 0, 1], 2: [0, 1, 0.5], 3: [0.5, 0, 0], 4: [0.5, 1, 0], 5: [1, 0, 0]}
        test_spline_nodes = getCoefficientVectors( self.uspline )
        self.assertEqual( test_spline_nodes, gold_spline_nodes )
    
    def test_getElementCoefficientVectorIds( self ):
        self.assertEqual( getElementCoefficientVectorIds( self.uspline, 0 ), [5, 2, 0] )
        self.assertEqual( getElementCoefficientVectorIds( self.uspline, 1 ), [3, 4, 1] )
    
    def test_getVertexConnectivity( self ):
        self.assertEqual( getVertexConnectivity( self.uspline ), [[0, 1],[1, 2]] )
    
    def test_getElementExtractionOperator( self ):
        gold_extraction_operator_0 = numpy.array( [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 0.5]] )
        gold_extraction_operator_1 = numpy.array( [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]] )
        test_extraction_operator_0 = getElementExtractionOperator( self.uspline, 0 )
        test_extraction_operator_1 = getElementExtractionOperator( self.uspline, 1 )
        self.assertTrue( numpy.allclose( test_extraction_operator_0, gold_extraction_operator_0 ) )
        self.assertTrue( numpy.allclose( test_extraction_operator_1, gold_extraction_operator_1 ) )
    
    def test_getElementBezierNodes( self ):
        gold_element_bezier_nodes_0 = numpy.array( [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]] )
        gold_element_bezier_nodes_1 = numpy.array( [[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]] )
        test_element_bezier_nodes_0 = getElementBezierNodes( self.uspline, 0 )
        test_element_bezier_nodes_1 = getElementBezierNodes( self.uspline, 1 )
        self.assertTrue( numpy.allclose( test_element_bezier_nodes_0, gold_element_bezier_nodes_0 ) )
        self.assertTrue( numpy.allclose( test_element_bezier_nodes_1, gold_element_bezier_nodes_1 ) )

    def test_getElementBezierVertices( self ):
        gold_element_bezier_vertices_0 = numpy.array( [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] )
        gold_element_bezier_vertices_1 = numpy.array( [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]] )
        test_element_bezier_vertices_0 = getElementBezierVertices( self.uspline, 0 )
        test_element_bezier_vertices_1 = getElementBezierVertices( self.uspline, 1 )
        self.assertTrue( numpy.allclose( test_element_bezier_vertices_0, gold_element_bezier_vertices_0 ) )
        self.assertTrue( numpy.allclose( test_element_bezier_vertices_1, gold_element_bezier_vertices_1 ) )

    def test_plotBasis( self ):
        fig, ax = plt.subplots()
        num_pts = 100
        for elem_idx in range( 0, getNumElems( self.uspline ) ):
            elem_id = elemIdFromElemIdx( self.uspline, elem_idx )
            elem_degree = getElementDegree( self.uspline, elem_id )
            C = getElementExtractionOperator( self.uspline, elem_id )
            elem_domain = getElementDomain( self.uspline, elem_id )
            x = numpy.linspace( elem_domain[0], elem_domain[1], num_pts )
            y = numpy.zeros( shape = ( 3, num_pts ) )
            for n in range( 0, elem_degree + 1 ):
                for i in range( 0, len( x ) ):
                    y[n, i] = basis.evalBernsteinBasis1D( elem_degree, n, elem_domain, x[i] )
            y = C @ y
            ax.plot( x, y.T, color = getLineColor( elem_idx ) )
        plt.show()

class Test_quadratic_bspline( unittest.TestCase ):
    def setUp( self ):
        self.uspline = readBEXT( "data/quadratic_bspline.json" )
    
    def test_getGlobalExtractionOperator( self ):
        C = getGlobalExtractionOperator( self.uspline )
    
class Test_multi_deg_uspline( unittest.TestCase ):
    def setUp( self ):
        self.uspline = readBEXT( "data/multi_deg_uspline.json" )
    
    def test_getNumElems( self ):
        self.assertEqual( getNumElems( self.uspline ), 4 )
    
    def test_getNumVertices( self ):
        self.assertEqual( getNumVertices( self.uspline ), 5 )
    
    def test_getElementDegree( self ):
        self.assertEqual( getElementDegree( self.uspline, 0 ),  1 )
        self.assertEqual( getElementDegree( self.uspline, 1 ),  2 )
        self.assertEqual( getElementDegree( self.uspline, 2 ),  3 )
        self.assertEqual( getElementDegree( self.uspline, 3 ),  4 )
    
    def test_getSplineNodes( self ):
        gold_spline_nodes = numpy.array( [ [0.0, 0.0, 0.0, 1.0], [1.953125, 0.0, 0.0, 1.0], [3.125, 0.0, 0.0, 1.0], [3.75, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 1.0] ] )
        test_spline_nodes = getSplineNodes( self.uspline )
        self.assertTrue( numpy.allclose( test_spline_nodes, gold_spline_nodes, atol = 1e-9 ) )
    
    def test_getCoefficientVectors( self ):
        gold_spline_nodes = { 0: [ 0, 0, 0, 0, 1 ], 1: [ 0,  0,  0,  0.2 ], 2: [ 0, 0, 0.24 ], 3: [ 0,  0.512 ], 4: [ 0.008,  0,  0,  0,  0 ], 5: [ 0.12,  0.04533333333333333,  0.01866666666666666,  0.008 ], 6: [ 0.192,  0.08,  0,  0,  0 ], 7: [ 0.2,  0.35,  0.6,  1.0,  0 ], 8: [ 0.24,  0.4,  0.64,  0.6 ], 9: [ 0.488,  0.232,  0.12 ], 10: [ 0.512,  0.768,  0.64 ], 11: [ 0.6,  0.57,  0.4,  0,  0 ], 12: [ 0.64,  0.5546666666666668,  0.3413333333333334,  0.192 ], 13: [ 1.0,  0.488 ] }
        test_spline_nodes = getCoefficientVectors( self.uspline )
        for i in gold_spline_nodes:
            self.assertTrue( numpy.allclose( gold_spline_nodes[i], test_spline_nodes[i], atol = 0.01 ) )
    
    def test_getElementCoefficientVectorIds( self ):
        self.assertEqual( getElementCoefficientVectorIds( self.uspline, 0 ), [13, 3] )
        self.assertEqual( getElementCoefficientVectorIds( self.uspline, 1 ), [9, 10, 2] )
        self.assertEqual( getElementCoefficientVectorIds( self.uspline, 2 ), [5, 12, 8, 1] )
        self.assertEqual( getElementCoefficientVectorIds( self.uspline, 3 ), [4, 6, 11, 7, 0] )
    
    def test_getVertexConnectivity( self ):
        self.assertEqual( getVertexConnectivity( self.uspline ), [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 4 ] ] )
    
    def test_getElementExtractionOperator( self ):
        gold_extraction_operator_0 = numpy.array( [[1.0, 0.488], [0.0, 0.512]] )
        gold_extraction_operator_1 = numpy.array( [[0.488, 0.232, 0.12], [0.512, 0.768, 0.64], [0.0, 0.0, 0.24]] )
        gold_extraction_operator_2 = numpy.array( [[0.12, 0.0453, 0.0187, 0.008], [0.64, 0.5547, 0.3413, 0.192], [0.24, 0.4, 0.64, 0.6], [0.0, 0.0, 0.0, 0.2]] )
        gold_extraction_operator_3 = numpy.array( [[0.008, 0.0, 0.0, 0.0, 0.0], [0.192, 0.08, 0.0, 0.0, 0.0], [0.6, 0.57, 0.4, 0.0, 0.0], [0.2, 0.35, 0.6, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]] )
        test_extraction_operator_0 = getElementExtractionOperator( self.uspline, 0 )
        test_extraction_operator_1 = getElementExtractionOperator( self.uspline, 1 )
        test_extraction_operator_2 = getElementExtractionOperator( self.uspline, 2 )
        test_extraction_operator_3 = getElementExtractionOperator( self.uspline, 3 )
        self.assertTrue( numpy.allclose( test_extraction_operator_0, gold_extraction_operator_0, atol = 0.01 ) )
        self.assertTrue( numpy.allclose( test_extraction_operator_1, gold_extraction_operator_1, atol = 0.01 ) )
        self.assertTrue( numpy.allclose( test_extraction_operator_2, gold_extraction_operator_2, atol = 0.01 ) )
        self.assertTrue( numpy.allclose( test_extraction_operator_3, gold_extraction_operator_3, atol = 0.01 ) )
    
    def test_getElementBezierNodes( self ):
        gold_element_bezier_nodes_0 = numpy.array( [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] )
        gold_element_bezier_nodes_1 = numpy.array( [[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]] )
        gold_element_bezier_nodes_2 = numpy.array( [[2.0, 0.0, 0.0], [2.0 + (1.0/3.0), 0.0, 0.0], [2.0 + (2.0/3.0), 0.0, 0.0], [3.0, 0.0, 0.0]] )
        gold_element_bezier_nodes_3 = numpy.array( [[3.0, 0.0, 0.0], [3.25, 0.0, 0.0], [3.5, 0.0, 0.0], [3.75, 0.0, 0.0], [4.0, 0.0, 0.0]] )
        test_element_bezier_nodes_0 = getElementBezierNodes( self.uspline, 0 )
        test_element_bezier_nodes_1 = getElementBezierNodes( self.uspline, 1 )
        test_element_bezier_nodes_2 = getElementBezierNodes( self.uspline, 2 )
        test_element_bezier_nodes_3 = getElementBezierNodes( self.uspline, 3 )
        self.assertTrue( numpy.allclose( test_element_bezier_nodes_0, gold_element_bezier_nodes_0 ) )
        self.assertTrue( numpy.allclose( test_element_bezier_nodes_1, gold_element_bezier_nodes_1 ) )
        self.assertTrue( numpy.allclose( test_element_bezier_nodes_2, gold_element_bezier_nodes_2 ) )
        self.assertTrue( numpy.allclose( test_element_bezier_nodes_3, gold_element_bezier_nodes_3 ) )

    def test_getElementBezierVertices( self ):
        gold_element_bezier_vertices_0 = numpy.array( [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] )
        gold_element_bezier_vertices_1 = numpy.array( [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]] )
        test_element_bezier_vertices_0 = getElementBezierVertices( self.uspline, 0 )
        test_element_bezier_vertices_1 = getElementBezierVertices( self.uspline, 1 )
        self.assertTrue( numpy.allclose( test_element_bezier_vertices_0, gold_element_bezier_vertices_0 ) )
        self.assertTrue( numpy.allclose( test_element_bezier_vertices_1, gold_element_bezier_vertices_1 ) )

    def test_plotBasis( self ):
        fig, ax = plt.subplots()
        num_pts = 100
        for elem_idx in range( 0, getNumElems( self.uspline ) ):
            elem_id = elemIdFromElemIdx( self.uspline, elem_idx )
            elem_degree = getElementDegree( self.uspline, elem_id )
            C = getElementExtractionOperator( self.uspline, elem_id )
            elem_domain = getElementDomain( self.uspline, elem_id )
            x = numpy.linspace( elem_domain[0], elem_domain[1], num_pts )
            y = numpy.zeros( shape = ( elem_degree + 1, num_pts ) )
            for n in range( 0, elem_degree + 1 ):
                for i in range( 0, len( x ) ):
                    y[n, i] = basis.evalBernsteinBasis1D( elem_degree, n, elem_domain, x[i] )
            y = C @ y
            ax.plot( x, y.T, color = getLineColor( elem_idx ) )
        plt.show()
    
def getLineColor( idx ):
    colors = list( matplotlib.colors.TABLEAU_COLORS.keys() )
    num_colors = len( colors )
    mod_idx = idx % num_colors
    return matplotlib.colors.TABLEAU_COLORS[ colors[ mod_idx ] ]