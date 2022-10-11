import json
import numpy
import unittest

def readBEXT( filename ):
    f = open( filename, "r" )
    uspline = json.load( f )
    f.close()
    return uspline

def getNumElems( uspline ):
    return uspline["num_elems"]

def getNumVertices( uspline ):
    return uspline["num_vertices"]

def getElementDegree( uspline, elem_idx ):
    return len( uspline["elements"]["element_blocks"][elem_idx]["node_ids"] ) - 1

def getSplineNodes( uspline ):
    return numpy.array( uspline["nodes"] )

def getCoefficientVectors( uspline ):
    coeff_vectors_list = uspline["coefficients"]["dense_coefficient_vectors"]
    coeff_vectors = {}
    for i in range( 0, len( coeff_vectors_list ) ):
        coeff_vectors[i] = coeff_vectors_list[i]["components"]
    return coeff_vectors

def getElementCoefficientVectorIds( uspline, elem_idx ):
    return uspline["elements"]["element_blocks"][elem_idx]["coeff_vector_ids"]

def getVertexConnectivity( uspline ):
    return uspline["vertex_connectivity"]

def getElementExtractionOperator( uspline, elem_idx ):
    coeff_vectors = getCoefficientVectors( uspline )    
    coeff_vector_ids = getElementCoefficientVectorIds( uspline, elem_idx )
    C = numpy.zeros( shape = (len( coeff_vector_ids ), len( coeff_vector_ids ) ), dtype = "double" )
    for n in range( 0, len( coeff_vector_ids ) ):
        C[n,:] = coeff_vectors[ coeff_vector_ids[n] ]
    return C

def getElementBezierNodes( uspline, elem_idx ):
    num_elems = getNumElems( uspline )
    spline_nodes = getSplineNodes( uspline )
    C = getElementExtractionOperator( uspline, elem_idx )
    element_spline_node_ids = numpy.array( uspline["elements"]["element_blocks"][elem_idx]["node_ids"] )
    element_spline_node_coords = numpy.array( uspline["nodes"] )[:,0:-1][element_spline_node_ids]
    element_bezier_node_coords = C.T @ element_spline_node_coords
    return element_bezier_node_coords

def getElementBezierVertices( uspline, elem_idx ):
    element_bezier_node_coords = getElementBezierNodes( uspline, elem_idx )
    vertex_connectivity = getVertexConnectivity( uspline )
    vertex_coords = numpy.array( [ element_bezier_node_coords[0], element_bezier_node_coords[-1] ] )
    return vertex_coords

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
        gold_spline_nodes = numpy.array( [[0.0, 0.0, 0.0, 1.0], [0.25, 0.0, 0.0, 1.0], [0.75, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]] )
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
        gold_element_bezier_nodes_0 = numpy.array( [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]] )
        gold_element_bezier_nodes_1 = numpy.array( [[0.5, 0.0, 0.0], [0.75, 0.0, 0.0], [1.0, 0.0, 0.0]] )
        test_element_bezier_nodes_0 = getElementBezierNodes( self.uspline, 0 )
        test_element_bezier_nodes_1 = getElementBezierNodes( self.uspline, 1 )
        self.assertTrue( numpy.allclose( test_element_bezier_nodes_0, gold_element_bezier_nodes_0 ) )
        self.assertTrue( numpy.allclose( test_element_bezier_nodes_1, gold_element_bezier_nodes_1 ) )

    def test_getElementBezierVertices( self ):
        gold_element_bezier_vertices_0 = numpy.array( [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]] )
        gold_element_bezier_vertices_1 = numpy.array( [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]] )
        test_element_bezier_vertices_0 = getElementBezierVertices( self.uspline, 0 )
        test_element_bezier_vertices_1 = getElementBezierVertices( self.uspline, 1 )
        self.assertTrue( numpy.allclose( test_element_bezier_vertices_0, gold_element_bezier_vertices_0 ) )
        self.assertTrue( numpy.allclose( test_element_bezier_vertices_1, gold_element_bezier_vertices_1 ) )
    