import json
import numpy

def readBEXT( filename ):
    f = open( filename, "r" )
    bext = json.load( f )
    f.close()
    return bext

def getNumElems( bext ):
    return bext["num_elems"]

def getNumVertices( bext ):
    return bext["num_vertices"]
    
def getElementExtractionOperators( bext ):
    num_elems = getNumElems( bext )
    C = {}
    coeff_vectors = bext["coefficients"]["dense_coefficient_vectors"]
    for e in range( 0, num_elems ):
        coeff_vector_ids = bext["elements"]["element_blocks"][e]["coeff_vector_ids"]
        C[e] = numpy.zeros( shape = (len( coeff_vector_ids ), len( coeff_vector_ids ) ), dtype = "double" )
        for n in range( 0, len( coeff_vector_ids ) ):
            C[e][n,:] = coeff_vectors[ coeff_vector_ids[n] ]["components"]
    return C

def getElementBezierNodes( bext ):
    num_elems = getNumElems( bext )
    spline_nodes = bext["nodes"]
    C = getElementExtractionOperators( bext )
    element_bezier_node_coords = {}
    for e in range( 0, num_elems ):
        element_spline_node_ids = numpy.array( bext["elements"]["element_blocks"][e]["node_ids"] )
        element_spline_node_coords = numpy.array( bext["nodes"] )[:,0:-1][element_spline_node_ids]
        element_bezier_node_coords[e] = C[e].T @ element_spline_node_coords
    return element_bezier_node_coords

def getBezierVertices( bext ):
    element_bezier_node_coords = getElementBezierNodes( bext )
    vertex_connectivity = bext["vertex_connectivity"]
    vertex_coords = [[]] * getNumVertices( bext )
    for e in range( 0, len( vertex_connectivity ) ):
        vertex_coords[vertex_connectivity[e][0]] = element_bezier_node_coords[e][0]
        vertex_coords[vertex_connectivity[e][1]] = element_bezier_node_coords[e][-1]
    vertex_coords = numpy.array( vertex_coords )
    return vertex_coords

