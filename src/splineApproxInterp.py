import unittest
import math
import numpy
import sympy
import scipy

if __name__ == "src.splineApprox":
    from src import basis
    from src import mesh
    from src import bext
    from src import quadrature
elif __name__ == "splineApprox":
    import basis
    import mesh
    import bext
    import quadrature

def computeSolution( target_fun, uspline ):
    x0 = numpy.zeros( bext.getNumNodes( uspline ) )
    sol = scipy.optimize.minimize( lambda x: computeFitError( x, target_fun, uspline ), x0 )
    coeff = sol.x
    return coeff

def evaluateSolutionAt( x, coeff, uspline ):
    elem_id = bext.getElementIdContainingPoint( uspline, x )
    elem_node_ids = bext.getElementNodeIds( uspline, elem_id )
    elem_domain = bext.getElementDomain( uspline, elem_id )
    elem_degree = bext.getElementDegree( uspline, elem_id )
    elem_extraction_operator = bext.getElementExtractionOperator( uspline, elem_id )
    xi = basis.affine_mapping_1D( elem_domain, [0.0, 1.0], x )
    y = 0.0
    for n in range( 0, len( elem_node_ids ) ):
        curr_node = elem_node_ids[n]
        y += coeff[curr_node] * ( elem_extraction_operator[n] @ basis.evalBernsteinBasis1DVector( degree = elem_degree, variate = xi ) )
    return y

def computeFitError( coeff, target_fun, uspline ):
    num_elems = bext.getNumElems( uspline )
    fit_error = 0.0
    for elem_idx in range( 0, num_elems ):
        elem_id = bext.elemIdFromElemIdx( uspline, elem_idx )
        fit_error += abs( computeElementFitError( coeff, elem_id, target_fun, uspline ) )
    return fit_error

def computeElementFitError( coeff, elem_id, target_fun, uspline ):
    elem_domain = bext.getElementDomain( uspline, elem_id )
    elem_degree = bext.getElementDegree( uspline, elem_id )
    num_qp = int( numpy.ceil( ( elem_degree + 1 ) / 2.0 ) )
    xi_qp, w_qp = quadrature.getGaussLegendreQuadrature( num_qp )
    elem_fit_error = 0.0
    for i in range( 0, len( xi_qp ) ):
        x_qp = basis.affine_mapping_1D( [-1.0, 1.0], elem_domain, xi_qp[i] )
        y = evaluateSolutionAt( x_qp, coeff, uspline )
        elem_fit_error += abs( target_fun( x_qp ) - y )
    return elem_fit_error