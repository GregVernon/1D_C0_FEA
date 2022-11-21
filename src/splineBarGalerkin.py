import numpy
import scipy
import unittest
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use( "qtAgg" )

from . import basis
from . import bext
from . import quadrature
from . import uspline

def computeSolution( problem, uspline_bext ):
    gram_matrix = assembleGramMatrix( problem, uspline_bext )
    print( "gram_matrix\n", gram_matrix )
    force_vector = assembleForceVector( problem, uspline_bext )
    print( "force_vector\n", force_vector )
    gram_matrix, force_vector = applyDisplacement( problem, gram_matrix, force_vector, uspline_bext )
    print( "gram_matrix\n", gram_matrix )
    print( "force_vector\n", force_vector )
    coeff = numpy.linalg.solve( gram_matrix, force_vector )
    print( "coeff\n", coeff )
    coeff = assembleSolution( coeff, problem, uspline_bext )
    print( "coeff\n", coeff )
    return coeff

def assembleSolution( coeff, problem, uspline_bext ):
    disp_node_id = bext.getNodeIdNearPoint( uspline_bext, problem[ "displacement" ][ "position" ] )
    coeff = numpy.insert( coeff, disp_node_id, problem[ "displacement" ][ "value" ], axis = 0 )
    return coeff

def applyDisplacement( problem, gram_matrix, force_vector, uspline_bext ):
    disp_node_id = bext.getNodeIdNearPoint( uspline_bext, problem[ "displacement" ][ "position" ] )
    force_vector -= gram_matrix[:,disp_node_id] * problem[ "displacement" ][ "value" ]
    gram_matrix = numpy.delete( numpy.delete( gram_matrix, disp_node_id, axis = 0 ), disp_node_id, axis = 1 )
    force_vector = numpy.delete( force_vector, disp_node_id, axis = 0 )
    return gram_matrix, force_vector

def applyTraction( problem, force_vector, uspline_bext ):
    elem_id = bext.getElementIdContainingPoint( uspline_bext, problem[ "traction" ][ "position" ] )
    elem_domain = bext.getElementDomain( uspline_bext, elem_id )
    elem_degree = bext.getElementDegree( uspline_bext, elem_id )
    elem_nodes = bext.getElementNodeIds( uspline_bext, elem_id )
    elem_extraction_operator = bext.getElementExtractionOperator( uspline_bext, elem_id )
    for i in range( 0, elem_degree + 1 ):
        I = elem_nodes[i]
        Ni = lambda x: basis.evalSplineBasis1D( elem_extraction_operator, i, elem_domain, x )
        force_vector[I] += Ni( problem[ "traction" ][ "position" ] ) * problem[ "traction" ][ "value" ]
    return force_vector

def evaluateConstitutiveModel( problem ):
    return problem[ "elastic_modulus" ] * problem[ "area" ]

def assembleGramMatrix( problem, uspline_bext ):
    basis_deriv = 1
    num_nodes = bext.getNumNodes( uspline_bext )
    num_elems = bext.getNumElems( uspline_bext )
    gram_matrix = numpy.zeros( shape = ( num_nodes, num_nodes ) )
    for elem_idx in range( 0, num_elems ):
        elem_id = bext.elemIdFromElemIdx( uspline_bext, elem_idx )
        elem_domain = bext.getElementDomain( uspline_bext, elem_id )
        elem_degree = bext.getElementDegree( uspline_bext, elem_id )
        elem_nodes = bext.getElementNodeIds( uspline_bext, elem_id )
        elem_jacobian = ( elem_domain[1] - elem_domain[0] ) / ( 1 - 0 )
        elem_extraction_operator = bext.getElementExtractionOperator( uspline_bext, elem_id )
        num_qp = int( numpy.ceil( ( 2*( elem_degree - basis_deriv ) + 1 ) / 2.0 ) )
        for i in range( 0, elem_degree + 1):
            I = elem_nodes[i]
            Ni = lambda x: basis.evalSplineBasisDeriv1D( elem_extraction_operator, i, basis_deriv, elem_domain, basis.affine_mapping_1D( [-1, 1], elem_domain, x ) )
            for j in range( 0, elem_degree + 1 ):
                J = elem_nodes[j]
                Nj = lambda x: basis.evalSplineBasisDeriv1D( elem_extraction_operator, j, basis_deriv, elem_domain, basis.affine_mapping_1D( [-1, 1], elem_domain, x ) )
                integrand = lambda x: Ni( x ) * evaluateConstitutiveModel( problem ) * Nj( x )
                gram_matrix[I, J] += quadrature.quad( integrand, elem_domain, num_qp )
    return gram_matrix

def assembleForceVector( problem, uspline_bext ):
    num_nodes = bext.getNumNodes( uspline_bext )
    num_elems = bext.getNumElems( uspline_bext )
    force_vector = numpy.zeros( num_nodes )
    for elem_idx in range( 0, num_elems ):
        elem_id = bext.elemIdFromElemIdx( uspline_bext, elem_idx )
        elem_domain = bext.getElementDomain( uspline_bext, elem_id )
        elem_degree = bext.getElementDegree( uspline_bext, elem_id )
        elem_nodes = bext.getElementNodeIds( uspline_bext, elem_id )
        elem_extraction_operator = bext.getElementExtractionOperator( uspline_bext, elem_id )
        num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) )
        for i in range( 0, elem_degree + 1 ):
            I = elem_nodes[i]
            Ni = lambda x: basis.evalSplineBasis1D( elem_extraction_operator, i, [-1, 1], x )
            integrand = lambda x: Ni( x ) * problem[ "body_force" ]
            force_vector[I] += quadrature.quad( integrand, elem_domain, num_qp )
    force_vector = applyTraction( problem, force_vector, uspline_bext )
    return force_vector

def evaluateSolutionAt( x, coeff, uspline_bext ):
    elem_id = bext.getElementIdContainingPoint( uspline_bext, x )
    elem_nodes = bext.getElementNodeIds( uspline_bext, elem_id )
    elem_domain = bext.getElementDomain( uspline_bext, elem_id )
    elem_degree = bext.getElementDegree( uspline_bext, elem_id )
    elem_extraction_operator = bext.getElementExtractionOperator( uspline_bext, elem_id )
    sol = 0.0
    for n in range( 0, len( elem_nodes ) ):
        curr_node = elem_nodes[n]
        sol += coeff[curr_node] * basis.evalSplineBasis1D( extraction_operator = elem_extraction_operator, basis_idx = n, domain = elem_domain, variate = x )
    return sol

def computeElementFitError( target_fun, coeff, uspline_bext, elem_id ):
    domain = bext.getDomain( uspline_bext )
    elem_domain = bext.getElementDomain( uspline_bext, elem_id )
    elem_degree = bext.getElementDegree( uspline_bext, elem_id )
    num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) + 1 )
    abs_err_fun = lambda x : abs( target_fun( basis.affine_mapping_1D( [-1, 1], elem_domain, x ) ) - evaluateSolutionAt( basis.affine_mapping_1D( [-1, 1], elem_domain, x ), coeff, uspline_bext ) )
    abs_error = quadrature.quad( abs_err_fun, elem_domain, num_qp )
    # abs_error, residual = scipy.integrate.quad( abs_err_fun, elem_domain[0], elem_domain[1], epsrel = 1e-12, limit = 100 )
    return abs_error

def computeFitError( target_fun, coeff, uspline_bext ):
    num_elems = bext.getNumElems( uspline_bext )
    abs_error = 0.0
    for elem_idx in range( 0, num_elems ):
        elem_id = bext.elemIdFromElemIdx( uspline_bext, elem_idx )
        abs_error += computeElementFitError( target_fun, coeff, uspline_bext, elem_id )
    domain = bext.getDomain( uspline_bext )
    target_fun_norm, _ = scipy.integrate.quad( lambda x: abs( target_fun(x) ), domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    rel_error = abs_error / target_fun_norm
    return abs_error, rel_error

def plotCompareGoldTestSolution( gold_coeff, test_coeff, uspline_bext ):
    domain = bext.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], test_coeff, uspline_bext )
        yt[i] = evaluateSolutionAt( x[i], gold_coeff, uspline_bext )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()

def plotCompareFunToExactSolution( problem, test_coeff, uspline_bext ):
    domain = bext.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    ya = numpy.zeros( 1000 )
    ye = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        ya[i] = evaluateSolutionAt( x[i], test_coeff, uspline_bext )
        ye[i] = evaluateExactSolutionAt( problem, x[i] )
    plt.plot( x, ya )
    plt.plot( x, ye )
    plt.show()

def plotSolution( sol_coeff, uspline_bext ):
    domain = bext.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = evaluateSolutionAt( x[i], sol_coeff, uspline_bext )   
    plt.plot( x, y )
    plt.plot( bext.getSplineNodes( uspline_bext )[:,0], sol_coeff, color = "k", marker = "o", markerfacecolor = "k" )
    plt.show()

def evaluateExactSolutionAt( problem, x ):
    term_1 = problem[ "traction" ][ "value" ] / evaluateConstitutiveModel( problem ) * x
    term_2 = problem[ "displacement" ][ "value" ]
    term_3 =  ( ( problem[ "length" ]**2.0 * problem[ "body_force" ] / 2 ) / evaluateConstitutiveModel( problem ) ) - ( ( ( problem[ "length" ] - x )**2.0 * problem[ "body_force" ] / 2 ) / evaluateConstitutiveModel( problem ) )
    sol = term_1 + term_2 + term_3
    return sol

def plotExactSolution( problem ):
    domain = [0, problem[ "length" ] ]
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = evaluateExactSolutionAt( problem, x[i] )
    plt.plot( x, y )
    plt.show()

class test_ComputeSolution( unittest.TestCase ):
    def test_simple( self ):
        problem = { "elastic_modulus": 200e9,
                    "area": 1.0,
                    "length": 5.0,
                    "traction": { "value": 9810.0, "position": 5.0 }, 
                    "displacement": { "value": 0.0, "position": 0.0 }, 
                    "body_force": 784800.0 }
        # spline_space = { "domain": [ 0, problem[ "length" ] ], "degree": [ 1 ], "continuity": [ -1, -1 ] }
        # spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
        # spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 0, -1 ] }
        spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_sol_coeff = computeSolution( problem = problem, uspline_bext = uspline_bext )
        print( test_sol_coeff )
        plotSolution( test_sol_coeff, uspline_bext )
        plotCompareFunToExactSolution( problem, test_sol_coeff, uspline_bext )
    
class test_plotExactSolution( unittest.TestCase ):
    def test_simple( self ):
        problem = { "elastic_modulus": 200e9,
                    "area": 1.0,
                    "length": 5.0,
                    "traction": { "value": 9810.0, "position": 5.0 }, 
                    "displacement": { "value": 0.0, "position": 0.0 }, 
                    "body_force": 784800.0 }
        plotExactSolution( problem = problem )