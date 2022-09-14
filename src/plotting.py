import numpy
import matplotlib.pyplot as plt
import unittest
import scipy
from scipy import special
import basis
import mesh

def plotPiecewiseApproximationCoeffs( ien_array, node_coords, coeff, color_by ):
    fig, ax = plt.subplots()
    colors = numpy.array( plt.cm.Dark2_r.colors )
    xi = numpy.linspace( -1.0, 1.0, 100 )
    num_elems = ien_array.shape[0]
    if ( color_by == "GLOBAL_ID" ):
        color_idx = numpy.array( range( 0, len( coeff ) ) ) % 8
    else:
        color_idx = 0
    ax.scatter( node_coords, coeff, c = colors[ color_idx ] )
    plt.show()
    return fig, ax, plt

def plotPiecewiseApproximation( ien_array, node_coords, coeff, basisEval, color_by ):
    fig, ax = plt.subplots()
    colors = plt.cm.Dark2_r.colors
    xi = numpy.linspace( -1.0, 1.0, 100 )
    num_elems = ien_array.shape[0]
    for e in range( 0, num_elems ):
        elem_nodes = ien_array[e]
        degree = len( elem_nodes ) - 1
        elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
        x = numpy.zeros( len( xi ) )
        y = numpy.zeros( len( xi ) )
        for i in range( 0, len( xi ) ):
            x[i] = mesh.paramToRefCoords( xi[i], elem_domain )
            for n in range( 0, len( elem_nodes ) ):
                curr_node = elem_nodes[n]
                y[i] += coeff[curr_node] * basisEval( degree = degree, basis_idx = n, variate = xi[i] )            
        if ( color_by == "ELEMENT_ID" ):
            color_idx = e % len( colors )
        else:
            color_idx = 0
        ax.plot(x, y, linewidth=2.0, color = colors[ color_idx ] )
    plt.show()
    return fig, ax, plt

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
                x[i] = mesh.paramToRefCoords( xi[i], elem_domain )
                y[i] = coeff[curr_node] * basisEval( degree = degree, basis_idx = n, variate = xi[i] )
            if ( color_by == "GLOBAL_ID" ):
                color_idx = curr_node % len( colors )
            elif ( color_by == "ELEMENT_ID" ):
                color_idx = e % len( colors )
            else:
                color_idx = n % len( colors )
            ax.plot(x, y, linewidth=2.0, color = colors[ color_idx ] )
    plt.show()
    return fig, ax, plt


class Test_plotBasisMesh( unittest.TestCase ):
    def test_3_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 4, 3, 1 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = basis.evalLagrangeBasis1D, color_by = "GLOBAL_ID" )

    def test_3_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 4, 3, 2 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = basis.evalLagrangeBasis1D, color_by = "GLOBAL_ID" )

    def test_3_quadratic_bernstein( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 4, 3, 2 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = basis.evalBernsteinBasis1D, color_by = "GLOBAL_ID" )

    def test_10_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 11, 10, 1 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = basis.evalLagrangeBasis1D, color_by = "GLOBAL_ID" )

    def test_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 11, 10, 2 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = basis.evalLagrangeBasis1D, color_by = "GLOBAL_ID" )
    
    def test_approx_erfc_10_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 10, 1 )
        coeff = scipy.special.erfc( node_coords )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = basis.evalLagrangeBasis1D, color_by = "GLOBAL_ID" )
    
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 10, 2 )
        coeff = scipy.special.erfc( node_coords )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = basis.evalLagrangeBasis1D, color_by = "GLOBAL_ID" )

class Test_plotPiecewiseApproximation( unittest.TestCase ):
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 2, 2 )
        coeff = scipy.special.erfc( node_coords )
        plotPiecewiseApproximation( ien_array = ien_array, node_coords = node_coords, coeff = coeff, basisEval = basis.evalLagrangeBasis1D, color_by = "ELEMENT_ID" )

class Test_plotPiecewiseApproximationCoeffs( unittest.TestCase ):
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 2, 2 )
        coeff = scipy.special.erfc( node_coords )
        plotPiecewiseApproximationCoeffs( ien_array = ien_array, node_coords = node_coords, coeff = coeff, color_by = "GLOBAL_ID" )