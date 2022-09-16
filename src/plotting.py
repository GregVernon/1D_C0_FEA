import numpy
import matplotlib
import matplotlib.pyplot as plt
import unittest
import scipy
from scipy import special

if __name__ == "src.plotting":
    from src import basis
    from src import mesh
elif __name__ == "plotting":
    import basis
    import mesh

## Uncomment this if you want to see figures
#matplotlib.use('TkAgg')

def plotPiecewiseFunctionFit( ien_array, node_coords, coeff, eval_basis, options ):
    fig, ax = plt.subplots()
    handles = { "fig": fig, "ax": ax }
    if ( "plotFunction" in options.keys() ):
        ax = plotFunction( options["plotFunction"]["lambda"], options["plotFunction"]["domain"], handles )
    if ( "plotMeshElementBoundaries" in options.keys() ):
        ax = plotMeshElementBoundaries( ien_array, node_coords, handles )
    if ( "plotMeshBasis" in options.keys() ):
        ax = plotMeshBasis( ien_array, node_coords, coeff, eval_basis, handles, options["plotMeshBasis"]["color_by"] )
    if ( "plotPiecewiseApproximation" in options.keys() ):
        ax = plotPiecewiseApproximation( ien_array, node_coords, coeff, eval_basis, handles, options["plotPiecewiseApproximation"]["color_by"] )
    if ( "plotPiecewiseApproximationCoeffs" in options.keys() ):
        ax = plotPiecewiseApproximationCoeffs( ien_array, node_coords, coeff, handles, options["plotPiecewiseApproximationCoeffs"]["color_by"] )
    plt.show()

def plotPiecewiseApproximationCoeffs( ien_array, node_coords, coeff, handles, color_by ):
    if ( not handles ):
        fig, ax = plt.subplots()
    else: 
        ax = handles[ "ax" ]
        fig = handles[ "ax" ]
    colors = numpy.array( plt.cm.Dark2_r.colors )
    xi = numpy.linspace( -1.0, 1.0, 100 )
    num_elems = ien_array.shape[0]
    if ( color_by == "GLOBAL_ID" ):
        color_idx = numpy.array( range( 0, len( coeff ) ) ) % 8
    else:
        color_idx = 0
    ax.scatter( node_coords, coeff, c = colors[ color_idx ] )
    if ( not handles ):
        plt.show()
    return fig, ax

def plotPiecewiseApproximation( ien_array, node_coords, coeff, eval_basis, handles, color_by ):
    if ( not handles ):
        fig, ax = plt.subplots()
    else: 
        ax = handles[ "ax" ]
        fig = handles[ "ax" ]
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
                y[i] += coeff[curr_node] * eval_basis( degree = degree, basis_idx = n, variate = xi[i] )            
        if ( color_by == "ELEMENT_ID" ):
            color_idx = e % len( colors )
        else:
            color_idx = 0
        ax.plot(x, y, linewidth=2.0, color = colors[ color_idx ] )
    if ( not handles ):
        plt.show()
    handles = { "fig": fig, "ax": ax }
    return handles

def plotMeshBasis( ien_array, node_coords, coeff, eval_basis, handles, color_by ):
    if ( not handles ):
        fig, ax = plt.subplots()
    else: 
        ax = handles[ "ax" ]
        fig = handles[ "ax" ]
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
                y[i] = coeff[curr_node] * eval_basis( degree = degree, basis_idx = n, variate = xi[i] )
            if ( color_by == "GLOBAL_ID" ):
                color_idx = curr_node % len( colors )
            elif ( color_by == "ELEMENT_ID" ):
                color_idx = e % len( colors )
            else:
                color_idx = n % len( colors )
            ax.plot(x, y, linewidth=2.0, color = colors[ color_idx ] )
    if ( not handles ):
        plt.show()
    handles = { "fig": fig, "ax": ax }
    return handles

def plotMeshElementBoundaries( ien_array, node_coords, handles ):
    if ( not handles ):
        fig, ax = plt.subplots()
    else: 
        ax = handles[ "ax" ]
        fig = handles[ "ax" ]
    num_elems = ien_array.shape[0]
    for e in range( 0, num_elems ):
        elem_nodes = ien_array[e]
        elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
        if ( e == 0 ):
            ax.axvline( x = elem_domain[0], linestyle = "-" )
            ax.axvline( x = elem_domain[1], linestyle = "--" )
        elif ( e == (num_elems - 1 ) ):
            ax.axvline( x = elem_domain[1], linestyle = "-" )
        else:
            ax.axvline( x = elem_domain[1], linestyle = "--" )
            
    if ( not handles ):
        plt.show()
    handles = { "fig": fig, "ax": ax }
    return handles

def plotFunction( fun, domain, handles ):
    if ( not handles ):
        fig, ax = plt.subplots()
    else: 
        ax = handles[ "ax" ]
        fig = handles[ "ax" ]
    x = numpy.linspace( domain[0], domain[1] )          
    ax.plot(x, fun( x ), linewidth=2.0, color = [0, 0, 0] )
    if ( not handles ):
        plt.show()
    handles = { "fig": fig, "ax": ax }
    return handles

class Test_plotPiecwiseFunctionFit( unittest.TestCase ):
    def test_single_element_quad_poly( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 1, 1, 1 )
        coeff = node_coords ** 2.0
        fun = lambda x : x ** 2.0
        domain = [ 0.0, 1.0 ]
        # options = { "plotMeshBasis": {"color_by": "GLOBAL_ID"}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        options = { "plotFunction": {"lambda": fun, "domain": domain}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )
    
    def test_sin_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, 3, 2 )
        coeff = numpy.sin( numpy.pi * node_coords )
        options = { "plotMeshBasis": {"color_by": "GLOBAL_ID"}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )

    def test_exp_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, 3, 2 )
        coeff = numpy.exp( node_coords )
        options = { "plotMeshBasis": {"color_by": "GLOBAL_ID"}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )

    def test_erfc_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 3, 2 )
        coeff = scipy.special.erfc( node_coords )
        options = { "plotMeshBasis": {"color_by": "GLOBAL_ID"}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )
    
    def test_compare_sin_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, 3, 2 )
        coeff = numpy.sin( numpy.pi * node_coords )
        fun = lambda x : numpy.sin( numpy.pi * x )
        domain = [ -1.0, 1.0 ]
        options = { "plotFunction": {"lambda": fun, "domain": domain}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )

    def test_compare_exp_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, 3, 2 )
        coeff = numpy.exp( node_coords )
        fun = lambda x : numpy.exp( x )
        domain = [ -1.0, 1.0 ]
        options = { "plotFunction": {"lambda": fun, "domain": domain}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )

    def test_compare_erfc_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 3, 2 )
        coeff = scipy.special.erfc( node_coords )
        fun = lambda x : scipy.special.erfc( x )
        domain = [ -2.0, 2.0 ]
        options = { "plotFunction": {"lambda": fun, "domain": domain}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )

class Test_plotBasisMesh( unittest.TestCase ):
    def test_3_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 4, 3, 1 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )

    def test_3_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 4, 3, 2 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )

    def test_3_quadratic_bernstein( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 4, 3, 2 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalBernsteinBasis1D, handles = [], color_by = "GLOBAL_ID" )

    def test_10_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 11, 10, 1 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )

    def test_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 11, 10, 2 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )
    
    def test_approx_erfc_10_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 10, 1 )
        coeff = scipy.special.erfc( node_coords )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )
    
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 10, 2 )
        coeff = scipy.special.erfc( node_coords )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )

class Test_plotPiecewiseApproximation( unittest.TestCase ):
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 2, 2 )
        coeff = scipy.special.erfc( node_coords )
        plotPiecewiseApproximation( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "ELEMENT_ID" )

class Test_plotPiecewiseApproximationCoeffs( unittest.TestCase ):
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 2, 2 )
        coeff = scipy.special.erfc( node_coords )
        plotPiecewiseApproximationCoeffs( ien_array = ien_array, node_coords = node_coords, coeff = coeff, handles = [], color_by = "GLOBAL_ID" )

class Test_plotMeshElementBoundaries( unittest.TestCase ):
    def test_plot_3_elems( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 3, 3, 1 )
        plotMeshElementBoundaries( ien_array = ien_array, node_coords = node_coords, handles = [] )