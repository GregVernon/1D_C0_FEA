import numpy
import matplotlib
import matplotlib.pyplot as plt
import unittest
import scipy
from scipy import special
from scipy import integrate

if __name__ == "src.plotting":
    from src import basis
    from src import mesh
    from src import approx
elif __name__ == "plotting":
    import basis
    import mesh
    import approx

## Uncomment this if you want to see figures
# matplotlib.use('TkAgg')

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
    num_elems = len( ien_array )
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
    num_elems = len( ien_array )
    for e in range( 0, num_elems ):
        elem_nodes = ien_array[e]
        degree = len( elem_nodes ) - 1
        elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
        x = numpy.zeros( len( xi ) )
        y = numpy.zeros( len( xi ) )
        for i in range( 0, len( xi ) ):
            if ( eval_basis.__name__ == "evalBernsteinBasis1D"):
                x[i] = basis.affine_mapping_1D( domain = [-1.0, 1.0], target_domain = [0.0, 1.0], x = xi[i] )
            else:
                x[i] = mesh.paramToSpatialCoords( xi[i], elem_domain )
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
    num_elems = len( ien_array )
    for e in range( 0, num_elems ):
        elem_nodes = ien_array[e]
        degree = len( elem_nodes ) - 1
        elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
        for n in range( 0, len( elem_nodes ) ):
            curr_node = elem_nodes[n]
            for i in range( 0, len( xi ) ):
                x[i] = mesh.paramToSpatialCoords( xi[i], elem_domain )
                if ( eval_basis.__name__ == "evalBernsteinBasis1D"):
                    y[i] = coeff[curr_node] * eval_basis( degree = degree, basis_idx = n, variate = basis.affine_mapping_1D( domain = [-1.0, 1.0], target_domain = [0.0, 1.0], x = xi[i] ) )
                else:
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
    num_elems = len( ien_array )
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
        node_coords, ien_array = mesh.generateMesh( 0, 1, [ 1 ] )
        coeff = node_coords ** 2.0
        fun = lambda x : x ** 2.0
        domain = [ 0.0, 1.0 ]
        # options = { "plotMeshBasis": {"color_by": "GLOBAL_ID"}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        options = { "plotFunction": {"lambda": fun, "domain": domain}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )
        plt.close()
    
    def test_sin_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, [ 2, 2, 2 ] )
        coeff = numpy.sin( numpy.pi * node_coords )
        options = { "plotMeshBasis": {"color_by": "GLOBAL_ID"}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )
        plt.close()

    def test_exp_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, [ 2, 2, 2 ] )
        coeff = numpy.exp( node_coords )
        options = { "plotMeshBasis": {"color_by": "GLOBAL_ID"}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )
        plt.close()

    def test_erfc_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, [ 2, 2, 2 ] )
        coeff = scipy.special.erfc( node_coords )
        options = { "plotMeshBasis": {"color_by": "GLOBAL_ID"}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"}, "plotPiecewiseApproximationCoeffs": {"color_by": "GLOBAL_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )
        plt.close()
    
    def test_compare_sin_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, [ 2, 2, 2 ] )
        coeff = numpy.sin( numpy.pi * node_coords )
        fun = lambda x : numpy.sin( numpy.pi * x )
        domain = [ -1.0, 1.0 ]
        options = { "plotFunction": {"lambda": fun, "domain": domain}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )
        plt.close()

    def test_compare_exp_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, [ 2, 2, 2 ] )
        coeff = numpy.exp( node_coords )
        fun = lambda x : numpy.exp( x )
        domain = [ -1.0, 1.0 ]
        options = { "plotFunction": {"lambda": fun, "domain": domain}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )
        plt.close()

    def test_compare_erfc_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, [ 2, 2, 2 ] )
        coeff = scipy.special.erfc( node_coords )
        fun = lambda x : scipy.special.erfc( x )
        domain = [ -2.0, 2.0 ]
        options = { "plotFunction": {"lambda": fun, "domain": domain}, "plotPiecewiseApproximation": {"color_by": "ELEMENT_ID"} }
        plotPiecewiseFunctionFit( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, options = options )
        plt.close()

class Test_plotBasisMesh( unittest.TestCase ):
    def test_3_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 3, [ 1, 1, 1 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )
        plt.close()

    def test_3_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 3, [ 2, 2, 2 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )
        plt.close()

    def test_3_quadratic_bernstein( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 3, [ 2, 2, 2 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalBernsteinBasis1D, handles = [], color_by = "GLOBAL_ID" )
        plt.close()

    def test_4_p_refine_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 4, [ 1, 2, 3, 4 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )
        plt.close()

    def test_4_p_refine_bernstein( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 4, [ 1, 2, 3, 4 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalBernsteinBasis1D, handles = [], color_by = "GLOBAL_ID" )
        plt.close()

    def test_10_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 10, [ 1 ]*10 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )
        plt.close()

    def test_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 10, [ 2 ]*10 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )
        plt.close()
    
    def test_approx_erfc_10_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, [ 1 ]*10 )
        coeff = scipy.special.erfc( node_coords )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )
        plt.close()
    
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, [ 2 ]*10 )
        coeff = scipy.special.erfc( node_coords )
        plotMeshBasis( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "GLOBAL_ID" )
        plt.close()

class Test_plotPiecewiseApproximation( unittest.TestCase ):
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, [ 2, 2 ] )
        coeff = scipy.special.erfc( node_coords )
        plotPiecewiseApproximation( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "ELEMENT_ID" )
        plt.close()

    def test_approx_erfc_5_p_refine_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, [ 2, 1, 2 ] )
        coeff = scipy.special.erfc( node_coords )
        plotPiecewiseApproximation( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1D, handles = [], color_by = "ELEMENT_ID" )
        plt.close()

class Test_plotPiecewiseApproximationCoeffs( unittest.TestCase ):
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, [ 2, 2 ] )
        coeff = scipy.special.erfc( node_coords )
        plotPiecewiseApproximationCoeffs( ien_array = ien_array, node_coords = node_coords, coeff = coeff, handles = [], color_by = "GLOBAL_ID" )
        plt.close()

class Test_plotMeshElementBoundaries( unittest.TestCase ):
    def test_plot_3_elems( self ):
        node_coords, ien_array = mesh.generateMesh( 0, 3, [ 1, 1, 1 ] )
        plotMeshElementBoundaries( ien_array = ien_array, node_coords = node_coords, handles = [] )
        plt.close()

class Test_plotErrorConvergence( unittest.TestCase ):
    def test_linear_h_convergence_sin( self ):
        target_fun = lambda x : numpy.sin( numpy.pi * x )
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ -1.0, 1.0 ]
        degree = 1
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = approx.computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = approx.computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        fig, ax = plt.subplots()
        ax.loglog( num_elems, error, linewidth=2.0, color = [0, 0, 0] )
        ax.grid( visible = True, which = "both" )
        plt.show()
        plt.close()
    
    def test_quadratic_h_convergence_sin( self ):
        target_fun = lambda x : numpy.sin( numpy.pi * x )
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ -1.0, 1.0 ]
        degree = 1
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = approx.computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = approx.computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        fig, ax = plt.subplots()
        ax.loglog( num_elems, error, linewidth=2.0, color = [0, 0, 0] )
        ax.grid( visible = True, which = "both" )
        plt.show()
        plt.close()
        
    def test_two_element_p_convergence_sin( self ):
        target_fun = lambda x : numpy.sin( numpy.pi * x )
        num_elems = 2
        domain = [ -1.0, 1.0 ]
        p = numpy.array( range( 0, 6 ), dtype = int )
        degree = 1 * 2**p
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( degree ) ):
            degree_list = [ int( degree[i] ) ] * num_elems
            coeff, node_coords, ien_array = approx.computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = approx.computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        degree_log10 = numpy.log10( degree )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( degree_log10[-1] - degree_log10[0] ) )
        fig, ax = plt.subplots()
        ax.loglog( degree, error, linewidth=2.0, color = [0, 0, 0] )
        ax.grid( visible = True, which = "both" )
        plt.show()
        plt.close()
    