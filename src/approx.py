import unittest
import math
import numpy
import sympy
import scipy

if __name__ == "src.approx":
    from src import basis
    from src import mesh
elif __name__ == "approx":
    import basis
    import mesh

def computeSolution( target_fun, domain, degree ):
    node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
    coeff = target_fun( node_coords )
    return coeff, node_coords, ien_array

def computeSolutionDegreeRefine( target_fun, domain, num_elems, target_error ):
    degree = [ 1 ] * num_elems
    is_converged = False
    while is_converged == False:
        coeff, node_coords, ien_array = computeSolution( target_fun, domain, degree )
        degree, fit_error = solutionBasedDegreeRefine( target_fun, coeff, node_coords, ien_array, basis.evalLagrangeBasis1D, target_error )
        if numpy.all( fit_error <= target_error ):
            is_converged = True
    return coeff, node_coords, ien_array

def solutionBasedDegreeRefine( target_fun, coeff, node_coords, ien_array, eval_basis, target_error ):
    num_elems = len( ien_array )
    fit_error = numpy.zeros( num_elems )
    degree = numpy.zeros( num_elems, dtype = "int" )
    for e in range( 0, num_elems ):
        degree[e] = len( ien_array[e] ) - 1
        fit_error[e], _ = computeElementFitError( target_fun, coeff, node_coords, ien_array, e, eval_basis )
    refine_elem_idx = numpy.argwhere( fit_error > target_error ).squeeze()
    degree[refine_elem_idx] += 1
    return degree, fit_error

def evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis ):
    elem_idx = mesh.getElementIdxContainingPoint( node_coords, ien_array, x )
    elem_nodes = ien_array[elem_idx]
    elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
    xi = mesh.refToParamCoords( x, elem_domain )
    degree = len( elem_nodes ) - 1
    y = 0.0
    for n in range( 0, len( elem_nodes ) ):
        curr_node = elem_nodes[n]
        y += coeff[curr_node] * eval_basis( degree = degree, basis_idx = n, variate = xi )
    return y

def computeElementFitError( target_fun, coeff, node_coords, ien_array, elem_idx, eval_basis ):
    elem_nodes = ien_array[elem_idx]
    domain = [ node_coords[elem_nodes[0]], node_coords[elem_nodes[-1]] ]
    abs_err_fun = lambda x : abs( target_fun( x ) - evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis ) )
    fit_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = 100 )
    return fit_error, residual

def computeFitError( target_fun, coeff, node_coords, ien_array, eval_basis ):
    num_elems = len( ien_array )
    domain = [ min( node_coords ), max( node_coords ) ]
    abs_err_fun = lambda x : abs( target_fun( x ) - evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis ) )
    fit_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    return fit_error, residual

class Test_evaluateSolutionAt( unittest.TestCase ):
    def test_single_linear_element( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, [ 1 ] )
        coeff = numpy.array( [-1.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = -1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )

    def test_two_linear_elements( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, [ 1, 1 ] )
        coeff = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )

    def test_single_quadratic_element( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, [ 2 ] )
        coeff = numpy.array( [+1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )

    def test_two_quadratic_elements( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, [ 2, 2 ] )
        coeff = numpy.array( [ 1.0, 0.25, 0.5, 0.25, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.00 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +0.25 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +0.50 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +0.25 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.00 )


class Test_computeSolution( unittest.TestCase ):
    def test_single_linear_element_poly( self ):
        test_solution, _, _ = computeSolution( target_fun = lambda x : x, domain = [-1.0, 1.0 ], degree = [ 1 ] )
        gold_solution = numpy.array( [ -1.0, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
    
    def test_single_quad_element_poly( self ):
        test_solution, _, _ = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], degree = [ 2 ] )
        gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
    
    def test_two_linear_element_poly( self ):
        test_solution, _, _ = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], degree = [ 1, 1 ] )
        gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
    
    def test_four_quad_element_poly( self ):
        test_solution, _, _ = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], degree = [ 1, 1, 1, 1 ] )
        gold_solution = numpy.array( [ 1.0, 0.25, 0.0, 0.25, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

class Test_computeSolutionDegreeRefine( unittest.TestCase ):
    def test_single_element_sin( self ):
        target_fun = lambda x : numpy.sin( numpy.pi * x )
        domain = [ -1, 1 ]
        coeff, node_coords, ien_array = computeSolutionDegreeRefine( target_fun = target_fun, domain = domain, num_elems = 1, target_error = 1e-4 )
        print( mesh.getDegreeListFromIENArray( ien_array ) )

    def test_single_element_exp( self ):
        target_fun = lambda x : numpy.exp( x )
        domain = [ -1, 1 ]
        coeff, node_coords, ien_array = computeSolutionDegreeRefine( target_fun = target_fun, domain = domain, num_elems = 1, target_error = 1e-4 )
        print( mesh.getDegreeListFromIENArray( ien_array ) )
    
    def test_single_element_erfc( self ):
        target_fun = lambda x : scipy.special.erfc( x )
        domain = [ -2, 2 ]
        coeff, node_coords, ien_array = computeSolutionDegreeRefine( target_fun = target_fun, domain = domain, num_elems = 1, target_error = 1e-8 )
        print( mesh.getDegreeListFromIENArray( ien_array ) )
    
    def test_four_element_sin( self ):
        target_fun = lambda x : numpy.sin( numpy.pi * x )
        domain = [ -1, 1 ]
        coeff, node_coords, ien_array = computeSolutionDegreeRefine( target_fun = target_fun, domain = domain, num_elems = 4, target_error = 1e-4 )
        print( mesh.getDegreeListFromIENArray( ien_array ) )

    def test_four_element_exp( self ):
        target_fun = lambda x : numpy.exp( x )
        domain = [ -1, 1 ]
        coeff, node_coords, ien_array = computeSolutionDegreeRefine( target_fun = target_fun, domain = domain, num_elems = 4, target_error = 1e-4 )
        print( mesh.getDegreeListFromIENArray( ien_array ) )
    
    def test_four_element_erfc( self ):
        target_fun = lambda x : scipy.special.erfc( x )
        domain = [ -2, 2 ]
        coeff, node_coords, ien_array = computeSolutionDegreeRefine( target_fun = target_fun, domain = domain, num_elems = 4, target_error = 1e-8 )
        print( mesh.getDegreeListFromIENArray( ien_array ) )

class Test_computeFitError( unittest.TestCase ):
    def test_single_element_quad_poly( self ):
        target_fun = lambda x : x**2
        coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = [ 0.0, 1.0 ], degree = [ 1 ] )
        fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
        self.assertAlmostEqual( first = fit_error, second = 1.0 / 6.0 )
    
    def test_linear_approx_convergence_rate_quad_poly( self ):
        target_fun = lambda x : x**2
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ 0.0, 1.0 ]
        degree = 1
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        self.assertAlmostEqual( first = conv_rate, second = degree + 1, delta = 1e-1 )
    
    def test_quad_approx_convergence_rate_cubic_poly( self ):
        target_fun = lambda x : x**3
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ 0.0, 1.0 ]
        degree = 2
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        self.assertAlmostEqual( first = conv_rate, second = degree + 1, delta = 1e-1 )
    
    def test_linear_approx_convergence_rate_sin( self ):
        target_fun = lambda x : numpy.sin( numpy.pi * x )
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ -1.0, 1.0 ]
        degree = 1
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        self.assertAlmostEqual( first = conv_rate, second = degree + 1, delta = 5e-1 )
    
    def test_quad_approx_convergence_rate_sin( self ):
        target_fun = lambda x : numpy.sin( numpy.pi * x )
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ -1.0, 1.0 ]
        degree = 2
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        self.assertAlmostEqual( first = conv_rate, second = degree + 1, delta = 5e-1 )
    
    def test_linear_approx_convergence_rate_exp( self ):
        target_fun = lambda x : numpy.exp( x )
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ -1.0, 1.0 ]
        degree = 1
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        self.assertAlmostEqual( first = conv_rate, second = degree + 1, delta = 5e-1 )
    
    def test_quad_approx_convergence_rate_exp( self ):
        target_fun = lambda x : numpy.exp( x )
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ -1.0, 1.0 ]
        degree = 2
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        self.assertAlmostEqual( first = conv_rate, second = degree + 1, delta = 5e-1 )
    
    def test_linear_approx_convergence_rate_erfc( self ):
        target_fun = lambda x : scipy.special.erfc( x )
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ -2.0, 2.0 ]
        degree = 1
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        self.assertAlmostEqual( first = conv_rate, second = degree + 1, delta = 5e-1 )
    
    def test_quad_approx_convergence_rate_erfc( self ):
        target_fun = lambda x : scipy.special.erfc( x )
        n = numpy.array( range( 0, 10 ) )
        num_elems = 1 * 2**n
        domain = [ -2.0, 2.0 ]
        degree = 2
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( num_elems ) ):
            degree_list = [ degree ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        num_elems_log10 = numpy.log10( num_elems )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( num_elems_log10[-1] - num_elems_log10[0] ) )
        self.assertAlmostEqual( first = conv_rate, second = degree + 1, delta = 5e-1 )

    def test_two_element_p_convergence_sin( self ):
        target_fun = lambda x : numpy.sin( numpy.pi * x )
        num_elems = 2
        domain = [ -1.0, 1.0 ]
        degree = numpy.array( range( 1, 9 ), dtype = int )
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        for i in range( 0, len( degree ) ):
            degree_list = [ int( degree[i] ) ] * num_elems
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        degree_log10 = numpy.log10( degree )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( degree_log10[-1] - degree_log10[0] ) )
    
    def test_two_element_p_convergence_erfc( self ):
        target_fun = lambda x : scipy.special.erfc( x )
        num_elems = 2
        domain = [ -1.0, 1.0 ]
        degree = numpy.array( range( 1, 9 ), dtype = int )
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        num_nodes = []
        for i in range( 0, len( degree ) ):
            degree_list = [ int( degree[i] ) ] * num_elems
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            num_nodes.append( len(coeff) )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        degree_log10 = numpy.log10( degree )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( degree_log10[-1] - degree_log10[0] ) )
    
    def test_hp_convergence_sin( self ):
        target_fun = lambda x : numpy.sin( numpy.pi * x )
        n = numpy.array( range( 1, 11 ) )
        num_elems = 1 * 2**n
        domain = [ -1.0, 1.0 ]
        degree = numpy.array( range( 1, 11 ), dtype = int )
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        num_nodes = []
        for i in range( 0, len( degree ) ):
            degree_list = [ int( degree[i] ) ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            num_nodes.append( len(coeff) )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        degree_log10 = numpy.log10( degree )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( degree_log10[-1] - degree_log10[0] ) )

    def test_hp_convergence_erfc( self ):
        target_fun = lambda x : scipy.special.erfc( x )
        n = numpy.array( range( 1, 11 ) )
        num_elems = 1 * 2**n
        domain = [ -1.0, 1.0 ]
        degree = numpy.array( range( 1, 11 ), dtype = int )
        eval_basis = basis.evalLagrangeBasis1D
        error = []
        num_nodes = []
        for i in range( 0, len( degree ) ):
            degree_list = [ int( degree[i] ) ] * num_elems[i]
            coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree_list )
            fit_error, residual = computeFitError( target_fun = target_fun, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D )
            num_nodes.append( len(coeff) )
            error.append( fit_error )
        error_log10 = numpy.log10( error )
        degree_log10 = numpy.log10( degree )
        conv_rate = abs( ( error_log10[-1] - error_log10[0] ) / ( degree_log10[-1] - degree_log10[0] ) )
