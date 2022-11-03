import unittest
import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "src.meshApproxGalerkin":
    from src import basis
    from src import mesh
    from src import quadrature
elif __name__ == "meshApproxGalerkin":
    import basis
    import mesh
    import quadrature

def computeSolution( target_fun, domain, degree, solution_basis ):
    node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
    gram_matrix = assembleGramMatrix( node_coords, ien_array, solution_basis )
    force_vector = assembleForceVector( target_fun, node_coords, ien_array, solution_basis )
    coeff = numpy.linalg.solve( gram_matrix, force_vector )
    return coeff, node_coords, ien_array

def assembleGramMatrix( node_coords, ien_array, solution_basis ):
    num_nodes = len( node_coords )
    num_elems = len( ien_array )
    gram_matrix = numpy.zeros( shape = ( num_nodes, num_nodes ) )
    for elem_idx in range( 0, num_elems ):
        elem_degree = len( ien_array[elem_idx] ) - 1
        elem_domain = mesh.getElementDomain( node_coords, ien_array, elem_idx )
        num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) )
        for i in range( 0, elem_degree + 1):
            I = ien_array[elem_idx][i]
            Ni = lambda x: solution_basis( elem_degree, i, [-1, 1], x )
            for j in range( 0, elem_degree + 1 ):
                J = ien_array[elem_idx][j]
                Nj = lambda x: solution_basis( elem_degree, j, [-1, 1], x )
                integrand = lambda x: Ni(x ) * Nj( x )
                gram_matrix[I, J] += quadrature.quad( integrand, elem_domain, num_qp )
    return gram_matrix

def assembleForceVector( target_fun, node_coords, ien_array, solution_basis ):
    num_nodes = len( node_coords )
    num_elems = len( ien_array )
    force_vector = numpy.zeros( num_nodes )
    for elem_idx in range( 0, num_elems ):
        elem_degree = len( ien_array[elem_idx] ) - 1
        elem_domain = mesh.getElementDomain( node_coords, ien_array, elem_idx )
        num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) )
        for i in range( 0, elem_degree + 1 ):
            I = ien_array[elem_idx][i]
            Ni = lambda x: solution_basis( elem_degree, i, [-1, 1], x )
            integrand = lambda x: Ni( x ) * target_fun( basis.affine_mapping_1D( [-1, 1], elem_domain, x ) )
            force_vector[I] += quadrature.quad( integrand, elem_domain, num_qp )
    return force_vector

def evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis ):
    elem_idx = mesh.getElementIdxContainingPoint( node_coords, ien_array, x )
    elem_nodes = ien_array[elem_idx]
    elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
    degree = len( elem_nodes ) - 1
    y = 0.0
    for n in range( 0, len( elem_nodes ) ):
        curr_node = elem_nodes[n]
        y += coeff[curr_node] * eval_basis( degree = degree, basis_idx = n, domain = elem_domain, variate = x )
    return y

def computeElementFitError( target_fun, coeff, node_coords, ien_array, elem_idx, eval_basis ):
    elem_nodes = ien_array[elem_idx]
    domain = [ node_coords[elem_nodes[0]], node_coords[elem_nodes[-1]] ]
    abs_err_fun = lambda x : abs( target_fun( x ) - evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis ) )
    abs_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = 100 )
    return abs_error, residual

def computeFitError( target_fun, coeff, node_coords, ien_array, eval_basis ):
    num_elems = len( ien_array )
    domain = [ min( node_coords ), max( node_coords ) ]
    target_fun_norm, _ = scipy.integrate.quad( lambda x: abs( target_fun(x) ), domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    abs_err_fun = lambda x : abs( target_fun( x ) - evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis ) )
    abs_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    rel_error = abs_error / target_fun_norm
    return abs_error, rel_error

def plotCompareGoldTestSolution( gold_coeff, test_coeff, node_coords, ien_array, solution_basis ):
    domain = [ min( node_coords ), max( node_coords ) ]
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], gold_coeff, node_coords, ien_array, solution_basis )
        yt[i] = evaluateSolutionAt( x[i], test_coeff, node_coords, ien_array, solution_basis )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()

def plotCompareFunToTestSolution( fun, test_coeff, node_coords, ien_array, solution_basis ):
    x = numpy.linspace( min( node_coords ), max( node_coords ), 1000 )
    y = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = fun( x[i] )
        yt[i] = evaluateSolutionAt( x[i], test_coeff, node_coords, ien_array, solution_basis )
    plt.plot( x, y )
    plt.plot( x, yt )
    plt.show()

class Test_computeSolution( unittest.TestCase ):
    def test_cubic_polynomial_target( self ):
        # print( "POLY TEST" )
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        domain = [ 0, 1 ]
        degree = [2]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis =solution_basis )
        gold_sol_coeff = numpy.array( [ 1.0 / 120.0, 9.0 / 80.0, 1.0 / 40.0, -1.0 / 16.0, -1.0 / 120.0 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )

    def test_sin_target( self ):
        # print( "SIN TEST" )
        target_fun = lambda x: numpy.sin( numpy.pi * x )
        domain = [ 0, 1 ]
        degree = [2]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ -0.02607008, 0.9185523, 1.01739261, 0.9185523, -0.02607008 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )
        
    def test_erfc_target( self ):
        # print( "ERFC TEST" )
        target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
        domain = [ -2, 2 ]
        degree = [3]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ 1.98344387, 2.0330054, 1.86372084, 1., 0.13627916, -0.0330054, 0.01655613 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )
    
    def test_exptx_target( self ):
        # print( "EXPT TEST" )
        target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
        domain = [ -1, 1 ]
        degree = [5]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = ( [ -1.00022471, -1.19005562, -0.9792369, 0.70884334, 1.73001439, 0.99212064, 0.44183573, 0.87014465, 0.5572111, 0.85241908, 0.99175228 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )

# class Test_evaluateSolutionAt( unittest.TestCase ):
#     def test_constant_bernstein( self ):
#         for x in numpy.arange( -1, 1, 7 ):
#             for coeff in numpy.arange( -1, 1, 7 ):
#                 self.assertAlmostEqual( evaluateSolutionAt( x = x, domain = [-1.0, 1.0], coeff = numpy.array( [coeff] ), solution_basis = basis.evalBernsteinBasis1D ), coeff )
    
#     def test_linear_bernstein( self ):
#         coeff = numpy.array( [1.0, 2.0] )
#         self.assertAlmostEqual( evaluateSolutionAt( x = -1.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 1.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 1.5 )
#         self.assertAlmostEqual( evaluateSolutionAt( x = +1.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 2.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.0, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 1.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.5, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 1.5 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  1.0, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 2.0 )
    
#     def test_quadratic_bernstein( self ):
#         coeff = numpy.array( [1.0, 3.0, 2.0] )
#         self.assertAlmostEqual( evaluateSolutionAt( x = -1.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 1.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 9.0 / 4.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x = +1.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 2.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.0, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 1.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.5, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 9.0 / 4.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  1.0, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalBernsteinBasis1D ), 2.0 )

#     def test_constant_lagrange( self ):
#         for x in numpy.arange( -1, 1, 7 ):
#             for coeff in numpy.arange( -1, 1, 7 ):
#                 self.assertAlmostEqual( evaluateSolutionAt( x = x, domain = [-1.0, 1.0], coeff = numpy.array( [coeff] ), solution_basis = basis.evalLagrangeBasis1D ), coeff )
    
#     def test_linear_lagrange( self ):
#         coeff = numpy.array( [1.0, 2.0] )
#         self.assertAlmostEqual( evaluateSolutionAt( x = -1.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 1.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 1.5 )
#         self.assertAlmostEqual( evaluateSolutionAt( x = +1.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 2.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.0, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 1.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.5, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 1.5 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  1.0, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 2.0 )
    
#     def test_quadratic_lagrange( self ):
#         coeff = numpy.array( [1.0, 3.0, 2.0] )
#         self.assertAlmostEqual( evaluateSolutionAt( x = -1.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 1.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 3.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x = +1.0, domain = [-1.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 2.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.0, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 1.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  0.5, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 3.0 )
#         self.assertAlmostEqual( evaluateSolutionAt( x =  1.0, domain = [0.0, 1.0], coeff = coeff, solution_basis = basis.evalLagrangeBasis1D ), 2.0 )

class Test_assembleGramMatrix( unittest.TestCase ):
    def test_linear_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 1, 1 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 2, 2 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [1/15, 1/30, -1/60, 0, 0 ], [1/30, 4/15, 1/30, 0, 0], [-1/60, 1/30, 2/15, 1/30, -1/60], [ 0, 0, 1/30, 4/15, 1/30], [0, 0, -1/60, 1/30, 1/15] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [ 0.03809524,  0.02946429, -0.01071429,  0.00565476,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [ 0.02946429,  0.19285714, -0.02410714, -0.01071429,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [-0.01071429, -0.02410714,  0.19285714,  0.02946429,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [ 0.00565476, -0.01071429,  0.02946429,  0.07619048,  0.02946429, -0.01071429,  0.00565476 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000,  0.02946429,  0.19285714, -0.02410714, -0.01071429 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000, -0.01071429, -0.02410714,  0.19285714,  0.02946429 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000,  0.00565476, -0.01071429,  0.02946429,  0.03809524 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_linear_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 1, 1 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 2, 2 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_gram_matrix = numpy.array( [ [1/10, 1/20, 1/60, 0, 0 ], [1/20, 1/15, 1/20, 0, 0 ], [1/60, 1/20, 1/5, 1/20, 1/60], [0, 0, 1/20, 1/15, 1/20], [0, 0, 1/60, 1/20, 1/10] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_gram_matrix = numpy.array( [ [1/14, 1/28, 1/70, 1/280, 0, 0, 0 ], [1/28, 3/70, 9/280, 1/70, 0, 0, 0 ], [1/70, 9/280, 3/70, 1/28, 0, 0, 0 ], [1/280, 1/70, 1/28, 1/7, 1/28, 1/70, 1/280], [0, 0, 0, 1/28, 3/70, 9/280, 1/70], [0, 0, 0, 1/70, 9/280, 3/70, 1/28], [0, 0, 0, 1/280, 1/70, 1/28, 1/14 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

class Test_assembleForceVector( unittest.TestCase ):
    def test_lagrange_const_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, numpy.pi / 8.0, 3.0 * numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, numpy.pi / 16.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_linear_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: 2*x + numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ 0.20468287, 0.62654862, 0.73904862, 0.51769908, 0.81404862, 0.92654862, 0.31301621 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_quadratic_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: x**2.0
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ 1.04166667e-03, 0, 2.81250000e-02, 3.33333333e-02, 6.56250000e-02, 1.50000000e-01, 5.52083333e-02 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_lagrange_const_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 4.0, numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 8.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_linear_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: 2*x + numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ 0.41769908, 0.44269908, 0.46769908, 1.03539816, 0.56769908, 0.59269908, 0.61769908 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_quadratic_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: x**2.0
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ 1/480, 1/160, 1/80, 1/15, 1/16, 13/160, 49/480 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
