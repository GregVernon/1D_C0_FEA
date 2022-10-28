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

def plotCompareGoldTestSolution( gold_coeff, test_coeff, domain, solution_basis ):
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], domain, gold_coeff, solution_basis )
        yt[i] = evaluateSolutionAt( x[i], domain, test_coeff, solution_basis )
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
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
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
        # gold_sol_coeff = numpy.array( [ (12*(numpy.pi**2 - 10))/(numpy.pi**3), -(6*(3*numpy.pi**2 - 40))/(numpy.pi**3), (12*(numpy.pi**2 - 10))/(numpy.pi**3)] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [0, 1], solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )
        
    def test_erfc_target( self ):
        # print( "ERFC TEST" )
        target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
        domain = [ -2, 2 ]
        degree = [3]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        # gold_sol_coeff = numpy.array( [ 1.8962208131568558391841630949727, 2.6917062016799657617278998883219, -0.69170620167996576172789988832194, 0.10377918684314416081583690502732] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [-2, 2], solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )
    
    def test_exptx_target( self ):
        # print( "EXPT TEST" )
        target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
        domain = [ -1, 1 ]
        degree = [5]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        # gold_sol_coeff = ( [ -0.74841381974620419634327921170757, -3.4222814978197825394922980704166, 7.1463655364038831935841354617843, -2.9824200396151998304868767455064, 1.6115460899636204992283970407553, 0.87876479932866366847320748048494 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [-1, 1], solution_basis )
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

# class Test_assembleGramMatrix( unittest.TestCase ):
#     def test_quadratic_legendre( self ):
#         test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = basis.evalLegendreBasis1D )
#         gold_gram_matrix = numpy.array( [ [1.0, 0.0, 0.0], [0.0, 1.0/3.0, 0.0], [0.0, 0.0, 0.2] ] )
#         self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
#     def test_cubic_legendre( self ):
#         test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 3, solution_basis = basis.evalLegendreBasis1D )
#         gold_gram_matrix = numpy.array( [ [1.0, 0.0, 0.0, 0.0], [0.0, 1.0/3.0, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0], [ 0.0, 0.0, 0.0, 1.0/7.0] ] )
#         self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

#     def test_linear_bernstein( self ):
#         test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 1, solution_basis = basis.evalBernsteinBasis1D )
#         gold_gram_matrix = numpy.array( [ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ] )
#         self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

#     def test_quadratic_bernstein( self ):
#         test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = basis.evalBernsteinBasis1D )
#         gold_gram_matrix = numpy.array( [ [0.2, 0.1, 1.0/30.0], [0.1, 2.0/15.0, 0.1], [1.0/30.0, 0.1, 0.2] ] )
#         self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
#     def test_cubic_bernstein( self ):
#         test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 3, solution_basis = basis.evalBernsteinBasis1D )
#         gold_gram_matrix = numpy.array( [ [1.0/7.0, 1.0/14.0, 1.0/35.0, 1.0/140.0], [1.0/14.0, 3.0/35.0, 9.0/140.0, 1.0/35.0], [1.0/35.0, 9.0/140.0, 3.0/35.0, 1.0/14.0], [ 1.0/140.0, 1.0/35.0, 1.0/14.0, 1.0/7.0] ] )
#         self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

#     def test_linear_lagrange( self ):
#         test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 1, solution_basis = basis.evalLagrangeBasis1D )
#         gold_gram_matrix = numpy.array( [ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ] )
#         self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

#     def test_quadratic_lagrange( self ):
#         test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = basis.evalLagrangeBasis1D )
#         gold_gram_matrix = numpy.array( [ [2.0/15.0, 1.0/15.0, -1.0/30.0], [1.0/15.0, 8.0/15.0, 1.0/15.0], [-1.0/30.0, 1.0/15.0, 2.0/15.0] ] )
#         self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
#     def test_cubic_lagrange( self ):
#         test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 3, solution_basis = basis.evalLagrangeBasis1D )
#         gold_gram_matrix = numpy.array( [ [8.0/105.0, 33.0/560.0, -3.0/140.0, 19.0/1680.0], [33.0/560.0, 27.0/70.0, -27.0/560.0, -3.0/140.0], [-3.0/140.0, -27.0/560.0, 27.0/70.0, 33/560.0], [ 19.0/1680.0, -3.0/140.0, 33.0/560.0, 8.0/105.0] ] )
#         self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

# class Test_assembleForceVector( unittest.TestCase ):
#     def test_legendre_const_force_fun( self ):
#         test_force_vector = assembleForceVector( target_fun = lambda x: numpy.pi, domain = [0, 1], degree = 1, solution_basis = basis.evalLegendreBasis1D )
#         gold_force_vector = numpy.array( [ numpy.pi, 0.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
#     def test_legendre_linear_force_fun( self ):
#         test_force_vector = assembleForceVector( target_fun = lambda x: 2*x + numpy.pi, domain = [0, 1], degree = 1, solution_basis = basis.evalLegendreBasis1D )
#         gold_force_vector = numpy.array( [ numpy.pi + 1.0, 1.0/3.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
#     def test_legendre_quadratic_force_fun( self ):
#         test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 1, solution_basis = basis.evalLegendreBasis1D )
#         gold_force_vector = numpy.array( [ 1.0/3.0, 1.0/6.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
#         test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 2, solution_basis = basis.evalLegendreBasis1D )
#         gold_force_vector = numpy.array( [ 1.0/3.0, 1.0/6.0, 1.0/30.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_lagrange_const_force_fun( self ):
#         test_force_vector = assembleForceVector( target_fun = lambda x: numpy.pi, domain = [0, 1], degree = 1, solution_basis = basis.evalLagrangeBasis1D )
#         gold_force_vector = numpy.array( [ numpy.pi / 2.0, numpy.pi / 2.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
#     def test_lagrange_linear_force_fun( self ):
#         test_force_vector = assembleForceVector( target_fun = lambda x: 2*x + numpy.pi, domain = [0, 1], degree = 1, solution_basis = basis.evalLagrangeBasis1D )
#         gold_force_vector = numpy.array( [ numpy.pi/2.0 + 1.0/3.0, numpy.pi/2.0 + 2.0/3.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
#     def test_lagrange_quadratic_force_fun( self ):
#         test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 1, solution_basis = basis.evalLagrangeBasis1D )
#         gold_force_vector = numpy.array( [ 1.0/12.0, 1.0/4.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
#         test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 2, solution_basis = basis.evalLagrangeBasis1D )
#         gold_force_vector = numpy.array( [ -1.0/60.0, 1.0/5.0, 3.0/20.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_bernstein_const_force_fun( self ):
#         test_force_vector = assembleForceVector( target_fun = lambda x: numpy.pi, domain = [0, 1], degree = 1, solution_basis = basis.evalBernsteinBasis1D )
#         gold_force_vector = numpy.array( [ numpy.pi / 2.0, numpy.pi / 2.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
#     def test_bernstein_linear_force_fun( self ):
#         test_force_vector = assembleForceVector( target_fun = lambda x: 2*x + numpy.pi, domain = [0, 1], degree = 1, solution_basis = basis.evalBernsteinBasis1D )
#         gold_force_vector = numpy.array( [ numpy.pi/2.0 + 1.0/3.0, numpy.pi/2.0 + 2.0/3.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
#     def test_bernstein_quadratic_force_fun( self ):
#         test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 1, solution_basis = basis.evalBernsteinBasis1D )
#         gold_force_vector = numpy.array( [ 1.0/12.0, 1.0/4.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
#         test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 2, solution_basis = basis.evalBernsteinBasis1D )
#         gold_force_vector = numpy.array( [ 1.0/30.0, 1.0/10.0, 1.0/5.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
