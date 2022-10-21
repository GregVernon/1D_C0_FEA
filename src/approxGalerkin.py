import unittest
import numpy
import scipy

if __name__ == "src.approxGalerkin":
    from src import basis
    from src import mesh
    from src import quadrature
elif __name__ == "approxGalerkin":
    import basis
    import mesh
    import quadrature

def computeSolution( target_fun, domain, degree, solution_basis ):
    gram_matrix = assembleGramMatrix( domain, degree, solution_basis )
    force_vector = assembleForceVector( target_fun, domain, degree, solution_basis )
    coeff = numpy.linalg.solve( gram_matrix, force_vector )
    return coeff

def assembleGramMatrix( domain, degree, solution_basis ):
    gram_matrix = numpy.zeros( shape = ( degree + 1, degree + 1 ) )
    num_qp = int( numpy.ceil( ( degree**2 + 1 ) / 2.0 ) )
    for i in range( 0, degree + 1):
        Ni = lambda x: solution_basis( degree, i, basis.affine_mapping_1D( [-1, 1], [0, 1], x ) )
        for j in range( 0, degree + 1 ):
            Nj = lambda x: solution_basis( degree, j, basis.affine_mapping_1D( [-1, 1], [0, 1], x ) )
            integrand = lambda x: Ni( x ) * Nj( x )
            gram_matrix[i,j] += quadrature.quad( integrand, domain, num_qp )
    return gram_matrix

def assembleForceVector( target_fun, domain, degree, solution_basis ):
    err = float("inf")
    prev_force_vector = err * numpy.ones( shape = ( degree + 1 ) )
    num_qp = 0
    tol = 1e-5
    while ( err > tol ) and ( num_qp <= 15 ):
        force_vector = numpy.zeros( shape = ( degree + 1 ) )
        num_qp += 1
        for i in range( 0, degree + 1 ):
            Ni = lambda x: solution_basis( degree, i, basis.affine_mapping_1D( [-1, 1], [0, 1], x ) )
            integrand = lambda x: Ni( x ) * target_fun( basis.affine_mapping_1D( [-1, 1], domain, x ) )
            force_vector[i] += quadrature.quad( integrand, domain, num_qp )
        if num_qp > 1:
            err = numpy.linalg.norm( prev_force_vector - force_vector ) / numpy.linalg.norm( prev_force_vector, 1 )
        prev_force_vector = force_vector
    return force_vector

def evaluateSolutionAt( x, domain, coeff, solution_basis ):
    xi = basis.affine_mapping_1D( domain, [0, 1], x )
    degree = len( coeff ) - 1
    y = 0.0
    for n in range( 0, len( coeff ) ):
        y += coeff[n] * solution_basis( degree = degree, basis_idx = n, variate = xi )
    return y

def computeFitError( gold_coeff, test_coeff, domain, solution_basis ):
    err_fun = lambda x: abs( evaluateSolutionAt( x, domain, gold_coeff, solution_basis ) - evaluateSolutionAt( x, domain, test_coeff, solution_basis ) )
    abs_err, _ = scipy.integrate.quad( err_fun, domain[0], domain[1], epsrel = 1e-12, limit = 1000 )
    return abs_err

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

class Test_computeSolution( unittest.TestCase ):
    def test_cubic_polynomial_target( self ):
        # print( "POLY TEST" )
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        domain = [ 0, 1 ]
        degree = 2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis =solution_basis )
        gold_sol_coeff = numpy.array( [ 1.0 / 20.0, 1.0 / 20.0, -1.0 / 20.0 ] )
        fit_err = computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-12 )

    def test_sin_target( self ):
        # print( "SIN TEST" )
        target_fun = lambda x: numpy.sin( numpy.pi * x )
        domain = [ 0, 1 ]
        degree = 2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ (12*(numpy.pi**2 - 10))/(numpy.pi**3), -(6*(3*numpy.pi**2 - 40))/(numpy.pi**3), (12*(numpy.pi**2 - 10))/(numpy.pi**3)] )
        fit_err = computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [0, 1], solution_basis )
        self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-5 )
        
    def test_erfc_target( self ):
        # print( "ERFC TEST" )
        target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
        domain = [ -2, 2 ]
        degree = 3
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ 1.8962208131568558391841630949727, 2.6917062016799657617278998883219, -0.69170620167996576172789988832194, 0.10377918684314416081583690502732] )
        fit_err = computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [-2, 2], solution_basis )
        self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-4 )
    
    def test_exptx_target( self ):
        # print( "EXPT TEST" )
        target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
        domain = [ -1, 1 ]
        degree = 5
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = ( [ -0.74841381974620419634327921170757, -3.4222814978197825394922980704166, 7.1463655364038831935841354617843, -2.9824200396151998304868767455064, 1.6115460899636204992283970407553, 0.87876479932866366847320748048494 ] )
        fit_err = computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [-1, 1], solution_basis )
        self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-2 )
    