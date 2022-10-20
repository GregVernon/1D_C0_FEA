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
    prev_force_vector = err * numpy.zeros( shape = ( degree + 1 ) )
    num_qp = 0
    tol = 1e-14
    while err > tol:
        force_vector = numpy.zeros( shape = ( degree + 1 ) )
        num_qp += 1
        for i in range( 0, degree + 1 ):
            Ni = lambda x: solution_basis( degree, i, basis.affine_mapping_1D( [-1, 1], [0, 1], x ) )
            integrand = lambda x: Ni( x ) * target_fun( basis.affine_mapping_1D( [-1, 1], domain, x ) )
            force_vector[i] += quadrature.quad( integrand, domain, num_qp )
        if num_qp > 1:
            err = numpy.linalg.norm( prev_force_vector - force_vector )
        prev_force_vector = force_vector
    return force_vector

class Test_computeSolution( unittest.TestCase ):
    def test_polynomial_target( self ):
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        test_sol_coeff = computeSolution( target_fun = target_fun, domain = [0, 1], degree = 2, solution_basis = basis.evalBernsteinBasis1D )
        gold_sol_coeff = numpy.array( [ 1.0 / 20.0, 1.0 / 20.0, -1.0 / 20.0 ] )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
    
    def test_sin_target( self ):
        target_fun = lambda x: numpy.sin( numpy.pi * x )
        test_sol_coeff = computeSolution( target_fun = target_fun, domain = [0, 1], degree = 2, solution_basis = basis.evalBernsteinBasis1D )
        gold_sol_coeff = numpy.array( [ (12*(numpy.pi**2 - 10))/(numpy.pi**3), -(6*(3*numpy.pi**2 - 40))/(numpy.pi**3), (12*(numpy.pi**2 - 10))/(numpy.pi**3)] )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        