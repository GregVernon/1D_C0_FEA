import os
import sys
import unittest
import sympy
import numpy
import argparse
import itertools
import matplotlib
from matplotlib import pyplot as plt
from pyinstrument import Profiler

if __name__ == "src.cubitApproxGalerkin":
    from src import splineApproxGalerkin
    from src import bext
    from src import uspline
elif __name__ == "cubitApproxGalerkin":
    import splineApproxGalerkin
    import bext
    import uspline
elif __name__ == "__main__":
    file_path = os.path.realpath( __file__ )
    file_path = os.path.split( file_path )[0]
    sys.path.append( file_path + "/src" )
    import splineApproxGalerkin
    import bext
    import uspline

if __name__ == "CubitPythonInterpreter_2":
    # We are running within the Coreform Cubit application, cubit python module is already available
    pass
else:
    if "linux" in sys.platform:
        sys.path.append("/opt/Coreform-Cubit-2022.10/bin")
    elif "darwin" in sys.platform:
        pass
    elif "win" in sys.platform:
        sys.path.append( r"C:\Program Files\Coreform Cubit 2022.10\bin" )
    import cubit
    cubit.init(["cubit", "-nog"])

def main( target_fun, spline_space ):
    filename = "temp_uspline"
    uspline.make_uspline_mesh( spline_space, filename )
    uspline_bext = bext.readBEXT( filename + ".json" )
    sol = splineApproxGalerkin.computeSolution( target_fun, uspline_bext )
    splineApproxGalerkin.plotCompareFunToTestSolution( target_fun, sol, uspline_bext )
    return sol

def prepareCommandInputs( target_fun_str, domain, degree, continuity ):
    spline_space = { "domain": domain, "degree": degree, "continuity": continuity }
    target_fun = sympy.parsing.sympy_parser.parse_expr( target_fun_str )
    target_fun = sympy.lambdify( sympy.symbols( "x", real = True ), target_fun )
    return target_fun, spline_space

def parseCommandLineArguments( cli_args ):
    parser = argparse.ArgumentParser()
    parser.add_argument( "--function", "-f",   nargs = 1,   type = str,   required = True )
    parser.add_argument( "--domain", "-d",     nargs = 2,   type = float, required = True )
    parser.add_argument( "--degree", "-p",     nargs = '+', type = int,   required = True )
    parser.add_argument( "--continuity", "-c", nargs = '+', type = int,   required = True )
    args = parser.parse_args( cli_args.split() )
    return args.function[0], args.domain, args.degree, args.continuity

class Test_h_convergence_rates( unittest.TestCase ):
    def test_sin_linear( self ):
        target_fun_str = "sin(pi*x)"
        domain = [ 0, 1 ]
        degree = 1
        continuity = degree - 1
        num_iter = 7
        num_elems = 2**numpy.arange( 1, num_iter+1 )
        abs_err = numpy.zeros( num_iter )
        rel_err = numpy.zeros( num_iter )
        for i in range( 0, num_iter ):
            degree_list = [ degree ]*num_elems[i]
            continuity_list = list( itertools.chain( *[[-1], [continuity]*(num_elems[i]-1), [-1]] ) )
            target_fun, spline_space = prepareCommandInputs( target_fun_str, domain, degree_list, continuity_list )
            sol = main( target_fun, spline_space )
            uspline = bext.readBEXT( "temp_uspline.json" )
            abs_err[i], rel_err[i] = splineApproxGalerkin.computeFitError( target_fun, sol, uspline )
        fig, ax = plt.subplots()
        print( num_elems, rel_err )
        ax.loglog( num_elems, rel_err, linewidth=2.0, color = [0, 0, 0], marker = "o" )
        ax.grid( visible = True, which = "both" )
        plt.show()
    
    def test_sin_quadratic( self ):
        target_fun_str = "sin(pi*x)"
        domain = [ -1, 1 ]
        degree = 2
        continuity = degree - 1
        num_iter = 7
        num_elems = 2**numpy.arange( 1, num_iter+1 )
        abs_err = numpy.zeros( num_iter )
        rel_err = numpy.zeros( num_iter )
        for i in range( 0, num_iter ):
            degree_list = [ degree ]*num_elems[i]
            continuity_list = list( itertools.chain( *[[-1], [continuity]*(num_elems[i]-1), [-1]] ) )
            target_fun, spline_space = prepareCommandInputs( target_fun_str, domain, degree_list, continuity_list )
            sol = main( target_fun, spline_space )
            uspline = bext.readBEXT( "temp_uspline.json" )
            abs_err[i], rel_err[i] = splineApproxGalerkin.computeFitError( target_fun, sol, uspline )
        fig, ax = plt.subplots()
        print( num_elems, rel_err )
        ax.loglog( num_elems, rel_err, linewidth=2.0, color = [0, 0, 0], marker = "o" )
        ax.grid( visible = True, which = "both" )
        plt.show()

    def test_sin_quadratic_C0( self ):
        target_fun_str = "sin(pi*x)"
        domain = [ 0, 1 ]
        degree = 2
        continuity = 0
        num_iter = 7
        num_elems = 2**numpy.arange( 1, num_iter+1 )
        abs_err = numpy.zeros( num_iter )
        rel_err = numpy.zeros( num_iter )
        for i in range( 0, num_iter ):
            degree_list = [ degree ]*num_elems[i]
            continuity_list = list( itertools.chain( *[[-1], [continuity]*(num_elems[i]-1), [-1]] ) )
            target_fun, spline_space = prepareCommandInputs( target_fun_str, domain, degree_list, continuity_list )
            sol = main( target_fun, spline_space )
            uspline = bext.readBEXT( "temp_uspline.json" )
            abs_err[i], rel_err[i] = splineApproxGalerkin.computeFitError( target_fun, sol, uspline )
            # print(abs_err[i], rel_err[i])
            splineApproxGalerkin.plotCompareFunToTestSolution( target_fun, sol, uspline )
        fig, ax = plt.subplots()
        print( num_elems, rel_err )
        ax.loglog( num_elems, rel_err, linewidth=2.0, color = [0, 0, 0], marker = "o" )
        ax.grid( visible = True, which = "both" )
        plt.show()

if __name__ == "__main__":
    target_fun_str, domain, degree, continuity = parseCommandLineArguments( sys.argv[-1] )
    target_fun, spline_space = prepareCommandInputs( target_fun_str, domain, degree, continuity )
    main( target_fun, spline_space )