import os
import sys
# sys.path = [ i for i in sys.path if "Roaming" not in i ]
import sympy
import argparse

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

if __name__ == "__main__":
    target_fun_str, domain, degree, continuity = parseCommandLineArguments( sys.argv[-1] )
    target_fun, spline_space = prepareCommandInputs( target_fun_str, domain, degree, continuity )
    main( target_fun, spline_space )