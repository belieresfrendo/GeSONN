# imports
import torch

# local imports
from gesonn.ana1Tests import poissonTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

if __name__ == "__main__":
    circleDict = {
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "benchmark_circle",
        "symplecto_name": None,
        "to_be_trained": True,
        "boundary_condition": "homogeneous_dirichlet",
    }

    donutDict = {
        "rho_min": 0.5,
        "rho_max": 1,
        "file_name": "benchmark_donut",
        "symplecto_name": None,
        "to_be_trained": True,
        "boundary_condition": "homogeneous_dirichlet",
    }

    ellipseDict = {
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "benchmark_ellipse",
        "symplecto_name": "ellipse_benchmark",
        "to_be_trained": True,
        "boundary_condition": "homogeneous_dirichlet",
    }

    donutEllipseDict = {
        "rho_min": 0.5,
        "rho_max": 1,
        "file_name": "benchmark_donut_ellipse",
        "symplecto_name": "ellipse_benchmark",
        "to_be_trained": True,
        "boundary_condition": "homogeneous_dirichlet",
    }

    sourceList = ["one", "exp"]
    dictsList = [circleDict, donutDict, ellipseDict, donutEllipseDict]

    for source_term in sourceList:

        if source_term != sourceList[0]:
            for simuDict in dictsList:
                simuDict["file_name"] = simuDict["file_name"][:-11]
        for simuDict in dictsList:
            simuDict["file_name"] = simuDict["file_name"] + "_source_" + source_term
            simuDict["source_term"] = source_term

        testsDict = {
            "circle": circleDict,
            "donut": donutDict,
            "ellipse": ellipseDict,
            "donut_ellipse": donutEllipseDict,
        }
        poissonTest.main_poisson_test(testsDict, source_term)
