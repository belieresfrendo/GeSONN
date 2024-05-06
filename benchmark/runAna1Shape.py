# imports
import torch

# local imports
from gesonn.ana1Tests import sympTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

if __name__ == "__main__":

    bizaroidDict = {
        "file_name": "benchmark_bizaroid",
        "symplecto_name": "bizaroid",
        "to_be_trained": True,
    }
    avocadoDict = {
        "file_name": "benchmark_avocado",
        "symplecto_name": "avocado",
        "to_be_trained": True,
    }
    galaxyDict = {
        "file_name": "benchmark_galaxy",
        "symplecto_name": "galaxy",
        "to_be_trained": True,
    }
    ellipseDict = {
        "file_name": "benchmark_ellipse",
        "symplecto_name": "ellipse_benchmark",
        "to_be_trained": True,
    }

    testsDict = {
        "bizaroid": bizaroidDict,
        "avocado": avocadoDict,
        "galaxy": galaxyDict,
        "ellipse": ellipseDict,
    }
    sympTest.main_symp_test(testsDict)
