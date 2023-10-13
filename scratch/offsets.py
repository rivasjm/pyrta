import examples
import mast
from mast_adapters import export
from analysis import repr_wcrts

if __name__ == '__main__':
    system = examples.get_simple_gpu()
    input = "example.txt"
    export(system, input)
    mast.run(mast.MastAnalysis.OFFSET_PR, mast.MastAssignment.NONE, input, print_output=True, verbose=True)