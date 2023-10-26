import examples
import mast.mast_tools as mast

if __name__ == '__main__':
    system = examples.get_simple_gpu()
    input = "example.txt"
    mast.export(system, input)
    mast.run(mast.MastAnalysis.OFFSET_PR, mast.MastAssignment.NONE, input, print_output=True, verbose=True)