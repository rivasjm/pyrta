import matplotlib.pyplot as plt
import pandas as pd


def chart(excel_file):
    df = pd.read_excel(excel_file, index_col=0)
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    fig.tight_layout()
    fig.savefig("edf-local.png")


if __name__ == '__main__':
    chart("edf_local_scheds.xlsx")
