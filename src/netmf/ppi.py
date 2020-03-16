import logging
import os

import networkx as nx
import pandas as pd

from netmf import netmf_large_nx

HIPPIE_PATH = os.path.join(os.path.expanduser('~'), 'hippie.txt')


def get_graph() -> nx.Graph:
    df = pd.read_csv(
        HIPPIE_PATH,
        sep='\t',
        usecols=[0, 2],
        names=['source', 'target'],
    )
    df = df[df['source'].notna() & df['target'].notna()]
    return nx.from_pandas_edgelist(df)


def main():
    graph = get_graph()
    embedding = netmf_large_nx(graph)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
