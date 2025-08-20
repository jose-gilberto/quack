"""
[1] Pérez-Gállego, P., Castano, A., Quevedo, J. R., & del Coz, J. J. (2019). 
    Dynamic ensemble selection for quantification tasks. Information Fusion, 45, 1-15.
    <https://www.sciencedirect.com/science/article/pii/S1566253517303652>
"""

# Some of these datasets are build with one-versus-all strategy and others are
# naturally binary problems. These are the datasets used in [1].
UCI_BINARY_DATASETS = [
    'acute.a', 'acute.b',
    'balance.1', 'balance.2', 'balance.3',
    'breast-cancer',
    'cmc.1', 'cmc.2', 'cmc.3',
    'ctg.1', 'ctg.1', 'ctg.3',
    'diabetes',
    'german',
    'haberman',
    'ionosphere',
    'iris.1', 'iris.2', 'iris.3',
    'mammographic',
    'pageblocks.5', # <- why only .5 as a class
    'phoneme',
    'semeion',
    'sonar',
    'spambase',
    'spectf',
    'tictactoe',
    'transfusion',
    'wdbc',
    'wine.1', 'wine.2', 'wine.3',
    'wine-q-red',
    'wine-q-white',
    'yeast',
]
