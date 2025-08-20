from quack.datasets._uci_datasets import UCI_BINARY_DATASETS
from quack.utils import get_quack_home


def _fetch_uci_binary(dataset: str, data_home: str = None, normalize: bool = True, verbose: bool = False) -> any:
    assert dataset in UCI_BINARY_DATASETS, (
        f'<{dataset}> does not match any known name from the available binary UCI datasets.'
    )

    if data_home is None:
        data_home = get_quack_home()

    IDENTIFIERS = {
        "acute.a": 184,
        "acute.b": 184,
        "balance.1": 12,
        "balance.2": 12,
        "balance.3": 12,
        "breast-cancer": 15,
        "cmc.1": 30,
        "cmc.2": 30,
        "cmc.3": 30,
        # "ctg.1": ,  # not python importable
        # "ctg.2": ,  # not python importable
        # "ctg.3": ,  # not python importable
        # "german": ,  # not python importable
        "haberman": 43,
        "ionosphere": 52,
        "iris.1": 53,
        "iris.2": 53,
        "iris.3": 53,
        "mammographic": 161,
        "pageblocks.5": 78,
        # "semeion": ,  # not python importable
        "sonar": 151,
        "spambase": 94,
        "spectf": 96,
        "tictactoe": 101,
        "transfusion": 176,
        "wdbc": 17,
        "wine.1": 109,
        "wine.2": 109,
        "wine.3": 109,
        "wine-q-red": 186,
        "wine-q-white": 186,
        "yeast": 110,
    }

    DATASETS_NAMES = {
        "acute.a": "Acute Inflammations (urinary bladder)",
        "acute.b": "Acute Inflammations (renal pelvis)",
        "balance.1": "Balance Scale Weight & Distance Database (left)",
        "balance.2": "Balance Scale Weight & Distance Database (balanced)",
        "balance.3": "Balance Scale Weight & Distance Database (right)",
        "breast-cancer": "Breast Cancer Wisconsin (Original)",
        "cmc.1": "Contraceptive Method Choice (no use)",
        "cmc.2": "Contraceptive Method Choice (long term)",
        "cmc.3": "Contraceptive Method Choice (short term)",
        "ctg.1": "Cardiotocography Data Set (normal)",
        "ctg.2": "Cardiotocography Data Set (suspect)",
        "ctg.3": "Cardiotocography Data Set (pathologic)",
        "german": "Statlog German Credit Data",
        "haberman": "Haberman's Survival Data",
        "ionosphere": "Johns Hopkins University Ionosphere DB",
        "iris.1": "Iris Plants Database(x)",
        "iris.2": "Iris Plants Database(versicolour)",
        "iris.3": "Iris Plants Database(virginica)",
        "mammographic": "Mammographic Mass",
        "pageblocks.5": "Page Blocks Classification (5)",
        "semeion": "Semeion Handwritten Digit (8)",
        "sonar": "Sonar, Mines vs. Rocks",
        "spambase": "Spambase Data Set",
        "spectf": "SPECTF Heart Data",
        "tictactoe": "Tic-Tac-Toe Endgame Database",
        "transfusion": "Blood Transfusion Service Center Data Set",
        "wdbc": "Wisconsin Diagnostic Breast Cancer",
        "wine.1": "Wine Recognition Data (1)",
        "wine.2": "Wine Recognition Data (2)",
        "wine.3": "Wine Recognition Data (3)",
        "wine-q-red": "Wine Quality Red (6-10)",
        "wine-q-white": "Wine Quality White (6-10)",
        "yeast": "Yeast",
    }

    pos_class = {
        "acute.a": "yes",
        "acute.b": "yes",
        "balance.1": "L",
        "balance.2": "B",
        "balance.3": "R",
        "breast-cancer": 2,
        "cmc.1": 1,
        "cmc.2": 2,
        "cmc.3": 3,
        "ctg.1": 1,  # 1==Normal
        "ctg.2": 2,  # 2==Suspect
        "ctg.3": 3,  # 3==Pathologic
        "german": 1,
        "haberman": 2,
        "ionosphere": "b",
        "iris.1": "Iris-setosa",  # 1==Setosa
        "iris.2": "Iris-versicolor",  # 2==Versicolor
        "iris.3": "Iris-virginica",  # 3==Virginica
        "mammographic": 1,
        "pageblocks.5": 5,  # 5==block "graphic"
        "semeion": 1,
        "sonar": "R",
        "spambase": 1,
        "spectf": 0,
        "tictactoe": "negative",
        "transfusion": 1,
        "wdbc": "M",
        "wine.1": 1,
        "wine.2": 2,
        "wine.3": 3,
        "wine-q-red": 1,
        "wine-q-white": 1,
        "yeast": "NUC",
    }

