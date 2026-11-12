#---------------------------------------------------#
#    The answer keys were extracted manually        #
#---------------------------------------------------#

import pandas as pd
import re
import pathlib

# Repository root
root = pathlib.Path(__file__).resolve().parent.parent

answer_key_2014 = {
    "id_prova": "2014",
    "Nquestao": list(range(1, 101)),
    "gabarito": [
        'E','C','X','X','D','B','B','E','E','X',   # 1-10
        'E','E','A','C','D','D','B','X','B','A',   # 11-20
        'E','B','A','A','B','E','E','C','X','D',   # 21-30
        'A','B','C','X','E','B','A','D','A','E',   # 31-40
        'A','E','E','X','E','C','B','B','A','C',   # 41-50
        'E','A','B','D','E','B','B','D','A','A',   # 51-60
        'A','D','E','A','B','A','C','C','D','C',   # 61-70
        'E','C','X','A','E','E','D','B','A','A',   # 71-80
        'X','D','D','C','C','A','E','E','X','D',   # 81-90
        'D','C','D','E','X','A','C','E','C','A',   # 91-100
    ],
}

answer_key_2015_1 = {
    "id_prova": "2015_1",
    "Nquestao": list(range(1, 101)),
    "gabarito": [
        'B','B','C','B','D','C','D','E','E','X',   # 1-10
        'C','D','A','B','A','A','A','B','E','C',   # 11-20
        'E','A','A','D','X','A','D','B','A','X',   # 21-30
        'B','X','D','E','D','E','C','C','B','E',   # 31-40
        'D','D','E','X','A','B','B','B','C','B',   # 41-50
        'B','A','C','A','E','X','A','B','B','D',   # 51-60
        'D','B','A','X','A','B','C','C','X','C',   # 61-70
        'E','D','D','A','D','B','X','E','B','B',   # 71-80
        'A','A','A','C','X','A','X','D','C','E',   # 81-90
        'D','E','A','C','D','B','C','C','C','C',   # 91-100
    ],
}

answer_key_2015_2 = {
    "id_prova": "2015_2",
    "Nquestao": list(range(1, 101)),
    "gabarito": [
        'D','A','A','C','D','E','D','D','C','E',   # 1-10
        'B','D','E','D','A','C','A','C','C','D',   # 11-20
        'C','E','A','A','B','E','C','B','E','B',   # 21-30
        'A','A','E','D','D','C','D','A','D','C',   # 31-40
        'C','D','A','D','B','E','E','B','A','E',   # 41-50
        'D','D','E','C','C','D','A','B','D','A',   # 51-60
        'E','D','B','A','B','D','B','B','E','D',   # 61-70
        'B','C','A','A','D','A','C','A','A','C',   # 71-80
        'D','C','A','E','C','D','E','E','D','D',   # 81-90
        'C','B','B','A','C','A','C','B','D','B',   # 91-100
    ],
}

answer_key_2014 = pd.DataFrame(answer_key_2014)
answer_key_2015_1 = pd.DataFrame(answer_key_2015_1)
answer_key_2015_2 = pd.DataFrame(answer_key_2015_2)


answer_keys = pd.concat([answer_key_2014, answer_key_2015_1, answer_key_2015_2])

exams = pd.read_csv(root / "data" / "interim" / "provas_tjdft.csv")

exams_with_answer_keys = exams.merge(answer_keys, on = ["id_prova","Nquestao"])
exams_with_answer_keys.to_csv(root / 'data' / 'processed' / 'provas_tjdft_com_gabarito.csv', index=False, encoding='utf-8')



