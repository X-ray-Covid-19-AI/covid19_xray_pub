import pandas as pd
from typing import List


def vanilla_ensemble_df(df_list: List[pd.DataFrame], output_path):
    df = pd.concat(df_list)
    new_df = df.groupby(['image_name']).agg({'gt': 'first',
                                    'net_prediction': lambda x: x.value_counts().index[0],
                                    'corona_confidence': 'mean',
                                    'no_corona_confidence': 'mean',
                                    'mistake': lambda x: x.value_counts().index[0]})
    new_df.to_csv(output_path)


if __name__ == '__main__':
    path = r"C:\Daniel_private\Coronavirus\classification_by_xray_and_CT\code_test\ensemble_test\results.csv"
    df1 = pd.read_csv(r"C:\Daniel_private\Coronavirus\classification_by_xray_and_CT\code_test\ensemble_test\df1.csv")
    df2 = pd.read_csv(r"C:\Daniel_private\Coronavirus\classification_by_xray_and_CT\code_test\ensemble_test\df2.csv")
    df3 = pd.read_csv(r"C:\Daniel_private\Coronavirus\classification_by_xray_and_CT\code_test\ensemble_test\df3.csv")

    vanilla_ensemble_df([df1, df2, df3], path)