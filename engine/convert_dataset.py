import setup
import parse
import os
import pandas as pd
import numpy as np

CHUNK_SIZE = 50_000

if __name__ == "__main__":
    out_path = setup.DATASET_DIR+setup.DATASET_VECTORIZED
    csv_regular = pd.read_csv(setup.DATASET_DIR+setup.DATASET_REGULAR, chunksize=CHUNK_SIZE, 
                    nrows=setup.N_ROWS*setup.DATASET_REGULAR_SIZE, dtype={"Fen": np.string_, "Evaluation": np.string_})
    csv_random = pd.read_csv(setup.DATASET_DIR+setup.DATASET_RANDOM, chunksize=CHUNK_SIZE, 
                    nrows=setup.N_ROWS*setup.DATASET_RANDOM_SIZE, dtype={"Fen": np.string_, "Evaluation": np.string_})
    csv_tactic = pd.read_csv(setup.DATASET_DIR+setup.DATASET_TACTIC, chunksize=CHUNK_SIZE, 
                    nrows=setup.N_ROWS*setup.DATASET_TACTIC_SIZE, dtype={"Fen": np.string_, "Evaluation": np.string_}, usecols=[0, 1])

    if os.path.exists(out_path):
        os.remove(out_path)

    line_cnt = 0
    for csv in [csv_regular, csv_random, csv_tactic]:
        for i, df_chunk in enumerate(csv):
            features = [f"f_{str(x)}" for x in range(1, setup.N_FEATURES+1)]
            df_converted = pd.DataFrame(df_chunk["FEN"].apply(lambda fen_str: parse.fen_to_vector(fen_str)).to_list(), columns=features)
            df_converted["label"] = df_chunk.apply(lambda row: parse.normalize_stockfish_eval(row.FEN, row.Evaluation), axis=1).reset_index(drop=True)

            print(df_converted)
            line_cnt += len(df_converted)
            print(f"Converted lines: {line_cnt}")
            print(f"Writing to disk...")
            df_converted.to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))
            print(f"Done")
