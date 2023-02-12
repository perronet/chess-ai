import setup
import parse
import os
import pandas as pd
import numpy as np

CHUNK_SIZE = 50_000

if __name__ == "__main__":
    out_path = setup.DATASET_DIR+setup.DATASET_VECTORIZED

    for i, df_chunk in enumerate(pd.read_csv(setup.DATASET_DIR+setup.DATASET, chunksize=CHUNK_SIZE, nrows=setup.N_ROWS, dtype={"Fen": np.string_, "Evaluation": np.string_})):
        features = [f"f_{str(x)}" for x in range(1, setup.N_FEATURES+1)]
        df_converted = pd.DataFrame(df_chunk["FEN"].apply(lambda fen_str: parse.fen_to_vector(fen_str)).to_list(), columns=features)
        df_converted["label"] = df_chunk.apply(lambda row: parse.normalize_stockfish_eval(row.FEN, row.Evaluation), axis=1).reset_index(drop=True)

        print(df_converted)
        print(f"Converted lines: {(i+1)*CHUNK_SIZE}")
        print(f"Writing to disk...")
        df_converted.to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))
        print(f"Done")
