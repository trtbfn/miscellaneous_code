import polars as pl
import polars_hash 
import json
import random
import os
# ! import polars_hash / https://github.com/ion-elgreco/polars-hash

tmp_dir = r"F:\repos\word2vec-pytorch\tfree-word2vec\tmp\data"

# Define hyperparameters
config = {     
    # For more then 1 word in context size should be created extra columns 
    # using shift with value 2, -2, 3, -3, ..., n, -n          
    "context_window_size": 1,
    "neg_pair_seed_n": 5,
    "hash_seed_n": 10,
}

ng_cntxt_wrds_cols = [f"neg_context_word_{i}" for i in range(config["neg_pair_seed_n"])]
ng_lbl_vals = [1] + [0 for _ in range(config["neg_pair_seed_n"])]

neg_pair_seeds = [random.randint(0, 2**32) for _ in range(config["neg_pair_seed_n"])]
hash_seeds = [random.randint(0, 2**32) for _ in range(config["hash_seed_n"])]

config["hash_seeds"] = hash_seeds
config["neg_pair_seeds"] = neg_pair_seeds

with open(os.path.join(tmp_dir, "config.json"), "w") as f: 
    json.dump(config, f, indent=2)

def batch_generator(file_list, batch_size=100):
    """Yield successive batches of size `batch_size` from `file_list`."""
    for i in range(0, len(file_list), batch_size):
        yield file_list[i:i+batch_size]

 # Stage 1: produce small pos/neg Parquet files
with open(r"F:\repos\word2vec-pytorch\tfree-word2vec\tmp\pmi\dataset_config.json", "r") as f:
    all_files = json.load(f)["train"]

all_files = [os.path.join(r"F:\repos\word2vec-pytorch\tfree-word2vec\processed_files", file) for file in all_files]

for i, file_batch in enumerate(batch_generator(all_files, batch_size=1500)):
    df = pl.scan_csv(
        source=file_batch,
        has_header=False,
        new_columns=["center_word"], 
    )

    df.select(
        pl.col("center_word").str.split(by=" ").list.explode()
    ).with_columns_seq(
        pl.col("center_word").shift(config["context_window_size"]).alias("left_context"),
        pl.col("center_word").shift(-config["context_window_size"]).alias("right_context")
    ).fill_null("") \
    .with_columns(
        pl.concat_list("left_context", "right_context").alias("context_word")
    ) \
    .drop(["left_context", "right_context"]) \
    .explode("context_word") \
    .filter(pl.col("context_word") != "").with_columns_seq(
        # Negative pair generations
        [
            pl.col("context_word").shuffle(seed=seed).alias(ng_cntxt_col)
            for seed, ng_cntxt_col in zip(neg_pair_seeds, ng_cntxt_wrds_cols)
        ] + [
            pl.lit(ng_lbl_vals).alias("label")
        ]
    ) \
    .with_columns(
        pl.concat_list(["context_word"] + ng_cntxt_wrds_cols).alias("context_word")
    ) \
    .select(["center_word", "context_word", "label"]) \
    .explode("context_word", "label") \
    .with_columns(
        [
            pl.col("center_word").nchash.murmur32(seed=s).cast(pl.UInt32).alias(f"centw_{i}")
            for i, s in enumerate(hash_seeds)
        ] + [
            pl.col("context_word").nchash.murmur32(seed=s).cast(pl.UInt32).alias(f"contw_{i}")
            for i, s in enumerate(hash_seeds)
        ] 
    ). \
    collect() \
    .write_parquet(
        f"F:/repos/word2vec-pytorch/tfree-word2vec/tmp/data/batch_pairs_{i}.parquet"
    )