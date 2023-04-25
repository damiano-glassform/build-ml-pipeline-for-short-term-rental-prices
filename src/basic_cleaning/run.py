#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info("Getting raw data from W&B")
    df = pd.read_csv(artifact_local_path)
    # Drop outliers
    logger.info(
        f"Removing outliers (min price = {args.min_price}"
        f"max price = {args.max_price})"
    )
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, args.output_artifact)
        df.to_csv(export_path, index=False)

        logger.info(f"Uploading {args.output_artifact} to Weights & Biases")
        artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(export_path)
        run.log_artifact(artifact)

        artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", type=str, help="The input artifact", required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type", type=str, help="Type for the output artifact", required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price", type=float, help="Minimum price to consider", required=True
    )

    parser.add_argument(
        "--max_price", type=float, help="Maximum price to consider", required=True
    )

    args = parser.parse_args()

    go(args)
