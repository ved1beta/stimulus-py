#!/usr/bin/env python3

import argparse

import yaml
from jsonschema import ValidationError, validate


def get_args():
    """Get the arguments when using from the commandline

    This script reads a Json with very defined structure and creates all the Json files ready to be passed to
    the stimulus package.

    The structure of the Json is described here -> TODO paste here link to documentation.
    This Json and it's structure summarize how to generate all the transform - split and respective parameters combinations.
    Each resulting Json will hold only one combination of the above three things.

    This script will always generate at least on Json file that represent the combination that does not touch the data (no transform)
    and uses the defalut split behaviour.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-j",
        "--yaml",
        type=str,
        required=True,
        metavar="FILE",
        help="The yaml config file that hold all transform - split - parameter info",
    )
    parser.add_argument(
        "-d",
        "--out_dir",
        type=str,
        required=False,
        nargs="?",
        const="./",
        default="./",
        metavar="DIR",
        help="The output dir where all he json are written to. Output Json will be called split-#[number].json transorm-#[number].json. Default -> ./",
    )

    args = parser.parse_args()
    return args


def main(config_yaml: str, out_dir_path: str) -> str:
    # read the yaml experiment config and load it to dictionary
    yaml_config = {}
    with open(config_yaml) as conf_file:
        yaml_config = yaml.safe_load(conf_file)

    # here the json schema is defined as a dictionary
    schema = {
        "type": "object",
        "properties": {
            "global_params": {
                "type": "object",
                "properties": {
                    "seed": {"type": "integer"},
                },
                "required": ["seed"],
            },
            "columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column_name": {"type": "string"},
                        "column_type": {"type": "string", "enum": ["input", "label"]},
                        "data_type": {"type": "string"},
                        "parsing": {"type": "string"},
                    },
                    "required": ["column_name", "column_type", "data_type", "parsing"],
                },
            },
            "transforms": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "transformation_name": {"type": "string"},
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "column_name": {"type": "string"},
                                        "transformations": {
                                            "oneOf": [
                                                {
                                                    "type": "object",
                                                    "properties": {
                                                        "name": {"type": "string"},
                                                        "params": {},
                                                    },
                                                    "required": ["name", "params"],
                                                },
                                                {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "params": {},
                                                        },
                                                        "required": ["name", "params"],
                                                    },
                                                },
                                            ],
                                        },
                                    },
                                    "required": ["column_name", "transformations"],
                                },
                            },
                        },
                        "required": ["transformation_name", "columns"],
                    },
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "transformation_name": {"type": "string"},
                                "columns": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "column_name": {"type": "string"},
                                            "transformations": {
                                                "oneOf": [
                                                    {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "params": {},
                                                        },
                                                        "required": ["name", "params"],
                                                    },
                                                    {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "name": {"type": "string"},
                                                                "params": {},
                                                            },
                                                            "required": ["name", "params"],
                                                        },
                                                    },
                                                ],
                                            },
                                        },
                                        "required": ["column_name", "transformations"],
                                    },
                                },
                            },
                            "required": ["transformation_name", "columns"],
                        },
                    },
                ],
                "optional": True,
            },
            "split": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "split_method": {"type": "string"},
                        "params": {},
                    },
                    "required": ["split_method", "params"],
                },
                "optional": True,
            },
        },
        "required": ["global_params", "columns"],
    }

    #
    # TODO move all the syster error custom error messages to loggers
    #

    # Define custom error messages for specific validation errors
    custom_messages = {
        ("split",): "Error: 'split' should include 'split_method', 'params'.",
        ("split", "split_method"): "Error: 'split' is present, but 'split_method' is missing.",
        (
            "transforms",
        ): "Error: 'transforms' should include 'transformation_name', 'columns', 'column_name', 'transformations', 'name', 'params' ",
        ("transforms", "transformation_name"): "Error: 'transforms' is present, but 'transformation_name' is missing.",
        ("global_params", "seed"): "Error: 'seed' value should be a integer.",
        ("global_params",): "Error: 'global_params' should include 'seed'",
        ("columns",): "Error: 'columns' should include 'column_name', 'column_type', 'data_type', and 'parsing'.",
    }

    # Check for specific required fields
    required_fields_messages = {
        "column_name": "Error: Each column must include 'column_name'.",
        "column_type": "Error: Each column must include 'column_type'.",
        "data_type": "Error: Each column must include 'data_type'.",
        "parsing": "Error: Each column must include 'parsing'.",
        "transformations": "Error: Each transformation must include 'transformations'.",
        "name": "Error: Each transformation must include 'name'.",
    }

    # here the input yaml is validated against the schema
    try:
        validate(instance=yaml_config, schema=schema)
        print("Validation successful!")

    except ValidationError as e:
        # Access the path to the failed validation
        path = tuple(e.absolute_path)

        # Check for a generic 'params' missing case in transformations
        if "transformations" in path and "params" in str(e.message):
            raise SystemError("Error: 'params' is required for each transformations.")

        # check also for split
        if "split" in path and "params" in str(e.message):
            raise SystemError("Error: 'params' is required for each split_method.")

        # Provide custom message if available
        if path in custom_messages:
            raise SystemError(custom_messages[path])

        # Loop through required fields and provide custom messages
        for field in required_fields_messages:
            if field in path:
                raise SystemError(required_fields_messages[field])

        # Fallback to the default error message if no custom message is available
        raise SystemError(f"Validation error at {path}: {e.message}")


if __name__ == "__main__":
    args = get_args()
    main(args.yaml, args.out_dir)
