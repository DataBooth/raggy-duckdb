import tomllib


def load_config(config_path: str = "conf/config.toml") -> dict:
    """
    Load the configuration from a TOML file using Python's built-in tomllib.

    Args:
        config_path (str): Path to the toml config file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, "rb") as f:  # open in binary mode
            config = tomllib.load(f)
        return config
    except Exception as e:
        # You can keep using loguru or other logging here if set up
        raise RuntimeError(f"Failed to load config file {config_path}: {e}")
