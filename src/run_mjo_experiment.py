from .runner import run_config


CONFIG_PATH = "configs/mjo.yaml"


def main():
    run_config(CONFIG_PATH)


if __name__ == "__main__":
    main()
