import argparse

from tests import part_a, part_b, part_c, part_d

PARTS = {
    "a": part_a,
    "b": part_b,
    "c": part_c,
    "d": part_d,
}


def main():
    parser = argparse.ArgumentParser(description="Run a specific project part.")
    parser.add_argument(
        "part",
        choices=sorted(PARTS.keys()),
        help="Which part to run (a, b, c, or d).",
    )
    args = parser.parse_args()
    PARTS[args.part]()


if __name__ == "__main__":
    main()
