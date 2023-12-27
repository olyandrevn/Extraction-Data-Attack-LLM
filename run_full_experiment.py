import sys
from argparse import ArgumentParser

from full_experiment import FullExperiment, FullExperimentArgs


def parse_arguments(argv):
    parser = ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default="bigcode/starcoderbase-3b", help="The name of the model, extracted from the checkpoint path.")
    parser.add_argument('--filelog', type=str, default="full-exp-log-v1.txt", help="The file name for logging.")
    parser.add_argument('--base_filetable', type=str, default="base-exp-table-v1.csv", help="Path to the base file containing data for further analysis.")
    parser.add_argument('--full_filetable', type=str, default="full-exp-table-v1.csv", help="The file name for logging.")

    return parser.parse_args(argv)


def main():
    experiment_args = FullExperimentArgs(
        base_filetable=args.base_filetable,
        checkpoint=args.checkpoint,
    )
    experiment = FullExperiment(
        filelog=args.filelog,
        filetable=args.filetable, 
        args=experiment_args,
    )
    experiment.setup()
    experiment.run()


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
