import sys
import torch
from argparse import ArgumentParser

from base_experiment import BaseExperiment, BaseExperimentArgs


def parse_arguments(argv):
    parser = ArgumentParser()

    parser.add_argument('--N', type=int, default=100, help="The total number of sequences to be generated during the experiment.")
    parser.add_argument('--batch_size', type=int, default=10, help="The number of sequences generated per batch.")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of each generated sequence.")
    parser.add_argument('--top_k', type=int, default=40, help="Parameter in sequence generation specifying the number of top probable vocabulary tokens to consider at each step.")
    parser.add_argument('--top_p', type=float, default=1.0, help="Parameter in sequence generation for cumulative probability cutoff in the token selection process.")

    parser.add_argument('--checkpoint', type=str, default="bigcode/starcoderbase-1b", help="The name of the model, extracted from the checkpoint path.")
    parser.add_argument('--filelog', type=str, default="base-exp-log-v1.txt", help="The file name for logging.")
    parser.add_argument('--filetable', type=str, default="base-exp-table-v1.csv", help="The file name for logging.")

    return parser.parse_args(argv)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_args = BaseExperimentArgs(
        N=args.N, 
        batch_size=args.batch_size, 
        seq_len=args.seq_len, 
        top_k=args.top_k, 
        top_p=args.top_p, 
        checkpoint=args.checkpoint,
        device=device
    )
    experiment = BaseExperiment(
        filelog=args.filelog,
        filetable=args.filetable, 
        args=experiment_args,
    )
    experiment.setup()
    experiment.run()


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
