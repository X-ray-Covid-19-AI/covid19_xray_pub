from argparse import ArgumentParser
from dnn import train, eval


def get_cml_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='subprogram',
                                       help="Call with different runmodes: train (calls dnn.train), eval (calls dnn.eval")

    train_parser = subparsers.add_parser('train')
    train.fill_parser(train_parser)

    eval_parser = subparsers.add_parser('eval')
    eval.fill_parser(eval_parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_cml_args()

    if args.subprogram == "train":
        train.run(args)

    if args.subprogram == "eval":
        eval.run(args)
