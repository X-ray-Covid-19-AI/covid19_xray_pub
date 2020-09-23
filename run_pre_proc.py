from argparse import ArgumentParser

##########################
# our imports:
from pre_proc import read_in_tabular, main_pre_proc

def get_cml_args():
    
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='subprogram',
                                       help="Call with different runmodes: Tabular (calls read_in_tabular)\n")

    tabular_parser = subparsers.add_parser('Tabular')
    read_in_tabular.fill_parser(tabular_parser)

    pre_proc_parser = subparsers.add_parser('pre_proc')
    main_pre_proc.fill_parser(pre_proc_parser)

    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_cml_args()

    # each subprogram is called with its args, filled in by the fill_parser function in each of the modules,
    # according to which subprogram was specified in the cml (e.g., python run_pre_proc.py Tabular [args...])
    if args.subprogram == 'Tabular':
        read_in_tabular.main(args)

    if args.subprogram == 'pre_proc':
        main_pre_proc.main(args)
