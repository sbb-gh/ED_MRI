from jofsto_code import jofsto_main, utils

if __name__ == "__main__":

    parser = jofsto_main.return_argparser()
    args = parser.parse_args()
    args = utils.load_yaml(args.cfg)

    jofsto_main.run(args)
