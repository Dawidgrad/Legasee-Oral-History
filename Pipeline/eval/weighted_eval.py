import wevaluation
import argparse

def main(args):
    wevaluation.weighted_eval(args.directory, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a directory of text files')
    parser.add_argument('-d', '--directory', help='Directory to evaluate', required=True)
    parser.add_argument('-o', '--output', help='Output directory', required=True)

    args = parser.parse_args()
    main(args)