import argparse
from src import train, evaluate

def main():
    parser = argparse.ArgumentParser(description="Ellipse Detection Project")
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                        help="Choose 'train' to train the model, or 'eval' to evaluate a saved model.")
    args = parser.parse_args()

    if args.mode == 'train':
        train.train()
    elif args.mode == 'eval':
        evaluate.evaluate()

if __name__ == '__main__':
    main()
