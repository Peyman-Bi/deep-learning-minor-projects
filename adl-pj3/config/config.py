
def make_config():
    import argparse
    parser = argparse.ArgumentParser(description='Model Configuration')
    parser.add_argument('--lr', type=float, default=.001, help='learning rate')
    parser.add_argument('--n_epochs', type=int, help='number of epochs')
    args = parser.parse_args()
    return args