import argparse

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=0.1, help='learning rate', type=float)
    parser.add_argument('--train_batch_size', default=32, help='batch size', type=int)
    parser.add_argument('--test_batch_size', default=512, help='batch size', type=int)
    parser.add_argument('--epochs', default=10, help='number of epochs', type=int)
    parser.add_argument('--print_step', default=1000, help='step size for print log', type=int)

    parser.add_argument('--dataset_dir', default='/data/private/Ad/amazon/np_prepro/', help='dataset path')
    parser.add_argument('--model_path', default='./models/', help='model load path', type=str)
    parser.add_argument('--log_path', default='./logs/', help='log path fot tensorboard', type=str)
    parser.add_argument('--is_reuse', default=False)
    parser.add_argument('--multi_gpu', default=False)

    parser.add_argument('--user_count', default=192403, help='number of users', type=int)
    parser.add_argument('--item_count', default=63001, help='number of items', type=int)
    parser.add_argument('--cate_count', default=801, help='number of categories', type=int)

    parser.add_argument('--user_dim', default=128, help='dimension of user', type=int)
    parser.add_argument('--item_dim', default=64, help='dimension of item', type=int)
    parser.add_argument('--cate_dim', default=64, help='dimension of category', type=int)

    parser.add_argument('--dim_layers', default=[80,40,1], type=int)

    args = parser.parse_args()

    return args
