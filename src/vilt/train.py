import os 
import argparse




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("base-dir", type=str, help="base data directory wrt to dataset")
    parser.add_argument("--image-dir", type=str, default='images')
    parser.add_argument("--train-file", type=str, default='data_train.csv')
    parser.add_argument("--eval-file", type=str, default='data_eval.csv')
    parser.add_argument("--answer-space", type=str, default='answer_space.txt')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cfg = parse_arguments()

    assert os.path.exists(cfg.base_dir), f"{cfg.base_dir} does not exists"
    
    image_dir = os.path.join(cfg.base_dir, cfg.image_dir)
    train_file = os.path.join(cfg.base_dir, cfg.train_file)
    eval_file = os.path.join(cfg.base_dir, cfg.eval_file)
    answer_space_file = os.path.join(cfg.base_dir, cfg.answer_space)
    

