import random
from train.parse_args import parse_SASGFRec_args
from utility.log_helper import *
from utility.metrics import *
from utility.dao_helper import *
from model.SASGFRec import SASGFRec
from dao.SeqGFRec_dataloader import FeatureGen
from dao.load_test_data import load_movie_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    df, kg_df = load_movie_data(args.corpus_name, kg=True)
    featureGen = FeatureGen(df, kg_df, input_max_length=args.maxlen, fan_outs=args.fan_outs, device=args.device)
    loader_train, loader_val, loader_kg = featureGen.prepare_loader(df, kg_df, batch_size=args.batch_size, valid_batch_size=args.valid_batch_size, kg_batch_size=args.kg_batch_size)
    model = SASGFRec(featureGen.num_users, featureGen.num_nodes, featureGen.num_relations, args)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    model.fit(loader_train, loader_val, loader_kg, adam_optimizer)


if __name__ == '__main__':
    args = parse_SASGFRec_args()
    train(args)
