import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, Data
import json
from cal_TSNE import calculate_TSNE, plot_statics
import math
import copy

from sklearn.metrics import average_precision_score, roc_auc_score

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                   'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--skip', type=int, default=1, help='sample the training dataset')
parser.add_argument('--random_sample', action='store_true', help='random sample the training dataset')
parser.add_argument('--id', type=int, default=0, help='no. of experiment')
parser.add_argument('--sample_type', type=str, default=None, help='sample type:[uniform, tbatch]')
parser.add_argument('--refresh_history_cache', action='store_true',
                    help='Whether to augment the model with the Refresh history cache')
parser.add_argument('--interval', type=int, default=10000, help='no. of experiment')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--visual_att', action='store_true')


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
interval = args.interval

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.data}-{args.prefix}-sample{args.skip}-{args.id}.pth' if args.skip != 1 \
    else f'./saved_models/{args.data}-{args.prefix}.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.data}-{args.prefix}-sample{args.skip}-{epoch}-{args.id}.pth' if args.skip != 1 \
    else f'./saved_checkpoints/{args.data}-{args.prefix}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes,
                              randomize_features=args.randomize_features,
                              sample=args.skip, random_sample=args.random_sample, exp_id=args.id,
                              sample_type=args.sample_type)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# =========================================================================================#
val_pred_wrong_list = {}
test_pred_wrong_list = {'1': [], '2': [], '3': [], '4': [], '5': []}
nn_val_pred_wrong_list = {}
nn_test_pred_wrong_list = {'1': [], '2': [], '3': [], '4': [], '5': []}
v_list = {}
for v in train_data.sources:
    if v not in v_list:
        v_list[v] = 1
    else:
        v_list[v] += 1
sort_v_list = np.array(sorted(v_list.items(), key=lambda item: item[1]))
average_degree = sort_v_list[:, 1].mean()
average_v_list = sort_v_list[np.logical_and(sort_v_list[:, 1] > math.floor(average_degree - 1),
                                            sort_v_list[:, 1] < math.floor(average_degree + 1))]

small_degree_idx = np.sort(np.where(train_data.sources == sort_v_list[:20, 0][:, None])[-1])
large_degree_idx = np.sort(np.where(train_data.sources == sort_v_list[-20:, 0][:, None])[-1])
average_degree_idx = np.sort(np.where(train_data.sources == average_v_list[:, 0][:, None])[-1])

small_degree_data = Data(train_data.sources[small_degree_idx], train_data.destinations[small_degree_idx],
                         train_data.timestamps[small_degree_idx], train_data.edge_idxs[small_degree_idx],
                         train_data.labels[small_degree_idx])
small_negative_edge = np.setdiff1d(train_data.destinations, small_degree_data.destinations, assume_unique=False)
small_neg_size = small_negative_edge.size

large_degree_data = Data(train_data.sources[large_degree_idx], train_data.destinations[large_degree_idx],
                         train_data.timestamps[large_degree_idx], train_data.edge_idxs[large_degree_idx],
                         train_data.labels[large_degree_idx])
large_negative_edge = np.setdiff1d(train_data.destinations, large_degree_data.destinations, assume_unique=False)
large_neg_size = large_negative_edge.size

average_degree_data = Data(train_data.sources[average_degree_idx], train_data.destinations[average_degree_idx],
                           train_data.timestamps[average_degree_idx], train_data.edge_idxs[average_degree_idx],
                           train_data.labels[average_degree_idx])
avg_negative_edge = np.setdiff1d(train_data.destinations, average_degree_data.destinations, assume_unique=False)
avg_neg_size = avg_negative_edge.size
# =========================================================================================#


for i in range(args.n_runs):
    results_path = "results/{}_{}/result_{}.pkl".format(args.data, args.prefix, i)
    parent_path = "results/{}_{}/".format(args.data, args.prefix)
    Path(parent_path).mkdir(parents=True, exist_ok=True)

    # Initialize Model
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep, refresh_history_cache=args.refresh_history_cache)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    val_aps = []

    new_nodes_val_f1s = []
    val_f1s = []

    new_nodes_val_accs = []
    val_accs = []

    new_nodes_val_spes = []
    val_spes = []

    new_nodes_val_sens = []
    val_sens = []

    epoch_times = []
    total_epoch_times = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    # wandb.watch(tgn, log="all")
    for epoch in range(NUM_EPOCH):
        train_ap = []
        train_auc = []
        start_epoch = time.time()
        ### Training

        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        # Train using only training graph
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))
        timestamps = train_data.timestamps
        max_timestamps = np.max(timestamps)
        interval_num = math.ceil(max_timestamps / interval)

        # for k in range(interval_num):
        #     loss = 0
        #     optimizer.zero_grad()
        #
        #     for j in range(args.backprop_every):
        #         cur_idx = np.logical_and(timestamps >= k * interval, timestamps < (k + 1) * interval)
        #         sources_batch, destinations_batch = train_data.sources[cur_idx], \
        #                                             train_data.destinations[cur_idx]
        #         edge_idxs_batch = train_data.edge_idxs[cur_idx]
        #         timestamps_batch = train_data.timestamps[cur_idx]

        for k in range(0, num_batch, args.backprop_every):
            loss = 0
            optimizer.zero_grad()

            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args.backprop_every):
                batch_idx = k + j

                if batch_idx >= num_batch:
                    continue

                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]

                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)  # sample negatives

                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)

                tgn = tgn.train()
                pos_prob, neg_prob, h, _ = tgn.compute_edge_probabilities(
                    sources_batch, destinations_batch,
                    negatives_batch,
                    timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS, epoch=epoch, batch=k)

                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

                if epoch == 51:
                    tgn.embedding_module.neighbor_finder = full_ngh_finder
                    test_ap, test_auc, test_f1, test_acc, test_spe, test_sen, test_wrong_set = eval_edge_prediction(
                        model=tgn,
                        negative_edge_sampler=test_rand_sampler,
                        data=test_data,
                        n_neighbors=NUM_NEIGHBORS,
                        split='small')
                    tgn.embedding_module.neighbor_finder = train_ngh_finder
                    train_ap.append(test_ap)
                    train_auc.append(test_auc)

            loss /= args.backprop_every

            # loss = loss.detach_().requires_grad_(True)
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            if USE_MEMORY:
                tgn.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        mean_meomry = []
        variance_memory = []
        std_memory = []
        cos_similarity_list = []
        step = 1

        if epoch == 0:
            for num in range(step, len(tgn.memory_list)):
                last_memory = tgn.memory_list[num - step]
                cur_memory = tgn.memory_list[num]
                # cos_similarity = torch.mean(
                #     torch.cosine_similarity(cur_memory.cpu().unsqueeze(1), cur_memory.cpu().unsqueeze(0), dim=-1))
                cos_similarity = torch.mean(torch.cosine_similarity(last_memory, cur_memory, dim=-1))
                cos_similarity_list.append(cos_similarity.cpu().detach().numpy())
                # if num % 45 == 0:
                #     calculate_TSNE(cur_memory, 'memory', num)
            for num in range(len(tgn.memory_list)):
                cur_memory = tgn.memory_list[num]
                mean_meomry.append(torch.mean(torch.mean(cur_memory.cpu(), dim=0)).detach().numpy())
                variance_memory.append(torch.var(cur_memory.cpu(), unbiased=True).detach().numpy())
                std_memory.append(torch.std(cur_memory.cpu()).detach().numpy())

            # tgn.memory_list = []
            # mean_meomry = np.stack(mean_meomry)
            # mean_meomry = calculate_TSNE(mean_meomry, n=1)
            # x_min, x_max = np.min(mean_meomry, 0), np.max(mean_meomry, 0)
            # mean_meomry = (mean_meomry - x_min) / (x_max - x_min)

            # plot_statics(train_ap, 'ap' + str(BATCH_SIZE))
            plot_statics(mean_meomry, 'mean' + str(BATCH_SIZE))
            plot_statics(cos_similarity_list, 'cosine similarity' + str(BATCH_SIZE))
            plot_statics(variance_memory, 'variance' + str(BATCH_SIZE))
            plot_statics(std_memory, 'standard deviation' + str(BATCH_SIZE))

            # plot_statics(train_ap, 'ap' + str(interval))
            # plot_statics(mean_meomry, 'mean' + str(interval))
            # plot_statics(cos_similarity_list, 'cosine similarity' + str(interval))
            # plot_statics(variance_memory, 'variance' + str(interval))
            # plot_statics(std_memory, 'standard deviation' + str(interval))

        # train_rand_sampler.seed = 1

        # small_ap, small_auc, small_ap_pos, small_ap_neg = eval_edge_prediction(
        #     model=tgn,
        #     negative_edge_sampler=train_rand_sampler,
        #     data=small_degree_data,
        #     n_neighbors=NUM_NEIGHBORS,
        #     split='small',
        #     negative_edge=small_negative_edge)
        #
        # large_ap, large_auc, large_ap_pos, large_ap_neg = eval_edge_prediction(
        #     model=tgn,
        #     negative_edge_sampler=train_rand_sampler,
        #     data=large_degree_data,
        #     n_neighbors=NUM_NEIGHBORS,
        #     split='large',
        #     negative_edge=large_negative_edge)
        #
        # avg_ap, avg_auc, avg_ap_pos, avg_ap_neg = eval_edge_prediction(
        #     model=tgn,
        #     negative_edge_sampler=train_rand_sampler,
        #     data=average_degree_data,
        #     n_neighbors=NUM_NEIGHBORS,
        #     split='avg',
        #     negative_edge=avg_negative_edge)
        # train_rand_sampler.seed = None

        ### Validation
        # Validation uses the full graph
        tgn.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = tgn.memory.backup_memory()

        val_ap, val_auc, val_f1, val_acc, val_spe, val_sen, val_wrong_set = eval_edge_prediction(model=tgn,
                                                                                                 negative_edge_sampler=val_rand_sampler,
                                                                                                 data=val_data,
                                                                                                 n_neighbors=NUM_NEIGHBORS,
                                                                                                 split='val')
        val_pred_wrong_list[str(i + 1) + '-' + str(epoch + 1)] = val_wrong_set

        # wandb.log({
        #     "val_ap": val_ap,
        #     "val_auc": val_auc
        # })

        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            tgn.memory.restore_memory(train_memory_backup)

        # Validate on unseen nodes
        nn_val_ap, nn_val_auc, nn_val_f1, nn_val_acc, nn_val_spe, nn_val_sen, nn_val_wrong_set = eval_edge_prediction(
            model=tgn,
            negative_edge_sampler=val_rand_sampler,
            data=new_node_val_data,
            n_neighbors=NUM_NEIGHBORS,
            split='val_unseen')

        nn_val_pred_wrong_list[str(i + 1) + '-' + str(epoch + 1)] = nn_val_wrong_set

        # wandb.log({
        #     "nn_val_ap": nn_val_ap,
        #     "nn_val_auc": nn_val_auc
        # })

        if USE_MEMORY:
            # Restore memory we had at the end of validation
            tgn.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        new_nodes_val_f1s.append(nn_val_f1)
        val_f1s.append(val_f1)
        new_nodes_val_accs.append(nn_val_acc)
        val_accs.append(val_acc)
        new_nodes_val_spes.append(nn_val_spe)
        val_spes.append(val_spe)
        new_nodes_val_sens.append(nn_val_sen)
        val_sens.append(val_sen)
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "val_f1s": val_f1s,
            "new_nodes_val_f1s": new_nodes_val_f1s,
            "val_accs": val_accs,
            "new_nodes_val_accs": new_nodes_val_accs,
            "val_spes": val_spes,
            "new_nodes_val_spes": new_nodes_val_spes,
            "val_sens": val_sens,
            "new_nodes_val_sens": new_nodes_val_sens,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
            'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        logger.info(
            'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

        # Early stopping
        # if model doesn't improve for 5 epochs, then will stop training and obtain the best model to evaluate in test set.
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgn.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgn.eval()
            break
        else:
            torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    ### Test
    tgn.embedding_module.neighbor_finder = full_ngh_finder
    test_ap, test_auc, test_f1, test_acc, test_spe, test_sen, test_wrong_set = eval_edge_prediction(model=tgn,
                                                                                                    negative_edge_sampler=test_rand_sampler,
                                                                                                    data=test_data,
                                                                                                    n_neighbors=NUM_NEIGHBORS,
                                                                                                    split='test',
                                                                                                    use_tsne=args.tsne,
                                                                                                    visual_att=args.visual_att)
    test_pred_wrong_list[str(i + 1)] = test_wrong_set

    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)

    # wandb.log({
    #     "test_ap": test_ap,
    #     "test_auc": test_auc
    # })

    # Test on unseen nodes
    nn_test_ap, nn_test_auc, nn_test_f1, nn_test_acc, nn_test_spe, nn_test_sen, nn_test_wrong_set = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=nn_test_rand_sampler,
        data=new_node_test_data,
        n_neighbors=NUM_NEIGHBORS,
        split='test_unseen',
        use_tsne=args.tsne,
        visual_att=args.visual_att)
    nn_test_pred_wrong_list[str(i + 1)] = nn_test_wrong_set

    # wandb.log({
    #     "nn_test_ap": nn_test_ap,
    #     "nn_test_auc": nn_test_auc
    # })

    torch.save(tgn.state_dict(), 'model.h5')
    # wandb.save('model.h5')

    logger.info(
        'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
    logger.info(
        'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
    # Save results for this run
    pickle.dump({
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "val_f1s": val_f1s,
        "new_nodes_val_f1s": new_nodes_val_f1s,
        "val_accs": val_accs,
        "new_nodes_val_accs": new_nodes_val_accs,
        "val_spes": val_spes,
        "new_nodes_val_spes": new_nodes_val_spes,
        "val_sens": val_sens,
        "new_nodes_val_sens": new_nodes_val_sens,
        "test_ap": test_ap,
        "new_node_test_ap": nn_test_ap,
        "test_f1": test_f1,
        "new_node_test_f1": nn_test_f1,
        "test_acc": test_acc,
        "new_node_test_acc": nn_test_acc,
        "test_spe": test_spe,
        "new_node_test_spe": nn_test_spe,
        "test_sen": test_sen,
        "new_node_test_sen": nn_test_sen,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    logger.info('Saving TGN model')
    if USE_MEMORY:
        # Restore memory at the end of validation (save a model which is ready for testing)
        tgn.memory.restore_memory(val_memory_backup)
    torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGN model saved')

pred_wrong_dict = {'val': [], 'nn_val': [], 'test': [], 'nn_test': []}
pred_wrong_dict['val'] = val_pred_wrong_list
pred_wrong_dict['test'] = test_pred_wrong_list
pred_wrong_dict['nn_val'] = nn_val_pred_wrong_list
pred_wrong_dict['nn_test'] = nn_test_pred_wrong_list

# pred_wrong_path = "pred_wrong/{}_pred_wrong_negative_new.json".format(args.data)
# pred_wrong_parent_path = "pred_wrong/"
pred_wrong_path = "pred_correct/{}_pred_correct_positive.json".format(args.data)
pred_wrong_parent_path = "pred_correct/"
Path(pred_wrong_parent_path).mkdir(parents=True, exist_ok=True)

# with open(pred_wrong_path, 'w') as file:
#     file.write(json.dumps(pred_wrong_dict))
