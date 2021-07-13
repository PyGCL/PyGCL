import argparse
import asyncio
import os.path as osp
from enum import Enum
from typing import List

import os

from GPUBurner.workhouse import WorkHouse, Worker, Job, Command


class Objective(Enum):
    InfoNCE = 'InfoNCE'
    JSD = 'JSD'
    Triplet = 'TM'
    Mixup = 'Mixup'
    BL = 'BL'
    BarlowTwins = 'BarlowTwins'
    VICReg = 'VICReg'


class Mode(Enum):
    LocalLocal = 'LL'
    LocalGlobal = 'LG'
    GlobalGlobal = 'GG'


class TaskType(Enum):
    NodeTask = 'Node'
    GraphTask = 'Graph'


class NumberOfAugment:
    Once = True
    Twice = False


def resolve_param_path(dataset: str, mode: Mode, objective: Objective):
    dataset_param = {
        'WikiCS': 'wikics',
        'Amazon-Computers': 'amazon_computers',
        'Amazon-Photo': 'amazon_photo',
        'Coauthor-CS': 'coauthor_cs',
        'Coauthor-Phy': 'coauthor_phy',
        'PROTEINS': 'proteins',
        'PTC_MR': 'ptc_mr',
        'REDDIT-BINARY': 'reddit_binary',
        'IMDB-MULTI': 'imdb_multi',
    }
    mode_dir = {
        Mode.LocalLocal: 'GRACE',
        Mode.LocalGlobal: 'MVGRL',
        Mode.GlobalGlobal: 'GlobalGRACE',
    }
    objective_suffix = {
        Objective.InfoNCE: '',
        Objective.Mixup: '',
        Objective.JSD: '_jsd',
        Objective.Triplet: '_triplet',
        Objective.BL: '_bl',
        Objective.BarlowTwins: '_bt',
        Objective.VICReg: '_vicreg'
    }
    param_dir = mode_dir[mode]
    param_name = dataset_param[dataset]
    param_suffix = objective_suffix[objective]
    return osp.join('./params', param_dir, f'{param_name}{param_suffix}.json')


def resolve_task_type(dataset: str) -> TaskType:
    if dataset in {'WikiCS', 'Amazon-Computers', 'Amazon-Photo', 'Coauthor-CS', 'Coauthor-Phy'}:
        return TaskType.NodeTask
    else:
        return TaskType.GraphTask


class GCLJob(Job):
    def __init__(self,
                 dataset: str,
                 mode: Mode, objective: Objective,
                 topo_aug: str, feat_aug: str,
                 repeat_id: int,
                 param_path: str = None,
                 exp_name: str = 'default',
                 **kwargs):
        super(GCLJob, self).__init__()
        self.mode = mode
        self.objective = objective
        self.dataset = dataset
        self.topo_aug = topo_aug
        self.feat_aug = feat_aug
        self.aug1 = self.topo_aug + '+' + self.feat_aug
        self.aug2 = self.aug1
        self.params = kwargs
        self.param_path = resolve_param_path(dataset, mode, objective) if param_path is None else param_path
        self.task_type = resolve_task_type(dataset)
        self.log_dir = osp.join('./exp', f'bench-{exp_name}')
        os.makedirs(self.log_dir, exist_ok=True)
        args_desc = '@'.join(f'{k}={v}' for k, v in kwargs.items())
        self.log_path = \
            osp.join(
                self.log_dir,
                f'{dataset}@{mode.value}@{objective.value}@{topo_aug}@{feat_aug}@{args_desc}@{repeat_id}.log')

    def resolve(self, worker_id: int) -> List[Command]:
        objective_name = {
            Objective.InfoNCE: 'nt_xent',
            Objective.Mixup: 'mixup',
            Objective.JSD: 'jsd',
            Objective.Triplet: 'triplet',
            Objective.BarlowTwins: 'barlow_twins',
            Objective.VICReg: 'vicreg',
        }
        train_script = {
            TaskType.NodeTask: {
                Mode.LocalLocal: 'train_node_l2l.py',
                Mode.LocalGlobal: 'train_node_g2l.py'
            },
            TaskType.GraphTask: {
                Mode.LocalLocal: 'train_graph_l2l.py',
                Mode.LocalGlobal: 'train_graph_g2l.py',
                Mode.GlobalGlobal: 'train_graph_g2g.py',
            }
        }
        train_script_bl = {
            TaskType.NodeTask: {
                Mode.LocalLocal: 'train_node_BGRL_l2l_norm.py',
                Mode.LocalGlobal: 'train_node_BGRL_g2l.py',
            },
            TaskType.GraphTask: {
                Mode.LocalLocal: 'train_graph_BGRL_l2l.py',
                Mode.LocalGlobal: 'train_graph_BGRL_g2l.py',
                Mode.GlobalGlobal: 'train_graph_BGRL_g2g.py'
            }
        }
        additional_params = []
        for k, v in self.params.items():
            additional_params.append(f'--{k}')
            additional_params.append(f'{v}')

        if self.objective != Objective.BL:
            return [
                Command(
                    exe="python",
                    args=[
                        train_script[self.task_type][self.mode],
                        "--dataset", self.dataset,
                        "--param_path", self.param_path,
                        "--loss", objective_name[self.objective],
                        "--aug1", self.aug1, "--aug2", self.aug2,
                        *additional_params,
                        "--device", f"cuda:{args.gpus[worker_id % len(args.gpus)]}"
                    ],
                    stdout_redirect=self.log_path
                )
            ]
        else:
            return [
                Command(
                    exe="python",
                    args=[
                        train_script_bl[self.task_type][self.mode],
                        "--dataset", self.dataset,
                        "--param_path", self.param_path,
                        "--aug1", self.aug1, "--aug2", self.aug2,
                        *additional_params,
                        "--device", f"cuda:{args.gpus[worker_id % len(args.gpus)]}"
                    ],
                    stdout_redirect=self.log_path
                )
            ]


registered_runners = dict()


def register_runner(func):
    registered_runners[func.__name__] = func
    return func


@register_runner
def ll_obj_ablation_ptc_mr():
    jobs = []

    for i in range(10):
        for objective in [Objective.InfoNCE, Objective.JSD, Objective.Triplet]:
            jobs.append(GCLJob('PTC_MR', Mode.LocalLocal, objective, 'ER', 'FM', i, exp_name='ll_obj_ablation'))

    return jobs


@register_runner
def lg_obj_ablation_ptc_mr():
    jobs = []

    for i in range(10):
        for objective in [Objective.InfoNCE, Objective.JSD, Objective.Triplet]:
            jobs.append(GCLJob('PTC_MR', Mode.LocalGlobal, objective, 'ER', 'FM', i, exp_name='lg_obj_ablation'))

    return jobs


@register_runner
def obj_mode_ablation():
    jobs = []

    datasets = [
        'PTC_MR',
        'PROTEINS',
        'IMDB-MULTI',
    ]
    modes = [
        Mode.LocalLocal,
        Mode.LocalGlobal,
        Mode.GlobalGlobal,
    ]
    objectives = [
        Objective.InfoNCE,
        Objective.JSD,
        Objective.Triplet,
    ]
    for i in range(4):
        for obj in objectives:
            for mode in modes:
                for dataset in datasets:
                    jobs.append(GCLJob(dataset, mode, obj, 'ER', 'FM', i, exp_name='obj_mode_ablation'))

    return jobs


@register_runner
def ll_taug_ptc_mr():
    jobs = []
    topo_augs = [
        'ORI',
        'EA',
        'ER',
        'EA+ER',
        'ND',
        'PPR',
        'MKD',
        'RWS',
    ]
    for i in range(10):
        for topo_aug in topo_augs:
            jobs.append(GCLJob('PTC_MR', Mode.LocalLocal, Objective.InfoNCE, topo_aug, 'ORI', i, exp_name='ll_taug_ablation'))

    return jobs


@register_runner
def mixup_s_wiki():
    jobs = []
    ss = [120, 150, 180, 200]
    for i in range(5):
        for s in ss:
            jobs.append(
                GCLJob(
                    'WikiCS',
                    Mode.LocalLocal, Objective.Mixup,
                    'ER', 'FM',
                    i, exp_name='mixup_s_ablation', mixup_threshold=0.2, mixup_s=s))
    return jobs


@register_runner
def mixup_threshold_s_imdb_multi():
    jobs = []
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.9]
    ss = [10, 20, 50, 100, 120, 150, 180, 200]

    for i in range(5):
        for threshold in thresholds:
            for s in ss:
                jobs.append(
                    GCLJob(
                        'IMDB-MULTI',
                        Mode.GlobalGlobal, Objective.Mixup,
                        'ER', 'FM',
                        i,
                        param_path='params/GlobalGRACE/imdb_multi_triplet.json',
                        exp_name='mixup_threshold_s_ablation', mixup_threshold=threshold, mixup_s=s)
                )

    return jobs


@register_runner
def coauthor_phy():
    jobs = []

    topo_augs = ['ORI', 'EA', 'ER', 'EA+ER', 'ND', 'RWS']
    feat_augs = ['ORI']
    for i in range(3):
        for topo_aug in topo_augs:
            for feat_aug in feat_augs:
                jobs.append(
                    GCLJob(
                        'Coauthor-Phy',
                        Mode.LocalLocal, Objective.InfoNCE,
                        topo_aug, feat_aug,
                        i, exp_name='coauthor_phy'))

    topo_augs = ['ER', 'ND']
    feat_augs = ['FM', 'FD']
    for i in range(3):
        for topo_aug in topo_augs:
            for feat_aug in feat_augs:
                jobs.append(
                    GCLJob(
                        'Coauthor-Phy',
                        Mode.LocalLocal, Objective.InfoNCE,
                        topo_aug, feat_aug,
                        i, exp_name='coauthor_phy'))

    return jobs


@register_runner
def mixup_threshold_s_computers():
    jobs = []
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
    ss = [10, 20, 50, 100, 120, 150, 180, 200]
    for i in range(3):
        for threshold in thresholds:
            for s in ss:
                jobs.append(
                    GCLJob(
                        'Amazon-Computers',
                        Mode.LocalLocal, Objective.Mixup,
                        'ER', 'FM',
                        i, exp_name='mixup_threshold_s_ablation', mixup_threshold=threshold, mixup_s=s))
    return jobs


@register_runner
def tau_ablation():
    jobs = []
    taus = [0.05, 0.25, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    for i in range(3):
        for tau in taus:
            # jobs.append(
            #     GCLJob(
            #         'Coauthor-CS',
            #         Mode.LocalLocal, Objective.InfoNCE,
            #         'ER', 'FM',
            #         i, exp_name='tau_ablation', tau=tau))
            jobs.append(
                GCLJob(
                    'PROTEINS',
                    Mode.LocalLocal, Objective.InfoNCE,
                    'ER', 'FM',
                    i, exp_name='tau_ablation', tau=tau))
    return jobs


@register_runner
def obj_mode_reddit_binary():
    jobs = []

    modes = [
        Mode.LocalLocal,
        Mode.LocalGlobal,
        Mode.GlobalGlobal,
    ]
    objectives = [
        Objective.InfoNCE,
        Objective.JSD,
        Objective.Triplet,
    ]
    for i in range(1):
        for obj in objectives:
            for mode in modes:
                jobs.append(
                    GCLJob(
                        'REDDIT-BINARY',
                        mode, obj,
                        'ER', 'FM', i,
                        param_path='./params/GRACE/reddit_binary.json',
                        exp_name='obj_mode_ablation'))

    return jobs


@register_runner
def bilevel_aug_ptc_mr():
    jobs = []
    topo_augs = ['ORI', 'ER', 'ND']
    feat_augs = ['FM', 'FD']
    for i in range(3):
        for topo_aug in topo_augs:
            for feat_aug in feat_augs:
                jobs.append(
                    GCLJob(
                        'PTC_MR',
                        Mode.LocalLocal, Objective.InfoNCE,
                        topo_aug, feat_aug,
                        i, exp_name='bilevel_aug'))

    topo_augs = ['PPR', 'MKD']
    feat_augs = ['ER', 'FD']
    for i in range(3):
        for topo_aug in topo_augs:
            for feat_aug in feat_augs:
                jobs.append(
                    GCLJob(
                        'PTC_MR',
                        Mode.LocalLocal, Objective.InfoNCE,
                        topo_aug, feat_aug,
                        i, exp_name='bilevel_aug'))

    return jobs


@register_runner
def er_ea_sensitivity():
    jobs = []

    probs = [0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8]
    for i in range(10):
        for prob in probs:
            jobs.append(GCLJob('Coauthor-CS', Mode.LocalLocal, Objective.InfoNCE, 'ND', 'FM', i,
                               exp_name='ER_EA_ablation',
                               drop_node_prob1=prob,
                               drop_node_prob2=prob))
            jobs.append(GCLJob('Coauthor-CS', Mode.LocalLocal, Objective.InfoNCE, 'ER', 'FM', i,
                               exp_name='ER_EA_ablation',
                               drop_edge_prob1=prob,
                               drop_edge_prob2=prob))
            jobs.append(GCLJob('Coauthor-CS', Mode.LocalLocal, Objective.InfoNCE, 'EA', 'FM', i,
                               exp_name='ER_EA_ablation',
                               add_edge_prob1=prob,
                               add_edge_prob2=prob))

    return jobs


@register_runner
def bl_loss_graph():
    jobs = []

    datasets = ['IMDB-MULTI', 'PTC_MR', 'PROTEINS', 'REDDIT-BINARY']
    modes = [Mode.LocalLocal, Mode.LocalGlobal]
    for i in range(10):
        for dataset in datasets:
            for mode in modes:
                jobs.append(GCLJob(dataset, mode, Objective.BL, 'ER', 'FM', i,
                                   exp_name='bl_loss'))

    return jobs


@register_runner
def bilevel_aug_feat_node():
    jobs = []

    datasets = [
        # 'WikiCS', 'Coauthor-CS',
        'Coauthor-Phy',
        # 'Amazon-Computers',
        # 'PTC_MR', 'PROTEINS',
        # 'REDDIT-BINARY',
        # 'IMDB-MULTI'
    ]
    topo_augs = ['EA', 'RWS']
    feat_augs = ['FM', 'FD']
    for i in range(10):
        for dataset in datasets:
            for topo_aug in topo_augs:
                for feat_aug in feat_augs:
                    jobs.append(
                        GCLJob(
                            dataset,
                            Mode.LocalLocal, Objective.InfoNCE,
                            topo_aug, feat_aug,
                            i, exp_name='rebuttal_bilevel_aug'
                        )
                    )

    return jobs


@register_runner
def bilevel_aug_feat_graph():
    jobs = []

    datasets = [
        # 'WikiCS', 'Coauthor-CS',
        # 'Coauthor-Phy',
        # 'Amazon-Computers',
        # 'PTC_MR', 'PROTEINS',
        'REDDIT-BINARY',
        # 'IMDB-MULTI'
    ]
    topo_augs = ['EA', 'RWS']
    feat_augs = ['FM', 'FD']
    for i in range(10):
        for dataset in datasets:
            for topo_aug in topo_augs:
                for feat_aug in feat_augs:
                    jobs.append(
                        GCLJob(
                            dataset,
                            Mode.LocalLocal, Objective.InfoNCE,
                            topo_aug, feat_aug,
                            i, exp_name='bi-level-augmentations'
                        )
                    )

    return jobs


@register_runner
def compositional_random_augmentations():
    jobs = []

    datasets = [
        'WikiCS', 'Coauthor-CS', 'Amazon-Computers',
        'PTC_MR', 'PROTEINS', 'REDDIT-BINARY', 'IMDB-MULTI'
    ]
    base_augs = ['PPR', 'MKD']
    stoc_augs = ['EA', 'ND']
    for i in range(10):
        for dataset in datasets:
            for base_aug in base_augs:
                for stoc_aug in stoc_augs:
                    jobs.append(
                        GCLJob(
                            dataset,
                            Mode.LocalLocal, Objective.InfoNCE,
                            base_aug, stoc_aug,
                            i, exp_name='compositional-random-augmentations'
                        )
                    )

    return jobs


@register_runner
def bl_batch_norm_ablation():
    jobs = []

    norms = ['none', 'batch']

    for i in range(10):
        for encoder_norm in norms:
            for projector_norm in norms:
                for predictor_norm in norms:
                    job = GCLJob(
                        'WikiCS',
                        Mode.LocalLocal, Objective.BL,
                        topo_aug='FM', feat_aug='ER',
                        repeat_id=i, exp_name='bl-batch-norm-ablation',
                        encoder_norm=encoder_norm,
                        projector_norm=projector_norm,
                        predictor_norm=predictor_norm
                    )
                    jobs.append(job)

                    job = GCLJob(
                        'Amazon-Computers',
                        Mode.LocalLocal, Objective.BL,
                        topo_aug='FM', feat_aug='ER',
                        repeat_id=i, exp_name='bl-batch-norm-ablation',
                        encoder_norm=encoder_norm,
                        projector_norm=projector_norm,
                        predictor_norm=predictor_norm
                    )
                    jobs.append(job)

    return jobs


@register_runner
def node_bt_vicreg():
    jobs = []

    datasets = ['WikiCS', 'Amazon-Computers', 'Coauthor-CS', 'Coauthor-Phy']
    objectives = [Objective.BarlowTwins, Objective.VICReg]

    for i in range(10):
        for dataset in datasets:
            for objective in objectives:
                job = GCLJob(
                    dataset,
                    Mode.LocalLocal, objective,
                    'FM', 'ER',
                    i, exp_name='node-bt-vicreg'
                )
                jobs.append(job)

    return jobs


@register_runner
def graph_l2l_bt():
    jobs = []

    datasets = ['PTC_MR', 'PROTEINS', 'REDDIT-BINARY', 'IMDB-MULTI']
    for i in range(10):
        for dataset in datasets:
            job = GCLJob(
                dataset,
                Mode.LocalLocal, Objective.BarlowTwins,
                'FM', 'ER',
                i, exp_name='graph-l2l-bt'
            )
            jobs.append(job)

    return jobs


@register_runner
def graph_l2l_vicreg():
    jobs = []

    datasets = ['PTC_MR', 'PROTEINS', 'REDDIT-BINARY', 'IMDB-MULTI']
    for i in range(10):
        for dataset in datasets:
            job = GCLJob(
                dataset,
                Mode.LocalLocal, Objective.VICReg,
                'FM', 'ER',
                i, exp_name='graph-l2l-vicreg'
            )
            jobs.append(job)

    return jobs


@register_runner
def graph_g2g_bt_vicreg():
    jobs = []

    datasets = ['PTC_MR', 'PROTEINS', 'REDDIT-BINARY', 'IMDB-MULTI']
    for i in range(10):
        for dataset in datasets:
            job = GCLJob(
                dataset,
                Mode.GlobalGlobal, Objective.BarlowTwins,
                'FM', 'ER',
                i, exp_name='graph-g2g-bt-vicreg'
            )
            jobs.append(job)
            job = GCLJob(
                dataset,
                Mode.GlobalGlobal, Objective.VICReg,
                'FM', 'ER',
                i, exp_name='graph-g2g-bt-vicreg'
            )
            jobs.append(job)

    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=lambda x: [int(i) for i in x.split(',')], default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--parallel', type=int, default=None)
    parser.add_argument('--exp', type=str, default='hello')
    args = parser.parse_args()

    if args.parallel is None:
        args.parallel = len(args.gpus)

    runner = registered_runners[args.exp]
    jobs = runner()

    workhouse = WorkHouse(num_workers=args.parallel, jobs=jobs)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(workhouse.spawn_workers())
