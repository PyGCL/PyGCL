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
        }
        train_script = {
            TaskType.NodeTask: {
                Mode.LocalLocal: 'train_node.py',
                Mode.LocalGlobal: 'train_node_mvgrl.py'
            },
            TaskType.GraphTask: {
                Mode.LocalLocal: 'train_graph_GRACE.py',
                Mode.LocalGlobal: 'train_graph_mvgrl.py',
                Mode.GlobalGlobal: 'train_graph_g2g.py',
            }
        }
        additional_params = []
        for k, v in self.params.items():
            additional_params.append(f'--{k}')
            additional_params.append(f'{v}')
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
    taus = [0.05, 3]
    for i in range(3):
        for tau in taus:
            jobs.append(
                GCLJob(
                    'Coauthor-CS',
                    Mode.LocalLocal, Objective.InfoNCE,
                    'ER', 'FM',
                    i, exp_name='tau_ablation', tau=tau))
            # jobs.append(
            #     GCLJob(
            #         'IMDB-MULTI',
            #         Mode.LocalLocal, Objective.InfoNCE,
            #         'ER', 'FM',
            #         i, exp_name='tau_ablation', tau=tau))
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
