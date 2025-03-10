import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
from itertools import product
from utils.save_load import load


def load_acc(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "accuracy.pt"))


def load_time(exp, cs=False):
    return np.cumsum(load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "est_time.pt")))[
           ::config.EVAL_DISP_INTERVAL]


def load_ms(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "model_size.pt"))


if __name__ == "__main__":
    for dataset_name, client_sel in product(["CIFAR10", "FEMNIST"], [False, True]):
        if dataset_name == "CIFAR10":
            import configs.cifar10 as config

            time_lim = (-9000, 900000)
            acc_lim = (0, 0.9)
            lottery_ticket_acc_lim = (0, 0.9)
        elif dataset_name == "FEMNIST":
            import configs.femnist as config

            time_lim = (-1000, 100000)
            acc_lim = (0, 0.9)
            lottery_ticket_acc_lim = (0, 0.9)
        else:
            raise RuntimeError("Dataset not supported")

        result_path = join("results", config.EXP_NAME)
        if not os.path.isdir(f"results/{config.EXP_NAME}/figs"):
            os.makedirs(f"results/{config.EXP_NAME}/figs")
        fig_path = join(result_path, "figs")

        # Training
        for exp_name in ["conventional", "adaptive", "snip", "online", "iterative", "HeteroPrune"]:
            try:
                acc = load_acc(exp_name, client_sel)
                time = load_time(exp_name, client_sel)

                plt.plot(time, acc, linewidth=1)
            except FileNotFoundError:
                print(f"Skipping training results for {dataset_name}, {exp_name}. Client selection = {client_sel}.")

        plt.xlabel(r"Time (s)")
        plt.ylabel("Test Accuracy")
        plt.xlim(time_lim)
        plt.ylim(acc_lim)

        plt.grid(linestyle="--", color='black', lw='0.5', alpha=0.5)

        plt.legend(("0.2", "0.5", "0.8", "1"),
                   frameon=False, loc="center right")
        plt.savefig(join(fig_path, "training{}".format("_cs" if client_sel else "")), dpi=300)
        plt.close()

        # Model size
        total_num_params = None
        num_pre_rounds = 0

        for exp_name in ["conventional", "adaptive", "snip", "online", "iterative", "HeteroPrune"]:
            try:
                ms = load_ms(exp_name, client_sel)
                if exp_name == "conventional":
                    total_num_params = ms[0]
                ms.insert(0, total_num_params)

                if exp_name == "adaptive":
                    num_pre = len(ms) - config.MAX_ROUND
                    new_adaptive_ms = [ms for ms in ms[:num_pre:config.NUM_LOCAL_UPDATES]]
                    num_pre_rounds = len(new_adaptive_ms)
                    new_adaptive_ms.extend(ms[num_pre:])
                    plt.plot([i - num_pre_rounds for i in range(len(new_adaptive_ms))], np.array(new_adaptive_ms) / 1e6,
                             linewidth=1)
                else:
                    plt.plot(np.array(ms) / 1e6, linewidth=1)
            except FileNotFoundError:
                print(f"Skipping model size results for {dataset_name}, {exp_name}. Client selection = {client_sel}.")

        plt.xlim((-num_pre_rounds - 10, config.MAX_ROUND + 10))
        plt.axvline(x=0., linestyle="--", color='black', lw='0.5')
        plt.legend(("Conventional FL", "PruneFL", "SNIP", "Online Learning", "Iterative Pruning", "HeteroPrune"),
                   frameon=False, loc="center right")
        plt.xlabel("Round")
        plt.ylabel("Number of Parameters ($\\times10^6$)")
        plt.savefig(join(fig_path, "model_size{}".format("_cs" if client_sel else "")), dpi=300)
        plt.close()