from experiments import *
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()


    setup_end2end_L = {
        'n': [5],
        'L': np.linspace(2, 3, 2, dtype=int),
        'tau': [0.1],
        't_len': [10],
        'samples_per_chain': [500],
        'seed': range(2),
    }

    plot_end2end_L(setup_end2end_L)


